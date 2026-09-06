from __future__ import annotations

import os
from typing import Any

from .models import ExperimentConfig, ProjectConfig
from .tools import generate_candidates, benchmark_candidates

class AnalysisAgent:
    """Deterministic first-pass analysis. An LLM can be added later."""

    def run(self, project: ProjectConfig) -> dict[str, Any]:
        w = project.workload
        h = project.hardware

        if w.output_tokens > w.prompt_tokens:
            workload_type = "decode_heavy"
        elif w.prompt_tokens > 4 * w.output_tokens:
            workload_type = "prefill_heavy"
        else:
            workload_type = "mixed"

        estimated_weight_memory_fp16_gb = 8.0 * 2  # simplified V1 estimate for an 8B model
        pressure = "high" if w.concurrency >= 32 else "moderate"

        return {
            "model": project.model_name,
            "gpu": h.gpu,
            "workload_type": workload_type,
            "concurrency_pressure": pressure,
            "estimated_fp16_weight_memory_gb": estimated_weight_memory_fp16_gb,
            "initial_hypothesis": (
                "Test lower precision and KV-cache compression while "
                "respecting TTFT and quality constraints."
            ),
        }

class OptimizationAgent:
    """Creates the experiment plan from the analysis."""

    def run(self, project: ProjectConfig, analysis: dict[str, Any]):
        candidates = generate_candidates(project)
        # Prefer configurations that have a chance of improving memory/speed.
        candidates.sort(
            key=lambda x: (
                x.precision == "fp8",
                x.kv_cache_precision == "fp8",
                x.batch_size,
            ),
            reverse=True,
        )
        return candidates

class EvaluationResearchAgent:
    """
    Evaluates benchmark results and diagnoses the next step.
    This is deliberately deterministic in V1 so it is easy to test.
    """

    def run(
        self,
        project: ProjectConfig,
        analysis: dict[str, Any],
        results: list[dict[str, Any]],
    ):
        if not results:
            raise ValueError("No benchmark results were produced.")

        valid = [r for r in results if r["passed_constraints"]]
        pool = valid or results

        # Score: prioritize throughput, then memory efficiency.
        best = max(
            pool,
            key=lambda r: r["throughput_tok_s"] / max(r["vram_gb"], 0.1)
        )

        if not valid:
            diagnosis = (
                "No candidate satisfied all constraints. "
                "The next iteration should relax or refine the search space."
            )
            continue_loop = True
        else:
            diagnosis = (
                "Candidate satisfies the quality/TTFT constraints. "
                "Further search can test whether a better throughput-memory tradeoff exists."
            )
            continue_loop = False

        return best, diagnosis, continue_loop

def optional_llm_reasoning(prompt: str) -> str | None:
    """
    Optional hook for an LLM explanation layer.
    The optimization loop does not depend on it.
    """
    key = os.getenv("OPENAI_API_KEY")
    if not key:
        return None

    try:
        from langchain_openai import ChatOpenAI
        model_name = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
        llm = ChatOpenAI(model=model_name, temperature=0)
        response = llm.invoke(prompt)
        return response.content
    except Exception as exc:
        return f"LLM reasoning unavailable: {exc}"
