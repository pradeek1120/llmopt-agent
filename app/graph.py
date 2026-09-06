from __future__ import annotations

from langgraph.graph import StateGraph, END

from .agents import AnalysisAgent, OptimizationAgent, EvaluationResearchAgent
from .models import OptimizationState, ProjectConfig
from .tools import benchmark_candidates

analysis_agent = AnalysisAgent()
optimization_agent = OptimizationAgent()
evaluation_agent = EvaluationResearchAgent()

def build_graph(project: ProjectConfig):
    def analyze(state: OptimizationState):
        return {
            "analysis": analysis_agent.run(project),
            "iteration": state.get("iteration", 0) + 1,
        }

    def optimize(state: OptimizationState):
        candidates = optimization_agent.run(project, state["analysis"])
        return {"candidates": [c.model_dump() for c in candidates]}

    def benchmark(state: OptimizationState):
        candidates = state["candidates"]
        from .models import ExperimentConfig
        experiments = [ExperimentConfig.model_validate(c) for c in candidates]
        results = benchmark_candidates(project, experiments)
        return {"results": [r.model_dump() for r in results]}

    def evaluate(state: OptimizationState):
        best, diagnosis, continue_loop = evaluation_agent.run(
            project, state["analysis"], state["results"]
        )
        # V1 stops after a successful candidate to keep the graph simple.
        # The iteration field is retained for the adaptive-search V2.
        return {
            "selected": best,
            "diagnosis": diagnosis,
            "continue_loop": continue_loop and state.get("iteration", 0) < project.max_iterations,
        }

    graph = StateGraph(OptimizationState)
    graph.add_node("analysis", analyze)
    graph.add_node("optimization", optimize)
    graph.add_node("benchmark", benchmark)
    graph.add_node("evaluation", evaluate)

    graph.set_entry_point("analysis")
    graph.add_edge("analysis", "optimization")
    graph.add_edge("optimization", "benchmark")
    graph.add_edge("benchmark", "evaluation")
    graph.add_edge("evaluation", END)

    return graph.compile()
