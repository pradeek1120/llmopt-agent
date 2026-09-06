from pathlib import Path
import json
from .models import ProjectConfig

def load_config(path: str | Path) -> ProjectConfig:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    return ProjectConfig.model_validate(data)
