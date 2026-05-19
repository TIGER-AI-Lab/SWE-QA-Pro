import os
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CONFIGS_DIR = PROJECT_ROOT / "configs"
PROMPTS_DIR = PROJECT_ROOT / "prompts"
MODELS_YAML = CONFIGS_DIR / "models.yaml"

# Load .env from eval/ before any module reads os.environ.
load_dotenv(PROJECT_ROOT / ".env", override=False)


def env(name: str, default: str | None = None) -> str | None:
    return os.environ.get(name, default)
