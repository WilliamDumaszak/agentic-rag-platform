"""
Prompt version manager.

Loads prompt templates from prompts/vN.yaml files.
Tracks which version is active and computes a hash for audit trail logging.

Why hash prompts?
  When a model gives a wrong answer in production, the first question is:
  "which prompt was used?" The hash ties every response in the audit log
  to an exact prompt version — even if the file was changed afterward.
"""

import hashlib
import logging
from functools import lru_cache
from pathlib import Path

import yaml

logger = logging.getLogger(__name__)

_PROMPTS_DIR = Path(__file__).parent


@lru_cache(maxsize=16)
def load_prompts(version: str = "v2") -> dict:
    """
    Load a prompt version file. Cached — loads each version only once.

    Args:
        version: Version string, e.g. "v1", "v2".

    Returns:
        Dict with prompt templates and version metadata.
    """
    path = _PROMPTS_DIR / f"{version}.yaml"
    if not path.exists():
        logger.warning(f"Prompt version '{version}' not found at {path}. Falling back to v1.")
        path = _PROMPTS_DIR / "v1.yaml"

    with open(path) as f:
        data = yaml.safe_load(f)

    data["_version"] = version
    data["_hash"] = _compute_hash(path)
    logger.info(f"Loaded prompt version={version}, hash={data['_hash'][:8]}")
    return data


def get_prompt(template_name: str, version: str = "v2", **kwargs) -> str:
    """
    Get a rendered prompt template.

    Args:
        template_name: Key in the YAML file (e.g. "rag_user", "summarization").
        version:       Prompt version to use.
        **kwargs:      Variables to substitute into the template.

    Returns:
        Rendered prompt string.

    Example:
        prompt = get_prompt("rag_user", version="v2", context="...", question="...")
    """
    prompts = load_prompts(version)
    template = prompts.get(template_name)

    if template is None:
        raise KeyError(f"Prompt '{template_name}' not found in version '{version}'.")

    if kwargs:
        template = template.format(**kwargs)

    return template


def get_prompt_hash(version: str = "v2") -> str:
    """Return the full SHA-256 hash of a prompt version file. Used in audit logging."""
    prompts = load_prompts(version)
    return prompts["_hash"]


def list_versions() -> list[str]:
    """List all available prompt versions in the prompts directory."""
    return sorted(
        p.stem for p in _PROMPTS_DIR.glob("v*.yaml")
    )


def _compute_hash(path: Path) -> str:
    """Compute SHA-256 hash of a file's content."""
    with open(path, "rb") as f:
        return hashlib.sha256(f.read()).hexdigest()
