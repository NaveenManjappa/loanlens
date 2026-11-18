"""Reads prompt files from loanlens/prompts/.

importlib.resources finds the files inside the installed package, so this works from any working
directory and inside the Docker image, not only from the repo root.
"""

from functools import cache
from importlib.resources import files


@cache
def load_prompt(name: str) -> str:
    """The text of prompts/<name>, without the trailing newline."""
    return files("loanlens").joinpath("prompts", name).read_text(encoding="utf-8").strip()
