from dataclasses import dataclass
from pathlib import Path


@dataclass
class Project:
    """
    This class represents the project. It stores useful information about the structure, e.g. paths.
    """

    base_dir: Path = Path(__file__).parents[0]
    inputs_dir = base_dir / "inputs"
    checkpoint_dir = base_dir / "checkpoint"

    def __post_init__(self):
        # create the directories if they don't exist
        self.checkpoint_dir.mkdir(exist_ok=True)
