import json
import re
from pathlib import Path

from intent_to_schedule.adapter.data_model import CommandsData


def test_readme_command_examples_match_input_model() -> None:
    """Validate every command example in the README against the input model."""
    readme: str = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    examples: list[str] = [
        block
        for block in re.findall(
            r"^```json\s*\n(.*?)^```\s*$", readme, re.MULTILINE | re.DOTALL
        )
        if '"commands"' in block
    ]
    assert examples, "README must contain command examples"
    for example in examples:
        CommandsData.model_validate(json.loads(example))
