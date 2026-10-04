import json
import re
from pathlib import Path
from typing import Annotated

from pydantic import Field, TypeAdapter

from intent_to_schedule.adapter.data_model import CommandsData, QueryData

QUERY_ADAPTER: TypeAdapter[QueryData] = TypeAdapter(
    Annotated[QueryData, Field(discriminator="kind")]
)


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


def test_readme_query_examples_match_input_model() -> None:
    """Validate every README query example against the input model."""
    readme: str = (Path(__file__).resolve().parents[1] / "README.md").read_text()
    kinds: set[str] = set()
    block: str
    for block in re.findall(
        r"^```json\s*\n(.*?)^```\s*$", readme, re.MULTILINE | re.DOTALL
    ):
        source: dict[str, object] = json.loads(block)
        if "kind" in source:
            QUERY_ADAPTER.validate_python(source)
        if source.get("kind") in {"evaluation", "objective_policy"}:
            kinds.add(str(source["kind"]))
    assert kinds == {"evaluation", "objective_policy"}
