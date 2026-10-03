from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal

import openai

from intent_to_schedule.adapter.data_model import CommandData, DataModel, QueryData
from intent_to_schedule.application.query import Summary
from intent_to_schedule.application.translate import (
    Step,
    StepRecord,
    StepTranslator,
    Utterance,
)


class QueryOutput(DataModel):
    """Structured output for a query step."""

    kind: Literal["query"]
    query: QueryData


class ApplyOutput(DataModel):
    """Structured output for an apply step."""

    kind: Literal["apply"]
    commands: tuple[CommandData, ...]


class SolveOutput(DataModel):
    """Structured output for a solve step."""

    kind: Literal["solve"]
    stability: bool


class MessageOutput(DataModel):
    """Structured output for a message step."""

    kind: Literal["message"]
    text: str


# TODO: check that OpenAI strict structured outputs accept the select and limit defaults in QueryData; if not, drop the defaults from the shared forms.
class StepOutput(DataModel):
    """Root of the structured output for a translation step."""

    result: QueryOutput | ApplyOutput | SolveOutput | MessageOutput


def convert_step_output(output: StepOutput) -> Step:
    """Convert structured output to a translation step."""
    raise NotImplementedError


_ELEMENT_PROMPT: str = (
    "Translate the latest user utterance into commands that add, replace, or remove Tasks. "
    "Use only existing person, task, and constraint IDs from the snapshot. "
    "fixed_tasks are existing events that cannot be moved, but constraints may reference their ids; to make one movable, replace it with a Task of the same id. "
    "Ask a clarifying question with ambiguous when the meaning is not determined, such as with same-named people; do not guess. "
    "Importance low, medium, or high means how much it matters to do the Task at all. "
    "Required means the Task must be scheduled, including expressions such as 絶対. "
    "Stability means how strongly to keep the Task at its previous time. "
)

_QUESTION_PROMPT: str = (
    "\nClarifying questions are read by the user: ask in the user's language and never mention internal terms "
    "(stability, hard, soft, strength, importance, required, drop, measure, evaluation, IDs). "
    "When you want to ask about one of them, phrase it like these examples:\n"
    "- stability false: 今の配置をいったん崩して、全体を組み直してもいいですか？\n"
    "- stability true: 今の配置はなるべく動かさずに調整しますか？\n"
    "- hard or soft: 絶対に守る条件ですか、それともできればの希望ですか？\n"
    "- strength: どのくらい強い希望ですか？\n"
    "- required: 必ず入れる必要がありますか？\n"
    "- importance or drop: 入りきらない場合は見送ってもいいですか？\n"
    "Do not ask about choices that lead to the same result."
)

_CONSTRAINT_PROMPT: str = (
    "Translate the latest user utterance into commands that add or remove constraints. "
    'Set stability false only when the user clearly asks to rebuild the whole schedule, such as "redo everything" or "start over"; true for partial changes or new additions. '
    "If it is unclear whether existing placements should be rebuilt, return ambiguous and ask whether it is fine to rearrange the whole schedule. "
    "Use only existing IDs from the snapshot. A measure extracts a schedule value and an evaluation scores it. "
    "fixed_tasks are existing events that cannot be moved, but constraints may reference their ids; to make one movable, replace it with a Task of the same id. "
    "Supported pairs: point with distance to an instant; interval with intrusion into a region; "
    "dependency with distance or shortfall of a duration, measuring the gap from the end of from_task to the start of to_task; "
    "aggregate with excess per day, using a count or total duration. "
    "To express wanting a task inside A, use the complement of A within the horizon as the intrusion region. "
    "Use hard only for must requirements. For soft preferences choose weak, normal, or strong strength. "
    "All times and durations must be on the slot grid. Ask a clarifying question instead of guessing. "
)


class OpenAIStepTranslator(StepTranslator):
    """StepTranslator backed by the OpenAI API."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        clock: Callable[[], datetime],
    ) -> None:
        self._client: openai.OpenAI = client
        self._model: str = model
        self._clock: Callable[[], datetime] = clock

    def translate(
        self,
        dialogue: Sequence[Utterance],
        summary: Summary,
        steps: Sequence[StepRecord],
    ) -> Step:
        """Choose the next step for the latest utterance."""
        raise NotImplementedError
