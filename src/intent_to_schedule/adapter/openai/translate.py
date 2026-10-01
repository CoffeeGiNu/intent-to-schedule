import json
from collections.abc import Sequence
from dataclasses import asdict
from datetime import datetime, timedelta
from enum import Enum
from typing import Callable, Literal

import openai

from intent_to_schedule.adapter.data_model import (
    ConstraintCommandData,
    DataModel,
    ElementCommandData,
    convert_command,
)
from intent_to_schedule.application.command import (
    ConstraintCommand,
    ElementCommand,
    ExecuteResult,
    Rejected,
    SchedulingCommand,
    execute_commands,
)
from intent_to_schedule.application.translate import (
    Ambiguous,
    CommandTranslator,
    Translated,
    TranslateResult,
    Utterance,
)
from intent_to_schedule.domain.consistency import (
    ConsistencyError,
    Validator,
    Violations,
)
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule


class AmbiguousOutput(DataModel):
    """JSON form of Ambiguous."""

    kind: Literal["ambiguous"]
    question: str


class ElementCommandsOutput(DataModel):
    """Element commands translated from an utterance."""

    kind: Literal["translated"]
    commands: tuple[ElementCommandData, ...]


class ConstraintCommandsOutput(DataModel):
    """Constraint commands translated from an utterance."""

    kind: Literal["translated"]
    commands: tuple[ConstraintCommandData, ...]


class ElementTranslationOutput(DataModel):
    """Root of the structured output for the element step: element commands or a clarifying question."""

    result: ElementCommandsOutput | AmbiguousOutput


class ConstraintTranslationOutput(DataModel):
    """Root of the structured output for the constraint step: constraint commands or a clarifying question."""

    result: ConstraintCommandsOutput | AmbiguousOutput


def convert_element_output(
    output: ElementTranslationOutput,
) -> tuple[ElementCommand, ...] | Ambiguous:
    """Convert element step output to element commands, generating IDs for added Tasks."""
    question: str
    commands: tuple[ElementCommandData, ...]
    match output.result:
        case AmbiguousOutput(question=question):
            return Ambiguous(question)
        case ElementCommandsOutput(commands=commands):
            return tuple(convert_command(command) for command in commands)


def convert_constraint_output(
    output: ConstraintTranslationOutput,
) -> tuple[ConstraintCommand, ...] | Ambiguous:
    """Convert constraint step output to constraint commands, generating IDs for added constraints."""
    question: str
    commands: tuple[ConstraintCommandData, ...]
    match output.result:
        case AmbiguousOutput(question=question):
            return Ambiguous(question)
        case ConstraintCommandsOutput(commands=commands):
            return tuple(convert_command(command) for command in commands)


_ELEMENT_PROMPT: str = (
    "Translate the latest user utterance into commands that add, replace, or remove Tasks. "
    "Use only existing person, task, and constraint IDs from the snapshot. "
    "Ask a clarifying question with ambiguous when the meaning is not determined, such as with same-named people; do not guess. "
    "Importance low, medium, or high means how much it matters to do the Task at all. "
    "Required means the Task must be scheduled, including expressions such as 絶対. "
    "Stability means how strongly to keep the Task at its previous time. "
)

_CONSTRAINT_PROMPT: str = (
    "Translate the latest user utterance into commands that add or remove constraints. "
    "Use only existing IDs from the snapshot. A measure extracts a schedule value and an evaluation scores it. "
    "Supported pairs: point with distance to an instant; interval with intrusion into a region; "
    "dependency with distance or shortfall of a duration, measuring the gap from the end of from_task to the start of to_task; "
    "aggregate with excess per day, using a count or total duration. "
    "To express wanting a task inside A, use the complement of A within the horizon as the intrusion region. "
    "Use hard only for must requirements. For soft preferences choose weak, normal, or strong strength. "
    "All times and durations must be on the slot grid. Ask a clarifying question instead of guessing. "
)


def _json_default(value: object) -> object:
    """Serialize snapshot values for the model."""
    match value:
        case datetime():
            return value.isoformat()
        case timedelta():
            return value.total_seconds()
        case Enum():
            return value.value
        case frozenset():
            return [asdict(item) for item in value]
        case _:
            raise TypeError(f"Cannot serialize {type(value).__name__}")


def _snapshot(
    problem: SchedulingProblem, previous: Schedule | None, now: datetime
) -> str:
    """Describe the current scheduling state."""
    snapshot: dict[str, object] = {
        "current_time": now.isoformat(),
        "problem": asdict(problem),
        "previous_schedule": asdict(previous) if previous is not None else None,
    }
    return json.dumps(snapshot, default=_json_default, ensure_ascii=False)


class OpenAICommandTranslator(CommandTranslator):
    """CommandTranslator backed by the OpenAI API."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        validator: Validator,
        clock: Callable[[], datetime],
    ) -> None:
        self._client: openai.OpenAI = client
        self._model: str = model
        self._validator: Validator = validator
        self._clock: Callable[[], datetime] = clock

    def translate(
        self,
        dialogue: Sequence[Utterance],
        problem: SchedulingProblem,
        previous: Schedule | None,
    ) -> TranslateResult:
        """Translate the latest utterance into element commands, then into constraint commands."""
        elements: tuple[tuple[ElementCommand, ...], SchedulingProblem] | Ambiguous = (
            self._translate_step(
                dialogue,
                problem,
                previous,
                _ELEMENT_PROMPT,
                ElementTranslationOutput,
                convert_element_output,
            )
        )
        if isinstance(elements, Ambiguous):
            return elements
        element_commands: tuple[ElementCommand, ...]
        updated_problem: SchedulingProblem
        element_commands, updated_problem = elements
        constraints: (
            tuple[tuple[ConstraintCommand, ...], SchedulingProblem] | Ambiguous
        ) = self._translate_step(
            dialogue,
            updated_problem,
            previous,
            _CONSTRAINT_PROMPT,
            ConstraintTranslationOutput,
            convert_constraint_output,
        )
        if isinstance(constraints, Ambiguous):
            return constraints
        constraint_commands: tuple[ConstraintCommand, ...] = constraints[0]
        return Translated((*element_commands, *constraint_commands))

    def _translate_step[Output: DataModel, Command: SchedulingCommand](
        self,
        dialogue: Sequence[Utterance],
        problem: SchedulingProblem,
        previous: Schedule | None,
        prompt: str,
        output_type: type[Output],
        convert: Callable[[Output], tuple[Command, ...] | Ambiguous],
    ) -> tuple[tuple[Command, ...], SchedulingProblem] | Ambiguous:
        """Request and validate one translation step."""
        messages: list[dict[str, str]] = [
            {
                "role": "system",
                "content": prompt
                + "\nSnapshot: "
                + _snapshot(problem, previous, self._clock()),
            },
            *(
                {"role": utterance.speaker.value, "content": utterance.text}
                for utterance in dialogue
            ),
        ]
        attempt: int
        error: ConsistencyError
        for attempt in range(3):
            response: openai.types.responses.ParsedResponse[Output] = (
                self._client.responses.parse(
                    model=self._model, input=messages, text_format=output_type
                )
            )
            output: Output | None = response.output_parsed
            if output is None:
                raise ValueError("Model returned no parsed translation")
            converted: tuple[Command, ...] | Ambiguous = convert(output)
            if isinstance(converted, Ambiguous):
                return converted
            try:
                updated_problem: SchedulingProblem = self._apply_commands(
                    converted, problem
                )
            except ConsistencyError as error:
                if attempt == 2:
                    raise
                messages.append(
                    {"role": "assistant", "content": output.model_dump_json()}
                )
                messages.append(
                    {
                        "role": "user",
                        "content": "Correct these violations: "
                        + "; ".join(item.message for item in error.violations.items),
                    }
                )
                continue
            return converted, updated_problem
        raise AssertionError("Translation retry loop ended unexpectedly")

    def _apply_commands(
        self, commands: Sequence[SchedulingCommand], problem: SchedulingProblem
    ) -> SchedulingProblem:
        """Apply commands and validate the resulting problem."""
        result: ExecuteResult = execute_commands(problem, commands)
        if isinstance(result, Rejected):
            raise ConsistencyError(result.violations)
        violations: Violations = self._validator.validate(result.problem)
        if not violations.is_empty:
            raise ConsistencyError(violations)
        return result.problem
