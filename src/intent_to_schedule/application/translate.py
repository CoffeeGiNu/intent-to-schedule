from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from intent_to_schedule.application.command import ExecuteResult, SchedulingCommand
from intent_to_schedule.application.query import (
    AnswerResult,
    SchedulingQuery,
    Summary,
)


class Speaker(Enum):
    """Who spoke an utterance."""

    USER = "user"
    ASSISTANT = "assistant"


@dataclass(frozen=True)
class Utterance:
    """One message in a conversation."""

    speaker: Speaker
    text: str


@dataclass(frozen=True)
class QueryStep:
    """Step that asks a query."""

    query: SchedulingQuery


@dataclass(frozen=True)
class ApplyStep:
    """Step that applies commands."""

    commands: tuple[SchedulingCommand, ...]


@dataclass(frozen=True)
class SolveStep:
    """Step that solves the problem and ends the turn."""

    stability: bool
    """Whether to keep Tasks near their previous start."""


@dataclass(frozen=True)
class MessageStep:
    """Step that replies to the user and ends the turn."""

    text: str


type TranslateResult = QueryStep | ApplyStep | SolveStep | MessageStep
"""Next step chosen for an utterance."""


@dataclass(frozen=True)
class QueryRecord:
    """Query step taken earlier in the turn and its result."""

    step: QueryStep
    result: AnswerResult


@dataclass(frozen=True)
class ApplyRecord:
    """Apply step taken earlier in the turn and its result."""

    step: ApplyStep
    result: ExecuteResult


type StepRecord = QueryRecord | ApplyRecord
"""Step taken earlier in the turn and its result."""


class CommandTranslator(Protocol):
    """Port that translates utterances into steps."""

    def translate(
        self,
        dialogue: Sequence[Utterance],
        summary: Summary,
        steps: Sequence[StepRecord],
    ) -> TranslateResult:
        """Choose the next step for the latest utterance."""
        ...
