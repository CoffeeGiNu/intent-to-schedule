from collections.abc import Sequence
from dataclasses import dataclass
from enum import Enum
from typing import Protocol

from intent_to_schedule.application.command import SchedulingCommand
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule


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
class Translated:
    """Commands translated from an utterance."""

    commands: tuple[SchedulingCommand, ...]
    stability: bool


@dataclass(frozen=True)
class Ambiguous:
    """Clarifying question for an utterance whose meaning is not determined."""

    question: str


type TranslateResult = Translated | Ambiguous
"""Result of translating an utterance."""


class CommandTranslator(Protocol):
    """Port that translates utterances into commands."""

    def translate(
        self,
        dialogue: Sequence[Utterance],
        problem: SchedulingProblem,
        previous: Schedule | None,
    ) -> TranslateResult:
        """Translate the latest utterance in a conversation."""
        ...
