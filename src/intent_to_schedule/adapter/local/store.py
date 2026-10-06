from collections.abc import Sequence
from pathlib import Path

from intent_to_schedule.adapter.local.state import (
    State,
    UtteranceState,
    to_dialogue,
    to_problem,
    to_problem_state,
    to_schedule,
    to_schedule_state,
)
from intent_to_schedule.application.converse import DialogueStore
from intent_to_schedule.application.store import StateStore
from intent_to_schedule.application.translate import Speaker, Utterance
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule


def load_state(path: Path) -> State:
    """Load a local state file."""
    return State.model_validate_json(path.read_text(encoding="utf-8"))


def save_state(path: Path, state: State) -> None:
    """Write a local state file and create its parent directories."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(state.model_dump_json(indent=2) + "\n", encoding="utf-8")


class LocalStateStore(StateStore):
    """Keep a scheduling problem and previous schedule in one local file."""

    def __init__(self, path: Path) -> None:
        self._path: Path = path

    def load_problem(self) -> SchedulingProblem:
        """Load the stored scheduling problem."""
        return to_problem(load_state(self._path).problem)

    def save_problem(self, problem: SchedulingProblem) -> None:
        """Replace the stored problem and create an initial state when needed."""
        state: State = (
            load_state(self._path)
            if self._path.exists()
            else State(
                problem=to_problem_state(problem),
                previous=None,
                dialogue=(),
            )
        )
        updated: State = state.model_copy(update={"problem": to_problem_state(problem)})
        save_state(self._path, updated)

    def load_previous(self) -> Schedule | None:
        """Load the stored previous schedule."""
        return to_schedule(load_state(self._path).previous)

    def save_previous(self, previous: Schedule | None) -> None:
        """Replace the stored previous schedule."""
        state: State = load_state(self._path)
        updated: State = state.model_copy(
            update={"previous": to_schedule_state(previous)}
        )
        save_state(self._path, updated)


class LocalDialogueStore(DialogueStore):
    """Keep conversation history in a local state file."""

    def __init__(self, path: Path) -> None:
        self._path: Path = path

    def load(self) -> tuple[Utterance, ...]:
        """Load the stored conversation history."""
        return to_dialogue(load_state(self._path).dialogue)

    def save(self, dialogue: Sequence[Utterance]) -> None:
        """Replace the stored conversation history."""
        state: State = load_state(self._path)
        utterances: tuple[UtteranceState, ...] = tuple(
            UtteranceState(
                speaker="user" if item.speaker is Speaker.USER else "assistant",
                text=item.text,
            )
            for item in dialogue
        )
        updated: State = state.model_copy(update={"dialogue": utterances})
        save_state(self._path, updated)
