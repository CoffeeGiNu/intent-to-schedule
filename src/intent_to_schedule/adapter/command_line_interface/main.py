"""Argparse entry point for JSON scheduling commands."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

import openai
from pydantic import ValidationError

from intent_to_schedule.adapter.command_line_interface.state import (
    CalendarInput,
    FixedTaskState,
    ProblemState,
    State,
    UtteranceState,
    load_state,
    save_state,
    to_dialogue,
    to_problem,
    to_problem_state,
    to_schedule,
    to_schedule_state,
)
from intent_to_schedule.adapter.data_model import CommandsData, convert_commands_input
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.adapter.openai.translate import OpenAICommandTranslator
from intent_to_schedule.application.command import (
    AddConstraint,
    AddTask,
    Executed,
    Rejected,
    RemoveConstraint,
    RemoveTask,
    ReplaceTask,
    SchedulingCommand,
)
from intent_to_schedule.application.converse import Conversation, Response
from intent_to_schedule.application.policy import DEFAULT_POLICY
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import Infeasible, Solved
from intent_to_schedule.application.translate import Ambiguous
from intent_to_schedule.domain.consistency import (
    AlignedToSlots,
    AllOf,
    AvailabilityForEveryone,
    ConsistencyError,
    ReferencesExist,
    SupportedCombinations,
    UniqueIds,
    Validator,
    Violations,
)
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule
from intent_to_schedule.domain.task import Task, TaskId


class JsonParser(argparse.ArgumentParser):
    """Argument parser whose errors can be returned as JSON."""

    def error(self, message: str) -> None:
        """Raise a parse error for the JSON error handler."""
        raise ValueError(message)


COMMANDS: dict[str, tuple[str, str]] = {
    "init": (
        "Create the state from a calendar JSON",
        "The calendar JSON holds horizon, slot, people, availabilities, and fixed_tasks.",
    ),
    "show": (
        "Print the state",
        "The state holds the problem with IDs, the previous schedule, and the dialogue.",
    ),
    "schema": ("Print the JSON Schema of the apply input", ""),
    "apply": (
        "Apply a batch of commands",
        "Reads commands JSON (see `schema`) from --file or stdin and prints the IDs it created. "
        "Add Tasks first, then reference their IDs in constraints. Exits 1 if rejected; the state is left unchanged.",
    ),
    "solve": ("Solve the problem and store the schedule", "Exits 2 if infeasible."),
    "chat": (
        "Translate an utterance with the OpenAI API, then apply and solve",
        "Reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.",
    ),
    "help": ("Print this message or the help of the given subcommand", ""),
}


def parser() -> JsonParser:
    """Create the command line argument parser."""
    root: JsonParser = JsonParser(
        prog="intent-to-schedule",
        usage="%(prog)s [OPTIONS] <COMMAND>",
        description="Apply scheduling commands to a stored problem and solve it. Every command prints one JSON document.",
        add_help=False,
    )
    root._optionals.title = "Options"
    root.add_argument(
        "--state",
        type=Path,
        default=Path(".state/state.json"),
        metavar="<PATH>",
        help="State file [default: .state/state.json]",
    )
    root.add_argument("-h", "--help", action="help", help="Print help")
    subcommands: argparse._SubParsersAction[argparse.ArgumentParser] = (
        root.add_subparsers(
            dest="command",
            required=True,
            title="Commands",
            metavar="<COMMAND>",
            prog="intent-to-schedule",
        )
    )
    command: argparse.ArgumentParser
    for name, (summary, detail) in COMMANDS.items():
        command = subcommands.add_parser(
            name,
            help=summary,
            description=f"{summary}. {detail}".strip(),
            add_help=False,
        )
        command._positionals.title = "Arguments"
        command._optionals.title = "Options"
        match name:
            case "init":
                command.add_argument(
                    "--calendar",
                    type=Path,
                    required=True,
                    metavar="<FILE>",
                    help="Calendar JSON file",
                )
            case "apply":
                command.add_argument(
                    "--file",
                    type=Path,
                    metavar="<FILE>",
                    help="Commands JSON file [default: stdin]",
                )
            case "solve":
                command.add_argument(
                    "--no-stability",
                    action="store_true",
                    help="Ignore the previous schedule and solve from scratch",
                )
            case "chat":
                command.add_argument("text", metavar="<TEXT>", help="Utterance")
                command.add_argument(
                    "--model",
                    required=True,
                    metavar="<MODEL>",
                    help="Model name, e.g. openai/gpt-5-mini",
                )
                command.add_argument(
                    "--now",
                    type=datetime.fromisoformat,
                    metavar="<ISO_DATETIME>",
                    help="Current time for words like tomorrow [default: start of the planning horizon]",
                )
            case "help":
                command.add_argument(
                    "topic",
                    nargs="?",
                    choices=[*COMMANDS],
                    metavar="<COMMAND>",
                    help="Subcommand",
                )
        if name not in ("help", "schema"):
            command.add_argument(
                "--state",
                type=Path,
                default=argparse.SUPPRESS,
                metavar="<PATH>",
                help="State file",
            )
        command.add_argument("-h", "--help", action="help", help="Print help")
    return root


def emit(value: object) -> None:
    """Print one JSON document."""
    print(json.dumps(value, ensure_ascii=False))


def reject(violations: Violations) -> int:
    """Print a rejected result."""
    emit({"rejected": [item.message for item in violations.items]})
    return 1


def scheduling(validator: Validator) -> Scheduling:
    """Wire the scheduling use case."""
    return Scheduling(MathOptSchedulingSolver(DEFAULT_POLICY), validator)


def init(path: Path, calendar_path: Path, service: Scheduling) -> int:
    """Create and validate a problem from calendar input."""
    calendar: CalendarInput = CalendarInput.model_validate_json(
        calendar_path.read_text(encoding="utf-8")
    )
    form: ProblemState = ProblemState(
        calendar=calendar,
        people=calendar.people,
        tasks=(),
        fixed_tasks=tuple(
            FixedTaskState(id=TaskId.generate(), **item.model_dump())
            for item in calendar.fixed_tasks
        ),
        constraints=(),
    )
    problem: SchedulingProblem = to_problem(form)
    result: Executed | Rejected = service.execute(problem, ())
    match result:
        case Rejected(violations=violations):
            return reject(violations)
        case Executed(problem=updated):
            state: State = State(
                problem=to_problem_state(updated), previous=None, dialogue=()
            )
            save_state(path, state)
            emit(state.model_dump(mode="json"))
            return 0


def command_record(command: SchedulingCommand) -> dict[str, str]:
    """Describe the ID created or touched by an applied command."""
    match command:
        case AddTask(task=task):
            return {"kind": "add_task", "task_id": task.id.value, "name": task.name}
        case ReplaceTask(task=task):
            return {"kind": "replace_task", "task_id": task.id.value}
        case RemoveTask(task_id=task_id):
            return {"kind": "remove_task", "task_id": task_id.value}
        case AddConstraint(constraint=constraint):
            return {"kind": "add_constraint", "constraint_id": constraint.id.value}
        case RemoveConstraint(constraint_id=constraint_id):
            return {"kind": "remove_constraint", "constraint_id": constraint_id.value}


def apply(path: Path, input_path: Path | None, service: Scheduling) -> int:
    """Apply a JSON command batch to the current problem."""
    state: State = load_state(path)
    source: str = (
        input_path.read_text(encoding="utf-8")
        if input_path is not None
        else sys.stdin.read()
    )
    commands: tuple[SchedulingCommand, ...] = convert_commands_input(
        CommandsData.model_validate_json(source)
    )
    result: Executed | Rejected = service.execute(to_problem(state.problem), commands)
    match result:
        case Rejected(violations=violations):
            return reject(violations)
        case Executed(problem=updated):
            save_state(
                path, state.model_copy(update={"problem": to_problem_state(updated)})
            )
            emit({"executed": [command_record(command) for command in commands]})
            return 0


def schedule_output(
    problem: SchedulingProblem, schedule: Schedule
) -> dict[str, object]:
    """Describe a solved schedule with task names and end times."""
    tasks: dict[TaskId, Task] = {task.id: task for task in problem.tasks}
    scheduled: list[dict[str, str]] = [
        {
            "task_id": item.task_id.value,
            "name": tasks[item.task_id].name,
            "start": item.start.isoformat(),
            "end": (item.start + tasks[item.task_id].duration).isoformat(),
        }
        for item in schedule.scheduled
    ]
    dropped: list[dict[str, str]] = [
        {"task_id": identifier.value, "name": tasks[identifier].name}
        for identifier in sorted(schedule.dropped_task_ids, key=lambda item: item.value)
    ]
    return {"scheduled": scheduled, "dropped": dropped}


def solve(path: Path, service: Scheduling, stability: bool) -> int:
    """Solve the current problem and persist the result."""
    state: State = load_state(path)
    problem: SchedulingProblem = to_problem(state.problem)
    previous: Schedule | None = to_schedule(state.previous) if stability else None
    result: Solved | Infeasible = service.solve(problem, previous)
    match result:
        case Infeasible():
            emit({"infeasible": True})
            return 2
        case Solved(schedule=schedule):
            save_state(
                path, state.model_copy(update={"previous": to_schedule_state(schedule)})
            )
            emit(schedule_output(problem, schedule))
            return 0


def chat(
    path: Path,
    text: str,
    model: str,
    now: datetime | None,
    service: Scheduling,
    validator: Validator,
) -> int:
    """Translate an utterance, then apply and solve it."""
    state: State = load_state(path)
    dialogue: tuple[UtteranceState, ...] = (
        *state.dialogue,
        UtteranceState(speaker="user", text=text),
    )
    problem: SchedulingProblem = to_problem(state.problem)
    current: datetime = now if now is not None else problem.calendar.grid.horizon.start
    conversation: Conversation = Conversation(
        OpenAICommandTranslator(openai.OpenAI(), model, validator, lambda: current),
        service,
    )
    response: Response = conversation.respond(
        to_dialogue(dialogue), problem, to_schedule(state.previous)
    )
    match response.outcome:
        case Ambiguous(question=question):
            updated: State = state.model_copy(
                update={
                    "dialogue": (
                        *dialogue,
                        UtteranceState(speaker="assistant", text=question),
                    )
                }
            )
            save_state(path, updated)
            emit({"question": question})
            return 0
        case Infeasible():
            updated = state.model_copy(
                update={
                    "problem": to_problem_state(response.problem),
                    "dialogue": (
                        *dialogue,
                        UtteranceState(speaker="assistant", text="Infeasible."),
                    ),
                }
            )
            save_state(path, updated)
            emit({"infeasible": True})
            return 2
        case Solved(schedule=schedule):
            updated = state.model_copy(
                update={
                    "problem": to_problem_state(response.problem),
                    "previous": to_schedule_state(schedule),
                    "dialogue": (
                        *dialogue,
                        UtteranceState(speaker="assistant", text="Scheduled."),
                    ),
                }
            )
            save_state(path, updated)
            emit(schedule_output(response.problem, schedule))
            return 0


def main(argv: list[str] | None = None) -> int:
    """Run one command and return its exit status."""
    try:
        root: JsonParser = parser()
        args: argparse.Namespace = root.parse_args(argv)
        if args.command == "help":
            root.parse_args([args.topic, "--help"] if args.topic else ["--help"])
        if args.command == "schema":
            emit(CommandsData.model_json_schema())
            return 0
        path: Path = args.state
        if args.command == "show":
            emit(load_state(path).model_dump(mode="json"))
            return 0
        validator: AllOf = AllOf(
            UniqueIds(),
            ReferencesExist(),
            AvailabilityForEveryone(),
            AlignedToSlots(),
            SupportedCombinations(),
        )
        service: Scheduling = scheduling(validator)
        match args.command:
            case "init":
                return init(path, args.calendar, service)
            case "apply":
                return apply(path, args.file, service)
            case "solve":
                return solve(path, service, not args.no_stability)
            case "chat":
                return chat(path, args.text, args.model, args.now, service, validator)
    except ConsistencyError as error:
        return reject(error.violations)
    except (
        OSError,
        ValueError,
        ValidationError,
        RuntimeError,
        openai.APIError,
    ) as error:
        emit({"error": str(error)})
        return 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
