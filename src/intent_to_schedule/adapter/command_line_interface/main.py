"""Argparse entry point for JSON scheduling commands."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn

import openai
from pydantic import Field, TypeAdapter, ValidationError

from intent_to_schedule.adapter.command_line_interface.state import (
    CalendarInput,
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
from intent_to_schedule.adapter.data_model import (
    CommandsData,
    FixedTaskData,
    QueryData,
    answer_record,
    command_record,
    convert_commands_input,
    convert_query,
)
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.adapter.openai.translate import OpenAIStepTranslator
from intent_to_schedule.application.command import (
    Executed,
    Rejected,
    SchedulingCommand,
)
from intent_to_schedule.application.converse import Conversation, Exhausted, Response
from intent_to_schedule.application.policy import DEFAULT_POLICY
from intent_to_schedule.application.query import AnswerResult, SchedulingQuery
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import Infeasible, Solved
from intent_to_schedule.application.translate import MessageStep
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

    def error(self, message: str) -> NoReturn:
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
    "schema": ("Print the JSON Schema of the apply or query input", ""),
    "apply": (
        "Apply a batch of commands",
        "Reads commands JSON (see `schema`) from --file or stdin and prints the IDs it created. "
        "Add Tasks first, then reference their IDs in constraints. Exits 1 if rejected; the state is left unchanged.",
    ),
    "query": (
        "Query the current problem and previous schedule",
        "Reads query JSON (see `schema query`) from --file or stdin and prints the answer. "
        "Exits 1 if rejected.",
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
    subcommands: argparse._SubParsersAction[JsonParser] = root.add_subparsers(
        dest="command",
        required=True,
        title="Commands",
        metavar="<COMMAND>",
        prog="intent-to-schedule",
    )
    command: argparse.ArgumentParser
    name: str
    summary: str
    detail: str
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
            case "schema":
                command.add_argument(
                    "schema_command",
                    nargs="?",
                    choices=("apply", "query"),
                    default="apply",
                    metavar="<COMMAND>",
                    help="Input command [default: apply]",
                )
            case "query":
                command.add_argument(
                    "--file",
                    type=Path,
                    metavar="<FILE>",
                    help="Query JSON file [default: stdin]",
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
            FixedTaskData(id=TaskId.generate(), **item.model_dump())
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
    problem: SchedulingProblem = to_problem(state.problem)
    result: Executed | Rejected = service.execute(problem, commands)
    match result:
        case Rejected(violations=violations):
            return reject(violations)
        case Executed(problem=updated):
            save_state(
                path, state.model_copy(update={"problem": to_problem_state(updated)})
            )
            emit(
                {
                    "executed": [
                        command_record(command, problem.calendar.grid)
                        for command in commands
                    ]
                }
            )
            return 0


def query(path: Path, input_path: Path | None) -> int:
    """Answer a JSON query about the current problem and previous schedule."""
    state: State = load_state(path)
    source: str = (
        input_path.read_text(encoding="utf-8")
        if input_path is not None
        else sys.stdin.read()
    )
    adapter: TypeAdapter[QueryData] = TypeAdapter(
        Annotated[QueryData, Field(discriminator="kind")]
    )
    request: SchedulingQuery = convert_query(adapter.validate_json(source))
    problem: SchedulingProblem = to_problem(state.problem)
    previous: Schedule | None = to_schedule(state.previous)
    result: AnswerResult = request.answer(problem, previous)
    if isinstance(result, Rejected):
        return reject(result.violations)
    emit(answer_record(result.answer))
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
) -> int:
    """Handle an utterance with the OpenAI API."""
    state: State = load_state(path)
    dialogue: tuple[UtteranceState, ...] = (
        *state.dialogue,
        UtteranceState(speaker="user", text=text),
    )
    problem: SchedulingProblem = to_problem(state.problem)
    current: datetime = now if now is not None else problem.calendar.grid.horizon.start
    conversation: Conversation = Conversation(
        OpenAIStepTranslator(openai.OpenAI(), model, lambda: current), service
    )
    response: Response = conversation.respond(
        to_dialogue(dialogue), problem, to_schedule(state.previous)
    )
    updates: dict[str, object] = {}
    assistant_text: str
    output: dict[str, object]
    status: int
    message: str
    schedule: Schedule
    match response.outcome:
        case MessageStep(text=message):
            assistant_text = message
            output = {"message": message}
            status = 0
        case Exhausted():
            assistant_text = "Step limit reached."
            output = {"exhausted": True}
            status = 1
        case Infeasible():
            updates["problem"] = to_problem_state(response.problem)
            assistant_text = "Infeasible."
            output = {"infeasible": True}
            status = 2
        case Solved(schedule=schedule):
            updates["problem"] = to_problem_state(response.problem)
            updates["previous"] = to_schedule_state(schedule)
            assistant_text = "Scheduled."
            output = schedule_output(response.problem, schedule)
            status = 0
    updates["dialogue"] = (
        *dialogue,
        UtteranceState(speaker="assistant", text=assistant_text),
    )
    save_state(path, state.model_copy(update=updates))
    emit(output)
    return status


def main(argv: list[str] | None = None) -> int:
    """Run one command and return its exit status."""
    try:
        root: JsonParser = parser()
        args: argparse.Namespace = root.parse_args(argv)
        if args.command == "help":
            root.parse_args([args.topic, "--help"] if args.topic else ["--help"])
        if args.command == "schema":
            emit(
                TypeAdapter(
                    Annotated[QueryData, Field(discriminator="kind")]
                ).json_schema()
                if args.schema_command == "query"
                else CommandsData.model_json_schema()
            )
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
            case "query":
                return query(path, args.file)
            case "solve":
                return solve(path, service, not args.no_stability)
            case "chat":
                return chat(path, args.text, args.model, args.now, service)
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
