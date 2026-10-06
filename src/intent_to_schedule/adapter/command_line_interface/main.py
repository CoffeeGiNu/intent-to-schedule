"""Argparse entry point for JSON scheduling commands."""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path
from typing import Annotated, NoReturn

import openai
from pydantic import Field, TypeAdapter, ValidationError

from intent_to_schedule.adapter.local.store import (
    LocalDialogueStore,
    LocalStateStore,
    load_state,
)
from intent_to_schedule.adapter.data_model import (
    APPLY_OPERATION_DESCRIPTION,
    CalendarInputData,
    CommandsData,
    QueryData,
    QUERY_OPERATION_DESCRIPTION,
    SCHEDULE_OPERATION_DESCRIPTION,
    ScheduleData,
    answer_result_record,
    answer_record,
    convert_calendar_input,
    convert_commands_input,
    convert_query,
    execute_result_record,
    rejected_record,
    solution_record,
)
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.adapter.openai.translate import OpenAIStepTranslator
from intent_to_schedule.application.command import (
    Executed,
    Rejected,
    SchedulingCommand,
)
from intent_to_schedule.application.converse import (
    Conversation,
    DialogueStore,
    Exhausted,
)
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.query import (
    AnswerResult,
    SchedulingQuery,
    Summary,
    summarize,
)
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.store import StateStore
from intent_to_schedule.application.solve import (
    FeasibleSolution,
    NoFeasibleSolution,
    OptimalSolution,
    Solution,
    SolutionNotFound,
)
from intent_to_schedule.application.translate import MessageStep
from intent_to_schedule.domain.consistency import (
    AlignedToSlots,
    AllOf,
    AvailabilityForEveryone,
    ConsistencyError,
    ReferencesExist,
    NonemptyTimeWindows,
    UniqueIds,
    Validator,
    Violations,
)
from intent_to_schedule.domain.problem import SchedulingProblem


class JsonParser(argparse.ArgumentParser):
    """Argument parser whose errors can be returned as JSON."""

    def error(self, message: str) -> NoReturn:
        """Raise a parse error for the JSON error handler."""
        raise ValueError(message)


COMMANDS: dict[str, tuple[str, str]] = {
    "init": (
        "Create state from a calendar JSON file",
        "Initializes the problem and clears the previous schedule and dialogue. "
        "Prints the summary; use show for the complete state.",
    ),
    "show": (
        "Print the complete state",
        "Includes the problem, previous schedule, and dialogue.",
    ),
    "schema": ("Print the JSON Schema of the apply or query input", ""),
    "apply": (
        "Apply a batch of commands",
        "Reads JSON from --file or standard input. "
        "See `schema apply` for input fields. "
        + APPLY_OPERATION_DESCRIPTION
        + " Exits 1 on rejection.",
    ),
    "query": (
        "Read a summary, records, or available start times",
        "Reads JSON from --file or standard input; see `schema query`. "
        + QUERY_OPERATION_DESCRIPTION
        + " Exits 1 on rejection.",
    ),
    "schedule": (
        "Schedule the problem and store the schedule",
        "Prints one JSON document. Exit codes: 0, 2, and 3 for optimal or feasible, no_feasible_solution, and solution_not_found respectively. "
        + SCHEDULE_OPERATION_DESCRIPTION
        + " `--no-stability` disables stability costs.",
    ),
    "chat": (
        "Run a demonstration conversation turn with OpenAI",
        "Uses query, apply, schedule, or message steps, with a limit of 12. "
        "query, apply, and schedule steps behave like those commands; accepted applies stay saved however the turn ends. "
        "Exit codes: 0, 2, and 3 for optimal or feasible, no_feasible_solution, and solution_not_found respectively. "
        "The assistant text is Scheduled. for optimal or feasible, No feasible solution. for no_feasible_solution, and Solution not found. for solution_not_found. "
        "Every outcome saves the dialogue. "
        "Reads OPENAI_API_KEY and OPENAI_BASE_URL from the environment.",
    ),
    "help": ("Print this message or the help of the given subcommand", ""),
}


def parser() -> JsonParser:
    """Create the command line argument parser."""
    root: JsonParser = JsonParser(
        prog="intent-to-schedule",
        usage="%(prog)s [OPTIONS] <COMMAND>",
        description="Query a stored scheduling problem, apply commands, and schedule it. Results are JSON; help is text.",
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
                    help="Read commands from this JSON file [default: standard input]",
                )
            case "schema":
                command.add_argument(
                    "schema_command",
                    nargs="?",
                    choices=("apply", "query"),
                    default="apply",
                    metavar="<COMMAND>",
                    help="Print the input schema for this command [default: apply]",
                )
            case "query":
                command.add_argument(
                    "--file",
                    type=Path,
                    metavar="<FILE>",
                    help="Read a query from this JSON file [default: standard input]",
                )
            case "schedule":
                command.add_argument(
                    "--no-stability",
                    action="store_true",
                    help="Disable stability costs and schedule from scratch",
                )
            case "chat":
                command.add_argument(
                    "text", metavar="<TEXT>", help="Scheduling request or question"
                )
                command.add_argument(
                    "--model",
                    required=True,
                    metavar="<MODEL>",
                    help="Model name accepted by the configured OpenAI service",
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
    emit(rejected_record(Rejected(violations)))
    return 1


def scheduling(validator: Validator, state_store: StateStore) -> Scheduling:
    """Wire the scheduling use case."""
    policy: ObjectivePolicy = DEFAULT_POLICY
    return Scheduling(MathOptSchedulingSolver(), validator, state_store, policy)


def init(
    calendar_path: Path,
    validator: Validator,
    state_store: StateStore,
    dialogue_store: DialogueStore,
) -> int:
    """Create and validate a problem from calendar input."""
    form: CalendarInputData = CalendarInputData.model_validate_json(
        calendar_path.read_text(encoding="utf-8")
    )
    problem: SchedulingProblem = convert_calendar_input(form)
    violations: Violations = validator.validate(problem)
    if not violations.is_empty:
        return reject(violations)
    state_store.save_problem(problem)
    state_store.save_previous(None)
    dialogue_store.save(())
    emit(answer_record(summarize(problem, None)))
    return 0


def apply(input_path: Path | None, service: Scheduling) -> int:
    """Apply a JSON command batch to the current problem."""
    source: str = (
        input_path.read_text(encoding="utf-8")
        if input_path is not None
        else sys.stdin.read()
    )
    commands: tuple[SchedulingCommand, ...] = convert_commands_input(
        CommandsData.model_validate_json(source)
    )
    result: Executed | Rejected = service.execute(commands)
    emit(execute_result_record(commands, result))
    return 1 if isinstance(result, Rejected) else 0


def query(input_path: Path | None, service: Scheduling) -> int:
    """Answer a JSON query about the current problem and previous schedule."""
    source: str = (
        input_path.read_text(encoding="utf-8")
        if input_path is not None
        else sys.stdin.read()
    )
    adapter: TypeAdapter[QueryData] = TypeAdapter(
        Annotated[QueryData, Field(discriminator="kind")]
    )
    request: SchedulingQuery = convert_query(adapter.validate_json(source))
    result: AnswerResult = service.answer(request)
    emit(answer_result_record(result))
    return 1 if isinstance(result, Rejected) else 0


def schedule(service: Scheduling, request: ScheduleData) -> int:
    """Schedule the current problem and persist a found schedule."""
    result: Solution = service.schedule(request.stability)
    emit(solution_record(result))
    if isinstance(result, NoFeasibleSolution):
        return 2
    if isinstance(result, SolutionNotFound):
        return 3
    return 0


def chat(
    text: str,
    model: str,
    now: datetime | None,
    service: Scheduling,
    dialogue_store: DialogueStore,
) -> int:
    """Handle an utterance with the OpenAI API."""
    summary: Summary = service.summarize()
    current: datetime = (
        now if now is not None else summary.grid.horizon.start
    )
    conversation: Conversation = Conversation(
        OpenAIStepTranslator(openai.OpenAI(), model, lambda: current),
        service,
        dialogue_store,
    )
    outcome: MessageStep | Solution | Exhausted = conversation.respond(text)
    output: dict[str, object]
    status: int
    message: str
    match outcome:
        case MessageStep(text=message):
            output = {"message": message}
            status = 0
        case Exhausted():
            output = {"exhausted": True}
            status = 1
        case NoFeasibleSolution():
            output = solution_record(outcome)
            status = 2
        case SolutionNotFound():
            output = solution_record(outcome)
            status = 3
        case OptimalSolution() | FeasibleSolution():
            output = solution_record(outcome)
            status = 0
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
            NonemptyTimeWindows(),
        )
        state_store: LocalStateStore = LocalStateStore(path)
        dialogue_store: LocalDialogueStore = LocalDialogueStore(path)
        service: Scheduling = scheduling(validator, state_store)
        match args.command:
            case "init":
                return init(args.calendar, validator, state_store, dialogue_store)
            case "apply":
                return apply(args.file, service)
            case "query":
                return query(args.file, service)
            case "schedule":
                return schedule(
                    service,
                    ScheduleData(stability=False) if args.no_stability else ScheduleData(),
                )
            case "chat":
                return chat(
                    args.text,
                    args.model,
                    args.now,
                    service,
                    dialogue_store,
                )
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
