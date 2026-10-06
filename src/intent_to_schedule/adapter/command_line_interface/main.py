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
    ScheduleState,
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
    conflicts_record,
    convert_commands_input,
    convert_query,
    schedule_summary_record,
)
from intent_to_schedule.adapter.mathopt.solve import MathOptSchedulingSolver
from intent_to_schedule.adapter.openai.translate import OpenAIStepTranslator
from intent_to_schedule.application.command import (
    Executed,
    Rejected,
    SchedulingCommand,
)
from intent_to_schedule.application.converse import Conversation, Exhausted, Response
from intent_to_schedule.application.policy import DEFAULT_POLICY, ObjectivePolicy
from intent_to_schedule.application.query import AnswerResult, SchedulingQuery, summarize
from intent_to_schedule.application.schedule import Scheduling
from intent_to_schedule.application.solve import (
    Conflicts,
    ConflictsNotFound,
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
from intent_to_schedule.domain.schedule import Schedule
from intent_to_schedule.domain.task import TaskId


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
        "Constraints use time_window, time_bound, task_gap, or daily_limit conditions; see `schema apply` for fields. "
        "Prints created identifiers and any time window rounding note. "
        "A time_window command whose constraint the batch keeps also gets warnings naming movable tasks "
        "that no available start keeps within (or out of) its rounded windows after the whole batch; "
        "the batch is still saved. "
        "Register absences missing from the calendar as fixed tasks (add_task with start), not as avoid constraints. "
        "Exits 1 on rejection without saving changes.",
    ),
    "query": (
        "Read a summary, records, or available start times",
        "Reads JSON from --file or standard input; see `schema query`. "
        "evaluation measures current constraints against the last saved solution, with violation amounts, units, soft costs (hard items and their breakdown parts have no cost), and task or date breakdowns. "
        "Filters are violated_only (default false), constraint_ids, and task_ids, combined with and. "
        "Hard violations come first, then highest soft costs. Without a saved solution, has_previous is false and items are empty. "
        "objective_policy returns the current drop_costs, weights, per_count, stability_drop_cost_ratio, hard_violation_weight, and required_drop_cost. "
        "agenda returns one person's working intervals, items, and free intervals for each date in date_range, \"horizon\" by default or start and end dates, including dates without working time. "
        "Items are the person's fixed tasks and the tasks placed for the person in the last saved solution, with participants recorded when scheduled; they may predate current changes. "
        "Listing limit defaults to 20, with a maximum of 100; total counts matches before limiting and truncated indicates omitted items. Exits 1 on rejection.",
    ),
    "schedule": (
        "Schedule the problem and store the schedule",
        "Prints one JSON document. status is optimal, feasible, no_feasible_solution, or solution_not_found. "
        "Exit codes: 0, 2, and 3 for optimal or feasible, no_feasible_solution, and solution_not_found respectively. "
        "Optimal and feasible results return {status, summary, items}; both replace the previous schedule. "
        "A no_feasible_solution result returns {status, conflicts}; a solution_not_found result returns {status, reason}. "
        "A reason is time_limit when a time limit stops scheduling, otherwise the MathOpt termination reason in lowercase, such as numerical_error. "
        "Those two results leave the previous schedule unchanged. "
        "summary contains total_cost, costs (dropped_tasks, soft_constraints, stability), "
        "and counts (scheduled_tasks, dropped_tasks, violated_soft_constraints, moved_tasks). "
        "The objective adds importance-based optional drop costs, weighted soft violations in hours (daily counts scaled by per_count), "
        "and stability costs from previous starts, weight * hours / (1 + weight * hours / limit) with the stability strength weight and limit = stability_drop_cost_ratio times the drop cost; "
        "they grow with every hour moved but stay below limit. Hard constraints require zero violation. "
        "Moved counts include changed starts whenever a previous schedule exists; --no-stability disables stability costs. "
        "Use query evaluation for constraint breakdowns and query objective_policy for current weights. "
        "If no feasible schedule exists, conflicts come from a relaxed solve that permits hard violations and required drops at costs that each exceed every single soft coefficient but not necessarily a sum of soft costs; a dropped required task also pays its importance-based drop cost. "
        "When the relaxed solve finds a schedule, conflicts.status is found; "
        "conflicts.constraints lists broken hard constraints in the evaluation item shape plus related_constraint_ids, the other hard constraints referencing the same tasks; "
        "conflicts.dropped_required_tasks lists task_id, name, and reason (no_free_start if participants share no free start, otherwise conflict). "
        "The relaxed schedule minimizes these costs together with the ordinary costs, and may not reach that minimum if the relaxed solve stops before proving optimality; it is one set of changes, not the fewest conflicts or every party to a conflict, so check related_constraint_ids. "
        "Relaxing every listed item (making it soft or optional, or removing it) makes the problem solvable. "
        "If the relaxed solve finds no schedule, conflicts is {status: not_found, reason}, where reason is time_limit if the time limit stopped it "
        "and otherwise the MathOpt termination reason in lowercase, such as numerical_error. The previous schedule is kept.",
    ),
    "chat": (
        "Run a demonstration conversation turn with OpenAI",
        "Uses query, apply, schedule, or message steps, with a limit of 12. "
        "A schedule result has status optimal, feasible, no_feasible_solution, or solution_not_found. "
        "Schedule results return {status, summary, items}, {status, conflicts}, or {status, reason}, according to the status. "
        "A reason is time_limit when a time limit stops scheduling, otherwise the MathOpt termination reason in lowercase, such as numerical_error. "
        "Exit codes: 0, 2, and 3 for optimal or feasible, no_feasible_solution, and solution_not_found respectively. "
        "The assistant text is Scheduled. for optimal or feasible, No feasible solution. for no_feasible_solution, and Solution not found. for solution_not_found. "
        "An optimal or feasible result saves the working problem, previous schedule, and dialogue. "
        "A no_feasible_solution or solution_not_found result saves the working problem and dialogue and keeps the previous schedule. "
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
    emit({"rejected": [item.message for item in violations.items]})
    return 1


def scheduling(validator: Validator) -> Scheduling:
    """Wire the scheduling use case."""
    policy: ObjectivePolicy = DEFAULT_POLICY
    return Scheduling(MathOptSchedulingSolver(), validator, policy)


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
            FixedTaskData(
                id=item.id if item.id is not None else TaskId.generate(),
                **item.model_dump(exclude={"id"}),
            )
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
            emit(answer_record(summarize(updated, None)))
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
                        command_record(command, updated)
                        for command in commands
                    ]
                }
            )
            return 0


def query(path: Path, input_path: Path | None, service: Scheduling) -> int:
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
    result: AnswerResult = service.answer(request, problem, previous)
    if isinstance(result, Rejected):
        return reject(result.violations)
    emit(answer_record(result.answer))
    return 0


def schedule_output(result: OptimalSolution | FeasibleSolution) -> dict[str, object]:
    """Describe a scheduled result."""
    form: ScheduleState | None = to_schedule_state(result.schedule)
    assert form is not None
    return {
        "status": "optimal" if isinstance(result, OptimalSolution) else "feasible",
        "summary": schedule_summary_record(result.summary),
        **form.model_dump(mode="json"),
    }


def no_feasible_solution_output(result: NoFeasibleSolution) -> dict[str, object]:
    """Describe a proven absence of a feasible schedule."""
    conflicts: dict[str, object]
    reason: str
    match result.conflicts:
        case Conflicts():
            conflicts = {"status": "found", **conflicts_record(result.conflicts)}
        case ConflictsNotFound(reason=reason):
            conflicts = {"status": "not_found", "reason": reason}
    return {"status": "no_feasible_solution", "conflicts": conflicts}


def solution_not_found_output(result: SolutionNotFound) -> dict[str, object]:
    """Describe a schedule search that ended without a result."""
    return {"status": "solution_not_found", "reason": result.reason}


def schedule(path: Path, service: Scheduling, stability: bool) -> int:
    """Schedule the current problem and persist a found schedule."""
    state: State = load_state(path)
    problem: SchedulingProblem = to_problem(state.problem)
    previous: Schedule | None = to_schedule(state.previous)
    result: Solution = service.schedule(problem, previous, stability)
    match result:
        case NoFeasibleSolution():
            emit(no_feasible_solution_output(result))
            return 2
        case SolutionNotFound():
            emit(solution_not_found_output(result))
            return 3
        case OptimalSolution(schedule=schedule) | FeasibleSolution(schedule=schedule):
            save_state(
                path, state.model_copy(update={"previous": to_schedule_state(schedule)})
            )
            emit(schedule_output(result))
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
        case NoFeasibleSolution():
            updates["problem"] = to_problem_state(response.problem)
            assistant_text = "No feasible solution."
            output = no_feasible_solution_output(response.outcome)
            status = 2
        case SolutionNotFound():
            updates["problem"] = to_problem_state(response.problem)
            assistant_text = "Solution not found."
            output = solution_not_found_output(response.outcome)
            status = 3
        case OptimalSolution(schedule=schedule) | FeasibleSolution(schedule=schedule):
            updates["problem"] = to_problem_state(response.problem)
            updates["previous"] = to_schedule_state(schedule)
            assistant_text = "Scheduled."
            output = schedule_output(response.outcome)
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
            NonemptyTimeWindows(),
        )
        service: Scheduling = scheduling(validator)
        match args.command:
            case "init":
                return init(path, args.calendar, service)
            case "apply":
                return apply(path, args.file, service)
            case "query":
                return query(path, args.file, service)
            case "schedule":
                return schedule(path, service, not args.no_stability)
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
