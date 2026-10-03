from dataclasses import replace
from datetime import datetime, timedelta, timezone
import io
import json
from pathlib import Path
import sys
from unittest.mock import Mock, patch

import pytest
from ortools.math_opt.python import mathopt
from pydantic import ValidationError

from intent_to_schedule.adapter.command_line_interface.main import main
from intent_to_schedule.adapter.command_line_interface.state import (
    State,
    load_state,
    save_state,
    to_problem,
    to_problem_state,
)
from intent_to_schedule.adapter.data_model import (
    AddTimeConstraintData,
    IntervalMeasureData,
    MeasureData,
    convert_command,
    convert_measure,
    to_measure_data,
)
from intent_to_schedule.adapter.mathopt.compile import CompiledProblem, compile_problem
from intent_to_schedule.application.command import (
    AddTimeConstraint,
    Executed,
    Rejected,
    RemoveTask,
    SchedulingCommand,
)
from intent_to_schedule.application.policy import DEFAULT_POLICY
from intent_to_schedule.application.time_windows import TimeRelation, TimeWindow
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.compatibility import is_supported
from intent_to_schedule.domain.consistency import ReferencesExist, SupportedCombinations
from intent_to_schedule.domain.constraint import (
    Constraint,
    ConstraintId,
    HardConstraint,
    SoftConstraint,
)
from intent_to_schedule.domain.evaluation import Distance, Intrusion
from intent_to_schedule.domain.measure import IntervalMeasure, PointMeasure
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import FixedTask, Importance, Task, TaskId


START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
SLOT: timedelta = timedelta(minutes=30)


@pytest.fixture
def problem() -> SchedulingProblem:
    """Build two tasks on a short calendar."""
    tasks: tuple[Task, ...] = tuple(
        Task(TaskId(name), name, duration, frozenset(), Importance.LOW, True)
        for name, duration in (("first", SLOT), ("second", SLOT * 2))
    )
    grid: TimeGrid = TimeGrid(TimeInterval(START, START + SLOT * 6), SLOT)
    return SchedulingProblem(Calendar(grid, ()), (), tasks, (), ())


def solve_details(problem: SchedulingProblem) -> tuple[float, dict[TaskId, int | None]]:
    """Return the objective and selected placements."""
    compiled: CompiledProblem = compile_problem(problem, DEFAULT_POLICY)
    result: mathopt.SolveResult = mathopt.solve(
        compiled.model, mathopt.SolverType.GSCIP
    )
    assert result.termination.reason is mathopt.TerminationReason.OPTIMAL
    values: dict[mathopt.Variable, float] = result.variable_values()
    placements: dict[TaskId, int | None] = {
        task.id: next(
            (
                start
                for start, variable in compiled.placements[task.id].items()
                if values[variable] > 0.5
            ),
            None,
        )
        for task in problem.tasks
    }
    return result.objective_value(), placements


@pytest.mark.parametrize("strength", [None, *Strength])
def test_multi_task_intrusion_matches_separate_constraints(
    problem: SchedulingProblem, strength: Strength | None
) -> None:
    """Match separate objectives and placements for every requirement."""
    task_ids: frozenset[TaskId] = frozenset(task.id for task in problem.tasks)
    evaluation: Intrusion = Intrusion(
        (TimeInterval(START, START + SLOT * (2 if strength is None else 6)),)
    )
    combined: Constraint = (
        HardConstraint(ConstraintId("combined"), IntervalMeasure(task_ids), evaluation)
        if strength is None
        else SoftConstraint(
            ConstraintId("combined"), IntervalMeasure(task_ids), evaluation, strength
        )
    )
    separate: tuple[Constraint, ...] = tuple(
        replace(
            combined,
            id=ConstraintId(task.id.value),
            measure=IntervalMeasure(frozenset({task.id})),
        )
        for task in problem.tasks
    )
    targets: tuple[Constraint, ...] = tuple(
        SoftConstraint(
            ConstraintId(f"target-{task.id.value}"),
            PointMeasure(task.id),
            Distance(START + SLOT * (index + (2 if strength is None else 0))),
            Strength.WEAK,
        )
        for index, task in enumerate(problem.tasks)
    )
    expected: tuple[float, dict[TaskId, int | None]] = (
        0.0 if strength is None else 1.5 * DEFAULT_POLICY.weight(strength),
        {
            task.id: index + (2 if strength is None else 0)
            for index, task in enumerate(problem.tasks)
        },
    )
    assert solve_details(replace(problem, constraints=(combined, *targets))) == expected
    assert (
        solve_details(replace(problem, constraints=(*separate, *targets))) == expected
    )


def test_multi_task_soft_intrusion_matches_separate_drops(
    problem: SchedulingProblem,
) -> None:
    """Match task dropping under summed soft penalties."""
    tasks: tuple[Task, ...] = (
        replace(problem.tasks[0], required=False),
        replace(problem.tasks[1], required=False, duration=SLOT * 3),
    )
    combined: SoftConstraint = SoftConstraint(
        ConstraintId("combined"),
        IntervalMeasure(frozenset(task.id for task in tasks)),
        Intrusion((problem.calendar.grid.horizon,)),
        Strength.NORMAL,
    )
    separate: tuple[Constraint, ...] = tuple(
        replace(
            combined,
            id=ConstraintId(task.id.value),
            measure=IntervalMeasure(frozenset({task.id})),
        )
        for task in tasks
    )
    targets: tuple[Constraint, ...] = tuple(
        SoftConstraint(
            ConstraintId(f"target-{task.id.value}"),
            PointMeasure(task.id),
            Distance(START),
            Strength.WEAK,
        )
        for task in tasks
    )
    expected: tuple[float, dict[TaskId, int | None]] = (
        7.5,
        {tasks[0].id: 0, tasks[1].id: None},
    )
    assert (
        solve_details(replace(problem, tasks=tasks, constraints=(combined, *targets)))
        == expected
    )
    assert (
        solve_details(replace(problem, tasks=tasks, constraints=(*separate, *targets)))
        == expected
    )


def test_multi_task_intrusion_sums_rounded_fixed_and_movable_overlap(
    problem: SchedulingProblem,
) -> None:
    """Count overlapping fixed and movable occupancy separately."""
    first: Task = problem.tasks[0]
    second: Task = problem.tasks[1]
    fixed: FixedTask = FixedTask(
        second.id,
        second.name,
        START + timedelta(minutes=10),
        second.duration,
        second.participant_ids,
    )
    combined: SoftConstraint = SoftConstraint(
        ConstraintId("combined"),
        IntervalMeasure(frozenset({first.id, fixed.id})),
        Intrusion((problem.calendar.grid.horizon,)),
        Strength.NORMAL,
    )
    separate: tuple[Constraint, ...] = tuple(
        replace(
            combined,
            id=ConstraintId(task_id.value),
            measure=IntervalMeasure(frozenset({task_id})),
        )
        for task_id in (first.id, fixed.id)
    )
    target: SoftConstraint = SoftConstraint(
        ConstraintId("target"),
        PointMeasure(first.id),
        Distance(START),
        Strength.WEAK,
    )
    given: SchedulingProblem = replace(problem, tasks=(first,), fixed_tasks=(fixed,))
    expected: tuple[float, dict[TaskId, int | None]] = (10.0, {first.id: 0})
    assert solve_details(replace(given, constraints=(combined, target))) == expected
    assert solve_details(replace(given, constraints=(*separate, target))) == expected


def test_multi_task_hard_intrusion_matches_separate_infeasibility(
    problem: SchedulingProblem,
) -> None:
    """Reject a hard region covering every possible placement."""
    combined: HardConstraint = HardConstraint(
        ConstraintId("combined"),
        IntervalMeasure(frozenset(task.id for task in problem.tasks)),
        Intrusion((problem.calendar.grid.horizon,)),
    )
    separate: tuple[Constraint, ...] = tuple(
        replace(
            combined,
            id=ConstraintId(task.id.value),
            measure=IntervalMeasure(frozenset({task.id})),
        )
        for task in problem.tasks
    )
    constraints: tuple[Constraint, ...]
    for constraints in ((combined,), separate):
        compiled: CompiledProblem = compile_problem(
            replace(problem, constraints=constraints), DEFAULT_POLICY
        )
        result: mathopt.SolveResult = mathopt.solve(
            compiled.model, mathopt.SolverType.GSCIP
        )
        assert result.termination.reason is mathopt.TerminationReason.INFEASIBLE


@pytest.mark.parametrize("fixed", [False, True])
def test_remove_task_keeps_remaining_interval_targets(
    problem: SchedulingProblem, fixed: bool
) -> None:
    """Shrink interval targets and remove the last constraint."""
    first: Task = problem.tasks[0]
    if fixed:
        problem = replace(
            problem,
            tasks=problem.tasks[1:],
            fixed_tasks=(
                FixedTask(
                    first.id, first.name, START, first.duration, first.participant_ids
                ),
            ),
        )
    measure: IntervalMeasure = IntervalMeasure(
        frozenset({first.id, problem.tasks[-1].id})
    )
    constraint: HardConstraint = HardConstraint(
        ConstraintId("combined"), measure, Intrusion(())
    )
    result: Executed | Rejected = RemoveTask(first.id).execute(
        replace(problem, constraints=(constraint,))
    )
    assert isinstance(result, Executed)
    remaining: IntervalMeasure = IntervalMeasure(frozenset({problem.tasks[-1].id}))
    assert result.problem.constraints == (replace(constraint, measure=remaining),)
    assert measure.without_task(TaskId("unrelated")) == measure
    result = RemoveTask(problem.tasks[-1].id).execute(result.problem)
    assert isinstance(result, Executed)
    assert result.problem.constraints == ()


def test_interval_measure_round_trips_sorted_targets() -> None:
    """Round trip interval targets in deterministic order."""
    data: IntervalMeasureData = IntervalMeasureData.model_validate(
        {"kind": "interval", "task_ids": ["second", "first", "first"]}
    )
    restored: IntervalMeasureData = IntervalMeasureData.model_validate_json(
        data.model_dump_json()
    )
    measure: IntervalMeasure = IntervalMeasure(
        frozenset({TaskId("first"), TaskId("second")})
    )
    assert convert_measure(restored) == measure
    output: MeasureData = to_measure_data(measure)
    assert isinstance(output, IntervalMeasureData)
    assert output.model_dump(mode="json") == {
        "kind": "interval",
        "task_ids": ["first", "second"],
    }
    assert (
        convert_measure(
            IntervalMeasureData.model_validate_json(output.model_dump_json())
        )
        == measure
    )


@pytest.mark.parametrize(
    "requirement", [{"kind": "hard"}, {"kind": "soft", "strength": "strong"}]
)
def test_add_time_constraint_round_trip_generates_one_identifier(
    requirement: dict[str, str],
) -> None:
    """Round trip a multi-task command with one generated identifier."""
    data: AddTimeConstraintData = AddTimeConstraintData.model_validate(
        {
            "kind": "add_time_constraint",
            "task_ids": ["second", "first", "first"],
            "relation": "within",
            "windows": [{}],
            "requirement": requirement,
        }
    )
    restored: AddTimeConstraintData = AddTimeConstraintData.model_validate_json(
        data.model_dump_json()
    )
    assert restored == data
    generation: Mock
    with patch.object(
        ConstraintId, "generate", return_value=ConstraintId("created")
    ) as generation:
        command: SchedulingCommand = convert_command(restored)
    generation.assert_called_once_with()
    assert isinstance(command, AddTimeConstraint)
    assert command.constraint_id == ConstraintId("created")
    assert command.task_ids == frozenset({TaskId("first"), TaskId("second")})


@pytest.mark.parametrize("data_type", [IntervalMeasureData, AddTimeConstraintData])
def test_empty_task_ids_are_rejected_by_data_models(
    data_type: type[IntervalMeasureData] | type[AddTimeConstraintData],
) -> None:
    """Reject an empty target list with an explanation."""
    data: dict[str, object] = {"kind": "interval", "task_ids": []}
    if data_type is AddTimeConstraintData:
        data.update(
            kind="add_time_constraint",
            relation="within",
            windows=[{}],
            requirement={"kind": "hard"},
        )
    error: pytest.ExceptionInfo[ValidationError]
    with pytest.raises(ValidationError, match="task_ids") as error:
        data_type.model_validate(data)
    assert "at least" in str(error.value)


def test_empty_interval_measure_is_rejected() -> None:
    """Reject an interval measure without targets."""
    with pytest.raises(ValueError, match="at least one Task"):
        IntervalMeasure(frozenset())


def test_empty_command_is_rejected(problem: SchedulingProblem) -> None:
    """Reject an application command without targets."""
    command: AddTimeConstraint = AddTimeConstraint(
        ConstraintId("empty"),
        frozenset(),
        TimeRelation.WITHIN,
        (TimeWindow(None, None, None),),
        None,
    )
    result: Executed | Rejected = command.execute(problem)
    assert isinstance(result, Rejected)
    assert "at least one Task" in result.violations.items[0].message


def test_multi_task_interval_validators_check_every_target(
    problem: SchedulingProblem,
) -> None:
    """Validate compatibility and every referenced Task."""
    measure: IntervalMeasure = IntervalMeasure(
        frozenset({problem.tasks[0].id, TaskId("missing-one"), TaskId("missing-two")})
    )
    evaluation: Intrusion = Intrusion(())
    constraint: HardConstraint = HardConstraint(
        ConstraintId("combined"), measure, evaluation
    )
    given: SchedulingProblem = replace(problem, constraints=(constraint,))
    assert is_supported(measure, evaluation)
    assert not is_supported(measure, Distance(START))
    assert SupportedCombinations().validate(given).is_empty
    assert {item.message for item in ReferencesExist().validate(given).items} == {
        "Constraint combined references missing task id missing-one.",
        "Constraint combined references missing task id missing-two.",
    }


@pytest.mark.parametrize("task_ids", [["first", "second"], [], ["first", "missing"]])
def test_apply_and_query_multi_task_constraint(
    problem: SchedulingProblem,
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
    monkeypatch: pytest.MonkeyPatch,
    task_ids: list[str],
) -> None:
    """Persist one constraint or reject invalid targets atomically."""
    path: Path = tmp_path / "state.json"
    save_state(
        path, State(problem=to_problem_state(problem), previous=None, dialogue=())
    )
    before: bytes = path.read_bytes()
    command: dict[str, object] = {
        "kind": "add_time_constraint",
        "task_ids": task_ids,
        "relation": "avoid",
        "windows": [{"time_range": {"start": "09:00", "end": "10:00"}}],
        "requirement": {"kind": "soft", "strength": "strong"},
    }
    monkeypatch.setattr(sys, "stdin", io.StringIO(json.dumps({"commands": [command]})))
    status: int = main(["--state", str(path), "apply"])
    output: dict[str, object] = json.loads(capsys.readouterr().out)
    if not task_ids or "missing" in task_ids:
        assert status == 1
        assert (
            "at least" if not task_ids else "missing task id missing"
        ) in json.dumps(output)
        assert path.read_bytes() == before
        return
    assert status == 0
    executed: object = output["executed"]
    assert isinstance(executed, list) and len(executed) == 1
    record: object = executed[0]
    assert isinstance(record, dict)
    constraint_id: object = record["constraint_id"]
    assert isinstance(constraint_id, str)
    stored: SchedulingProblem = to_problem(load_state(path).problem)
    assert len(stored.constraints) == 1
    assert stored.constraints[0].id.value == constraint_id
    assert stored.constraints[0].measure == IntervalMeasure(
        frozenset(TaskId(value) for value in task_ids)
    )
    selected: list[str]
    for selected in (["first"], ["second"], ["first", "second"]):
        monkeypatch.setattr(
            sys,
            "stdin",
            io.StringIO(
                json.dumps({"kind": "constraints", "filter": {"task_ids": selected}})
            ),
        )
        assert main(["--state", str(path), "query"]) == 0
        output = json.loads(capsys.readouterr().out)
        items: object = output["items"]
        assert output["total"] == 1
        assert isinstance(items, list) and len(items) == 1
        record = items[0]
        assert isinstance(record, dict)
        assert record["id"] == constraint_id
        assert record["measure"] == {
            "kind": "interval",
            "task_ids": ["first", "second"],
        }
