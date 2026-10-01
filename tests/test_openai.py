from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from intent_to_schedule.adapter.data_model import (
    AddConstraintOutput,
    AddTaskOutput,
    AggregateMeasureOutput,
    CountOutput,
    DependencyMeasureOutput,
    DistanceOutput,
    DurationOutput,
    ExcessOutput,
    HardConstraintOutput,
    InstantOutput,
    IntervalMeasureOutput,
    IntrusionOutput,
    OutputModel,
    PointMeasureOutput,
    RemoveConstraintOutput,
    RemoveTaskOutput,
    ReplaceTaskOutput,
    ShortfallOutput,
    SoftConstraintOutput,
    TaskOutput,
    TimeIntervalOutput,
)
from intent_to_schedule.adapter.openai.translate import (
    AmbiguousOutput,
    ConstraintCommandsOutput,
    ConstraintTranslationOutput,
    ElementCommandsOutput,
    ElementTranslationOutput,
    OpenAICommandTranslator,
    convert_constraint_output,
    convert_element_output,
)
from intent_to_schedule.application.command import AddConstraint, AddTask, RemoveConstraint, RemoveTask, ReplaceTask
from intent_to_schedule.application.translate import Ambiguous, Speaker, Translated, Utterance
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.consistency import ConsistencyError, ReferencesExist
from intent_to_schedule.domain.constraint import ConstraintId, HardConstraint, SoftConstraint
from intent_to_schedule.domain.evaluation import Distance, Excess, Intrusion, Shortfall
from intent_to_schedule.domain.measure import AggregateMeasure, AggregateQuantity, DependencyMeasure, IntervalMeasure, PointMeasure
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.schedule import Schedule, ScheduledTask
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId


START: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)


def _task_output(participant_ids: tuple[PersonId, ...] = ()) -> TaskOutput:
    """Create a structured Task for tests."""
    return TaskOutput(name="Review", duration=timedelta(hours=1), participant_ids=participant_ids, importance="high", required=True, stability="weak")


def _problem() -> SchedulingProblem:
    """Create a small scheduling problem."""
    calendar: Calendar = Calendar(TimeGrid(TimeInterval(START, START + timedelta(days=1)), timedelta(minutes=30)), (), ())
    person: Person = Person(PersonId("p1"), "Alex")
    task: Task = Task(TaskId("t1"), "Existing", timedelta(hours=1), frozenset({person.id}), Importance.MEDIUM, False)
    return SchedulingProblem(calendar, (person,), (task,), ())


class _FakeResponses:
    """Return canned parsed model outputs."""

    def __init__(self, outputs: list[OutputModel]) -> None:
        self.outputs: list[OutputModel] = outputs
        self.calls: list[tuple[str, list[dict[str, str]], type[OutputModel]]] = []

    def parse(self, *, model: str, input: list[dict[str, str]], text_format: type[OutputModel]) -> SimpleNamespace:
        """Record the request and return the next output."""
        self.calls.append((model, [message.copy() for message in input], text_format))
        output: OutputModel = self.outputs.pop(0)
        assert isinstance(output, text_format)
        return SimpleNamespace(output_parsed=output)


class _FakeClient:
    """Expose fake responses to the translator."""

    def __init__(self, outputs: list[OutputModel]) -> None:
        self.responses: _FakeResponses = _FakeResponses(outputs)


def test_convert_element_output() -> None:
    original: TaskId = TaskId("t1")
    person: PersonId = PersonId("p1")
    output: ElementTranslationOutput = ElementTranslationOutput(result=ElementCommandsOutput(kind="translated", commands=(
        AddTaskOutput(kind="add_task", task=_task_output((person, person))),
        ReplaceTaskOutput(kind="replace_task", task_id=original, replacement=_task_output((person,))),
        RemoveTaskOutput(kind="remove_task", task_id=original),
    )))
    with patch.object(TaskId, "generate", return_value=TaskId("new")):
        converted: tuple[AddTask | ReplaceTask | RemoveTask, ...] | Ambiguous = convert_element_output(output)
    assert converted == (
        AddTask(Task(TaskId("new"), "Review", timedelta(hours=1), frozenset({person}), Importance.HIGH, True, Strength.WEAK)),
        ReplaceTask(Task(original, "Review", timedelta(hours=1), frozenset({person}), Importance.HIGH, True, Strength.WEAK)),
        RemoveTask(original),
    )
    assert convert_element_output(ElementTranslationOutput(result=AmbiguousOutput(kind="ambiguous", question="Who?"))) == Ambiguous("Who?")


def test_convert_constraint_output() -> None:
    task_id: TaskId = TaskId("t1")
    other_id: TaskId = TaskId("t2")
    region: TimeIntervalOutput = TimeIntervalOutput(start=START, end=START + timedelta(hours=1))
    output: ConstraintTranslationOutput = ConstraintTranslationOutput(result=ConstraintCommandsOutput(kind="translated", commands=(
        AddConstraintOutput(kind="add_constraint", constraint=HardConstraintOutput(kind="hard", measure=PointMeasureOutput(kind="point", task_id=task_id), evaluation=DistanceOutput(kind="distance", target=InstantOutput(kind="instant", value=START)))),
        AddConstraintOutput(kind="add_constraint", constraint=SoftConstraintOutput(kind="soft", measure=IntervalMeasureOutput(kind="interval", task_id=task_id), evaluation=IntrusionOutput(kind="intrusion", region=(region,)), strength="weak")),
        AddConstraintOutput(kind="add_constraint", constraint=SoftConstraintOutput(kind="soft", measure=DependencyMeasureOutput(kind="dependency", from_task_id=task_id, to_task_id=other_id), evaluation=ShortfallOutput(kind="shortfall", lower=DurationOutput(kind="duration", value=timedelta(hours=1))), strength="strong")),
        AddConstraintOutput(kind="add_constraint", constraint=SoftConstraintOutput(kind="soft", measure=AggregateMeasureOutput(kind="aggregate", task_ids=(task_id, other_id, task_id), quantity="count"), evaluation=ExcessOutput(kind="excess", upper=CountOutput(kind="count", value=2)), strength="normal")),
        RemoveConstraintOutput(kind="remove_constraint", constraint_id=ConstraintId("old")),
    )))
    with patch.object(ConstraintId, "generate", side_effect=[ConstraintId(str(index)) for index in range(4)]):
        converted: tuple[AddConstraint | RemoveConstraint, ...] | Ambiguous = convert_constraint_output(output)
    assert converted == (
        AddConstraint(HardConstraint(ConstraintId("0"), PointMeasure(task_id), Distance(START))),
        AddConstraint(SoftConstraint(ConstraintId("1"), IntervalMeasure(task_id), Intrusion((TimeInterval(region.start, region.end),)), Strength.WEAK)),
        AddConstraint(SoftConstraint(ConstraintId("2"), DependencyMeasure(task_id, other_id), Shortfall(timedelta(hours=1)), Strength.STRONG)),
        AddConstraint(SoftConstraint(ConstraintId("3"), AggregateMeasure(frozenset({task_id, other_id}), AggregateQuantity.COUNT), Excess(2), Strength.NORMAL)),
        RemoveConstraint(ConstraintId("old")),
    )
    assert convert_constraint_output(ConstraintTranslationOutput(result=AmbiguousOutput(kind="ambiguous", question="When?"))) == Ambiguous("When?")


def test_translate_retries_rejected_command_and_uses_updated_snapshot() -> None:
    rejected: ElementTranslationOutput = ElementTranslationOutput(result=ElementCommandsOutput(kind="translated", commands=(RemoveTaskOutput(kind="remove_task", task_id=TaskId("missing")),)))
    accepted: ElementTranslationOutput = ElementTranslationOutput(result=ElementCommandsOutput(kind="translated", commands=(AddTaskOutput(kind="add_task", task=_task_output((PersonId("p1"),))),)))
    constraint: ConstraintTranslationOutput = ConstraintTranslationOutput(result=ConstraintCommandsOutput(kind="translated", commands=(AddConstraintOutput(kind="add_constraint", constraint=HardConstraintOutput(kind="hard", measure=PointMeasureOutput(kind="point", task_id=TaskId("t1")), evaluation=DistanceOutput(kind="distance", target=InstantOutput(kind="instant", value=START))),),)))
    client: _FakeClient = _FakeClient([rejected, accepted, constraint])
    previous: Schedule = Schedule((ScheduledTask(TaskId("t1"), START),), frozenset())
    dialogue: tuple[Utterance, ...] = (Utterance(Speaker.USER, "Plan a review"), Utterance(Speaker.ASSISTANT, "Okay"), Utterance(Speaker.USER, "Make it tomorrow"))
    with patch.object(TaskId, "generate", return_value=TaskId("new")), patch.object(ConstraintId, "generate", return_value=ConstraintId("c1")):
        result: Translated | Ambiguous = OpenAICommandTranslator(client, "test-model", (ReferencesExist(),)).translate(dialogue, _problem(), previous)
    assert isinstance(result, Translated)
    assert len(result.commands) == 2
    assert isinstance(result.commands[0], AddTask)
    assert isinstance(result.commands[1], AddConstraint)
    assert len(client.responses.calls) == 3
    assert client.responses.calls[0][1][1:] == [
        {"role": "user", "content": "Plan a review"},
        {"role": "assistant", "content": "Okay"},
        {"role": "user", "content": "Make it tomorrow"},
    ]
    assert '"current_time"' in client.responses.calls[0][1][0]["content"]
    assert '"previous_schedule"' in client.responses.calls[0][1][0]["content"]
    assert "Task does not exist" in client.responses.calls[1][1][-1]["content"]
    assert client.responses.calls[1][1][-2]["role"] == "assistant"
    assert '"new"' in client.responses.calls[2][1][0]["content"]


def test_translate_retries_validator_violations_then_raises() -> None:
    invalid: ElementTranslationOutput = ElementTranslationOutput(result=ElementCommandsOutput(kind="translated", commands=(AddTaskOutput(kind="add_task", task=_task_output((PersonId("missing"),))),)))
    client: _FakeClient = _FakeClient([invalid, invalid, invalid])
    translator: OpenAICommandTranslator = OpenAICommandTranslator(client, "test-model", (ReferencesExist(),))
    with pytest.raises(ConsistencyError, match="missing person id"):
        translator.translate((Utterance(Speaker.USER, "Add a review"),), _problem(), None)
    assert len(client.responses.calls) == 3
    assert "missing person id" in client.responses.calls[1][1][-1]["content"]
