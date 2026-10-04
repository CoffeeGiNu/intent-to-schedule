"""Tests for OpenAI conversation steps."""

import json
from datetime import datetime, timedelta, timezone
from typing import cast
from unittest.mock import MagicMock

import openai
import pytest
from openai.lib._parsing._responses import type_to_text_format_param
from openai.types.responses import ResponseFormatTextConfigParam

from intent_to_schedule.adapter.openai.translate import (
    OpenAIStepTranslator,
    StepOutput,
    convert_step_output,
)
from intent_to_schedule.application.command import (
    AddTask,
    Executed,
    Rejected,
    RemoveTask,
)
from intent_to_schedule.application.query import (
    Answered,
    PeopleAnswer,
    Summary,
    SummaryQuery,
)
from intent_to_schedule.application.translate import (
    ApplyRecord,
    ApplyStep,
    MessageStep,
    QueryRecord,
    QueryStep,
    SolveStep,
    Speaker,
    Step,
    StepRecord,
    Utterance,
)
from intent_to_schedule.domain.calendar import Calendar, TimeGrid, TimeInterval
from intent_to_schedule.domain.consistency import Violation, Violations
from intent_to_schedule.domain.person import Person, PersonId
from intent_to_schedule.domain.problem import SchedulingProblem
from intent_to_schedule.domain.strength import Strength
from intent_to_schedule.domain.task import Importance, Task, TaskId


def make_summary() -> Summary:
    """Build a small summary."""
    start: datetime = datetime(2026, 10, 1, 9, tzinfo=timezone.utc)
    return Summary(
        TimeGrid(TimeInterval(start, start + timedelta(hours=3)), timedelta(hours=1)),
        1,
        0,
        0,
        0,
        True,
    )


@pytest.mark.parametrize(
    "data, expected",
    [
        (
            {"result": {"kind": "message", "text": "When works for you?"}},
            MessageStep("When works for you?"),
        ),
        ({"result": {"kind": "solve", "stability": True}}, SolveStep(True)),
        ({"result": {"kind": "solve", "stability": False}}, SolveStep(False)),
        (
            {
                "result": {
                    "kind": "apply",
                    "commands": [{"kind": "remove_task", "task_id": "review"}],
                }
            },
            ApplyStep((RemoveTask(TaskId("review")),)),
        ),
        (
            {"result": {"kind": "query", "query": {"kind": "summary"}}},
            QueryStep(SummaryQuery()),
        ),
    ],
)
def test_convert_step_output(data: dict[str, object], expected: Step) -> None:
    """Convert terminal, query, and command steps to application values."""
    assert convert_step_output(StepOutput.model_validate(data)) == expected


def test_new_task_gets_generated_id() -> None:
    """Assign new task identifiers through command conversion."""
    output: StepOutput = StepOutput.model_validate(
        {
            "result": {
                "kind": "apply",
                "commands": [
                    {
                        "kind": "add_task",
                        "task": {
                            "name": "Review",
                            "duration": "PT1H",
                            "participant_ids": [],
                            "importance": "high",
                            "required": True,
                            "stability": "normal",
                        },
                    }
                ],
            }
        }
    )
    first: Step = convert_step_output(output)
    second: Step = convert_step_output(output)
    assert isinstance(first, ApplyStep) and isinstance(second, ApplyStep)
    assert isinstance(first.commands[0], AddTask) and isinstance(
        second.commands[0], AddTask
    )
    assert first.commands[0].task.id != second.commands[0].task.id


def test_translate_sends_summary_dialogue_and_step_results() -> None:
    """Send compact JSON context and make one parsed response request."""
    summary: Summary = make_summary()
    task: Task = Task(
        TaskId("review"),
        "Review",
        timedelta(hours=1),
        frozenset(),
        Importance.HIGH,
        True,
        Strength.NORMAL,
    )
    problem: SchedulingProblem = SchedulingProblem(
        Calendar(summary.grid, ()), (), (task,), (), ()
    )
    rejected: Rejected = Rejected(
        Violations((Violation("Task missing does not exist"),))
    )
    people: PeopleAnswer = PeopleAnswer((Person(PersonId("alice"), "Alice"),), 1)
    steps: tuple[StepRecord, ...] = (
        QueryRecord(QueryStep(SummaryQuery()), Answered(people)),
        ApplyRecord(ApplyStep((AddTask(task),)), Executed(problem)),
        ApplyRecord(ApplyStep((RemoveTask(TaskId("missing")),)), rejected),
        QueryRecord(QueryStep(SummaryQuery()), rejected),
    )
    client: MagicMock = MagicMock()
    client.responses.parse.return_value.output_parsed = StepOutput.model_validate(
        {"result": {"kind": "message", "text": "done"}}
    )
    now: datetime = datetime(2026, 9, 30, 10, tzinfo=timezone.utc)
    dialogue: tuple[Utterance, ...] = (
        Utterance(Speaker.USER, "Earlier request"),
        Utterance(Speaker.ASSISTANT, "Let me confirm"),
        Utterance(Speaker.USER, "Yes, please"),
    )
    result: Step = OpenAIStepTranslator(
        cast(openai.OpenAI, client), "demo-model", lambda: now
    ).translate(dialogue, summary, steps)
    assert result == MessageStep("done")
    client.responses.parse.assert_called_once()
    arguments: dict[str, object] = client.responses.parse.call_args.kwargs
    assert arguments["model"] == "demo-model"
    assert arguments["text_format"] is StepOutput
    messages: list[dict[str, str]] = cast(list[dict[str, str]], arguments["input"])
    assert messages[1:] == [
        {"role": item.speaker.value, "content": item.text} for item in dialogue
    ]
    context: dict[str, object] = json.loads(
        messages[0]["content"].split("\nContext: ", 1)[1]
    )
    assert context["current_time"] == now.isoformat()
    assert context["summary"] == {
        "kind": "summary",
        "grid": {
            "horizon": {
                "start": "2026-10-01T09:00:00Z",
                "end": "2026-10-01T12:00:00Z",
            },
            "slot": "PT1H",
        },
        "counts": {"people": 1, "tasks": 0, "fixed_tasks": 0, "constraints": 0},
        "has_previous": True,
    }
    assert "problem" not in context and "previous_schedule" not in context
    records: list[dict[str, object]] = cast(list[dict[str, object]], context["steps"])
    assert records == [
        {
            "kind": "query",
            "result": {
                "answered": {
                    "kind": "people",
                    "items": [{"id": "alice", "name": "Alice"}],
                    "total": 1,
                    "truncated": False,
                }
            },
        },
        {
            "kind": "apply",
            "result": {
                "executed": [
                    {"kind": "add_task", "task_id": "review", "name": "Review"}
                ]
            },
        },
        {"kind": "apply", "result": {"rejected": ["Task missing does not exist"]}},
        {"kind": "query", "result": {"rejected": ["Task missing does not exist"]}},
    ]


def test_missing_parsed_output_raises_without_retry() -> None:
    """Fail once when the model returns no structured output."""
    client: MagicMock = MagicMock()
    client.responses.parse.return_value.output_parsed = None
    with pytest.raises(ValueError, match="no parsed"):
        OpenAIStepTranslator(
            cast(openai.OpenAI, client),
            "demo-model",
            lambda: datetime(2026, 10, 1, tzinfo=timezone.utc),
        ).translate((), make_summary(), ())
    client.responses.parse.assert_called_once()


def test_installed_structured_output_helper_marks_defaulted_fields_required() -> None:
    """Inspect the installed helper's strict schema without a request."""
    text_format: ResponseFormatTextConfigParam = type_to_text_format_param(StepOutput)
    assert text_format["type"] == "json_schema"
    schema: dict[str, object] = text_format["schema"]
    definitions: dict[str, dict[str, object]] = cast(
        dict[str, dict[str, object]], schema["$defs"]
    )
    name: str
    for name in (
        "PeopleQueryData",
        "TasksQueryData",
        "ConstraintsQueryData",
        "PreviousScheduleQueryData",
        "AvailableStartsQueryData",
        "NewConstraintData",
        "ConstraintData",
        "TimeWindowConditionData",
        "TimeBoundConditionData",
        "TaskGapConditionData",
        "DailyLimitConditionData",
    ):
        definition: dict[str, object] = definitions[name]
        properties: dict[str, dict[str, object]] = cast(
            dict[str, dict[str, object]], definition["properties"]
        )
        assert set(cast(list[str], definition["required"])) == set(properties)
        assert definition["additionalProperties"] is False
    window_properties: dict[str, dict[str, object]] = cast(
        dict[str, dict[str, object]], definitions["TimeWindowData"]["properties"]
    )
    assert all("default" not in value for value in window_properties.values())
    assert "oneOf" not in json.dumps(schema)
    assert "discriminator" not in json.dumps(schema)
