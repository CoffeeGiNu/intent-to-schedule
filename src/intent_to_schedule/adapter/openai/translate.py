import json
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal

import openai
from openai.types.responses import (
    EasyInputMessageParam,
    ParsedResponse,
    ResponseInputParam,
)

from intent_to_schedule.adapter.data_model import (
    CommandData,
    DataModel,
    QueryData,
    answer_record,
    command_record,
    convert_command,
    convert_query,
)
from intent_to_schedule.application.command import Rejected
from intent_to_schedule.application.query import Summary
from intent_to_schedule.application.translate import (
    ApplyRecord,
    ApplyStep,
    MessageStep,
    QueryStep,
    SolveStep,
    Speaker,
    Step,
    StepRecord,
    StepTranslator,
    Utterance,
)


class QueryOutput(DataModel):
    """Structured output for a query step."""

    kind: Literal["query"]
    query: QueryData


class ApplyOutput(DataModel):
    """Structured output for an apply step."""

    kind: Literal["apply"]
    commands: tuple[CommandData, ...]


class SolveOutput(DataModel):
    """Structured output for a solve step."""

    kind: Literal["solve"]
    stability: bool


class MessageOutput(DataModel):
    """Structured output for a message step."""

    kind: Literal["message"]
    text: str


class StepOutput(DataModel):
    """Root of the structured output for a translation step."""

    result: QueryOutput | ApplyOutput | SolveOutput | MessageOutput


def convert_step_output(output: StepOutput) -> Step:
    """Convert structured output to a translation step."""
    query: QueryData
    commands: tuple[CommandData, ...]
    stability: bool
    text: str
    match output.result:
        case QueryOutput(query=query):
            return QueryStep(convert_query(query))
        case ApplyOutput(commands=commands):
            return ApplyStep(tuple(convert_command(command) for command in commands))
        case SolveOutput(stability=stability):
            return SolveStep(stability)
        case MessageOutput(text=text):
            return MessageStep(text)


_STEP_PROMPT: str = (
    "Choose one next step for the latest user utterance: query, apply, solve, or message. "
    "Context contains the current time, a summary of the working problem, and results of earlier steps in this turn. "
    "Use query to discover details and identifiers; use apply to add, replace, or remove tasks and constraints. "
    "After applying all requested changes, use solve to schedule and end the turn. "
    "Use message to answer a question or ask for clarification and end the turn. "
    "Changes are saved only when solve is reached; message discards edits made during this turn. "
    "Use only person, task, and constraint identifiers returned by queries or successful apply results; never invent identifiers. "
    "Query people to resolve names, then tasks to resolve movable or fixed events by name, date, and participants. "
    "Use select one when exactly one match is needed; narrow ambiguous matches or ask the user instead of guessing. "
    "Query constraints before changing them and previous_schedule for previous task placements. "
    "previous_schedule is the last obtained solution and may differ from the current problem. "
    "available_starts gives free candidates without movable tasks or constraints; solve makes the final decision. "
    "After adding a task, read its generated identifier from the executed result before referencing it in another apply. "
    "Rejected steps leave the working problem unchanged; use their explanations to correct the next step or ask the user. "
    "A task with start is fixed; without start it is movable. "
    "For absences or appointments missing from the calendar, such as a health check, half day off, external training, or dentist appointment, use add_task with name, start, duration, and participant_ids for that person. "
    "A fixed task occupies its participants for both existing and future tasks; use remove_task to undo it. "
    "Do not express such absences as avoid constraints; use avoid only when the time is free but some tasks, such as meetings, must stay out of it. "
    "Use replace_task with start and the same identifier to fix an existing task at that time; its constraints remain. "
    "To make a fixed task movable, use replace_task without start and include importance, required, and stability. "
    "Fixed tasks omit importance, required, and stability; constraints can reference them. "
    "Importance low, medium, or high means how much it matters to do the Task at all. "
    "Required means the Task must be scheduled, including expressions such as 'must' or 'definitely'; this does not make its placement preferences hard. "
    "Stability means how strongly to keep the Task at its previous time. "
    'Set solve stability false only for a clear request to rebuild the whole schedule, such as "redo everything" or "start over"; true for partial changes and additions. '
    "If it is unclear whether existing placements should be rebuilt, use message to ask whether rearranging the whole schedule is acceptable. "
    "Use add_constraint with a constraint containing requirement and condition, plus an optional id and label. "
    "The response returns constraint_id; label describes the request and has no scheduling effect. "
    "Use requirement kind hard for conditions that must hold if the task is scheduled; hard does not require scheduling the task. "
    "Use requirement kind soft with strength weak, normal, or strong, in increasing penalty weight, for preferences. "
    "Soft permits violations and trades their weighted costs against other preferences, dropping optional tasks, and moving previous placements; strong does not make a condition hard. "
    "The condition kind is time_window, time_bound, task_gap, or daily_limit. "
    "Task references must exist; task_ids must be non-empty. Group tasks sharing a condition in one constraint. "
    "Unscheduled tasks have no condition violation and contribute zero to daily limits; task_gap applies only when both tasks are scheduled. "
    "For time-of-day or weekday wishes use time_window with task_ids, relation, and windows. "
    "within keeps the whole task inside the combined windows; avoid prohibits or penalizes overlap. These apply to the whole task, not just its start. "
    "Soft time_window violations sum hours outside for within or overlapping for avoid, across listed tasks. "
    "A window combines date_range, weekdays, and time_range; multiple windows are alternatives. Write every field explicitly. "
    'date_range is "horizon" for every date of the horizon, or start and end dates that include the start date and exclude the end date. '
    "Use lowercase English weekday names; list all seven for every day; an empty weekday array matches no day. "
    "Time ranges use HH:MM, include the start, and exclude the end; start 00:00 with end 24:00 is the whole day, and end 24:00 means the end of the day. "
    "End must be later than start; split overnight wishes into two windows. "
    "Window times have no offset and use the horizon's starting offset; resolve relative dates using current_time. "
    "There are no default hours for words like afternoon; ask for missing bounds. "
    "Windows are clipped to the horizon and merged before rounding inward for within or outward for avoid; empty expansion is rejected. "
    "Changed boundaries produce a rounding note on add or replace; the entered windows remain stored. "
    "For a deadline or start-time request use time_bound with task_ids, boundary start or end, relation, and at. "
    "at_or_before is an inclusive latest time; at_or_after is an inclusive earliest time; at is exact equality. "
    "A deadline uses boundary end and at_or_before; an earliest start uses boundary start and at_or_after. "
    "at is a date and time with an offset; soft violations sum hours late, early, or away from the target, respectively. "
    "Use task_gap with from_task_id, to_task_id, relation, and a non-negative gap duration such as PT30M. "
    "The gap runs from the first task's end to the second task's start; at_least is an inclusive minimum and exactly is equality. "
    "at_least with PT0S orders tasks; exactly with PT0S and a soft requirement prefers starting immediately after a fixed appointment. "
    "Soft gap violations are hours short of the minimum or hours away from the exact gap. "
    "Use daily_limit with task_ids, quantity, and maximum. count needs a non-negative JSON integer; total_duration needs a non-negative duration string such as PT4H. "
    "The maximum is inclusive and zero is allowed; soft violations sum daily excess counts or hours. "
    "Daily limits group by start date in the horizon's starting offset and count the whole duration on that date, even across midnight. "
    "They cover every calendar date intersecting the half-open horizon, even dates without slot starts; an end exactly at midnight excludes that following date. "
    "A listed fixed task counts if its real start date is covered, even if its start time is outside the horizon; durations are not clipped. Tasks starting on other dates contribute zero. "
    "To cap a person's new meetings, query their movable meeting tasks and list all intended task identifiers, checking total and truncated. There is no person field on a constraint. "
    "Only listed tasks count, including fixed tasks if listed; tasks added later must be added to the list with replace_constraint. "
    "Use replace_constraint with an existing id and complete requirement and condition; include label to keep or change it, or omit label to clear it. "
    "Use remove_constraint with constraint_id to delete a constraint. Removing a task prunes task_ids conditions and deletes empty conditions or gaps referencing that task. "
    "Bounds, gaps, and duration maxima are compared without rounding; a hard exact start between slot boundaries is infeasible for a required movable task. "
    "Conditions on fixed tasks use their real intervals without rounding, including boundaries, gaps, daily duration, start date, and overlap with window slots. "
    "Movable task durations must align to the slot grid; condition times need not align. "
    "Fixed task starts and durations may fall between slot boundaries and occupy every slot they touch. "
)

_QUESTION_PROMPT: str = (
    "\nClarifying questions are read by the user: ask in the user's language and never mention internal terms "
    "(stability, hard, soft, strength, importance, required, drop, identifiers). "
    "When you want to ask about one of them, phrase it like these examples:\n"
    "- stability false: Is it all right to rearrange the whole schedule from scratch?\n"
    "- stability true: Should I keep the current placements as they are where possible?\n"
    "- hard or soft: Is this a must, or a preference if possible?\n"
    "- strength: How strong is this preference?\n"
    "- required: Does this have to be scheduled?\n"
    "- importance or drop: If it does not fit, can it be skipped?\n"
    "Do not ask about choices that lead to the same result."
)


class OpenAIStepTranslator(StepTranslator):
    """StepTranslator backed by the OpenAI API."""

    def __init__(
        self,
        client: openai.OpenAI,
        model: str,
        clock: Callable[[], datetime],
    ) -> None:
        self._client: openai.OpenAI = client
        self._model: str = model
        self._clock: Callable[[], datetime] = clock

    def translate(
        self,
        dialogue: Sequence[Utterance],
        summary: Summary,
        steps: Sequence[StepRecord],
    ) -> Step:
        """Choose the next step for the latest utterance."""
        summary_record: dict[str, object] = answer_record(summary)
        records: list[dict[str, object]] = []
        record: StepRecord
        for record in steps:
            result: dict[str, object]
            if isinstance(record.result, Rejected):
                result = {
                    "rejected": [
                        item.message for item in record.result.violations.items
                    ]
                }
            elif isinstance(record, ApplyRecord):
                result = {
                    "executed": [
                        command_record(command, record.result.problem)
                        for command in record.step.commands
                    ]
                }
            else:
                result = {"answered": answer_record(record.result.answer)}
            records.append(
                {
                    "kind": "apply" if isinstance(record, ApplyRecord) else "query",
                    "result": result,
                }
            )
        context: dict[str, object] = {
            "current_time": self._clock().isoformat(),
            "summary": summary_record,
            "steps": records,
        }
        messages: ResponseInputParam = [
            {
                "role": "system",
                "content": _STEP_PROMPT
                + _QUESTION_PROMPT
                + "\nContext: "
                + json.dumps(context, ensure_ascii=False),
            },
        ]
        utterance: Utterance
        for utterance in dialogue:
            message: EasyInputMessageParam = {
                "role": "user" if utterance.speaker is Speaker.USER else "assistant",
                "content": utterance.text,
            }
            messages.append(message)
        response: ParsedResponse[StepOutput] = self._client.responses.parse(
            model=self._model, input=messages, text_format=StepOutput
        )
        output: StepOutput | None = response.output_parsed
        if output is None:
            raise ValueError("Model returned no parsed translation")
        return convert_step_output(output)
