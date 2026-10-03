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
    "Use replace_task with start and the same identifier to fix an existing task at that time; its constraints remain. "
    "To make a fixed task movable, use replace_task without start and include importance, required, and stability. "
    "Fixed tasks omit importance, required, and stability; constraints can reference them. "
    "Importance low, medium, or high means how much it matters to do the Task at all. "
    "Required means the Task must be scheduled, including expressions such as 絶対; this does not make its placement preferences hard. "
    "Stability means how strongly to keep the Task at its previous time. "
    'Set solve stability false only for a clear request to rebuild the whole schedule, such as "redo everything" or "start over"; true for partial changes and additions. '
    "If it is unclear whether existing placements should be rebuilt, use message to ask whether rearranging the whole schedule is acceptable. "
    "Use hard for conditions that must hold if the task is scheduled; hard does not require scheduling the task. "
    "For soft preferences choose weak, normal, or strong strength. "
    "For time-of-day or weekday wishes use add_time_constraint with windows, rather than enumerating intervals or computing complements. "
    "Pass a non-empty task_ids list and group Tasks sharing a condition in one add_time_constraint. "
    "A multi-task interval intrusion sums each Task's overlap with the region; a hard condition excludes every Task from it. "
    "within keeps the whole task inside the windows; avoid prohibits or penalizes overlap. "
    "A window combines date_range, weekdays, and time_range; multiple windows are alternatives. "
    "Date ranges exclude the end date; null date_range means the whole horizon, null weekdays means all days, and null time_range means the whole day. "
    "A time_range end of null means the end of the day; split overnight wishes into two windows. "
    "Interpret local window times in the horizon's timezone and relative dates using current_time. "
    "There are no default hours for words like afternoon; ask for missing bounds. "
    "The system expands and rounds windows to slots. Do not substitute within for a condition on the start time alone; ask about the unsupported request. "
    "Other supported pairs: point with distance to an instant; interval with intrusion into a region; "
    "dependency with distance or shortfall of a duration from the end of from_task to the start of to_task; "
    "aggregate with excess per day using count or total_duration. "
    "Movable task durations and explicit constraint times must align to the slot grid; ask instead of guessing. "
    "Fixed task starts and durations may fall between slot boundaries and occupy every slot they touch. "
)

_QUESTION_PROMPT: str = (
    "\nClarifying questions are read by the user: ask in the user's language and never mention internal terms "
    "(stability, hard, soft, strength, importance, required, drop, measure, evaluation, identifiers). "
    "When you want to ask about one of them, phrase it like these examples:\n"
    "- stability false: 今の配置をいったん崩して、全体を組み直してもいいですか？\n"
    "- stability true: 今の配置はなるべく動かさずに調整しますか？\n"
    "- hard or soft: 絶対に守る条件ですか、それともできればの希望ですか？\n"
    "- strength: どのくらい強い希望ですか？\n"
    "- required: 必ず入れる必要がありますか？\n"
    "- importance or drop: 入りきらない場合は見送ってもいいですか？\n"
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
                        command_record(command, summary.grid)
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
