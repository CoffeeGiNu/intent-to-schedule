import json
from collections.abc import Callable, Sequence
from datetime import datetime
from typing import ClassVar, Literal

import openai
from pydantic import ConfigDict
from openai.types.responses import (
    EasyInputMessageParam,
    ParsedResponse,
    ResponseInputParam,
)

from intent_to_schedule.adapter.data_model import (
    APPLY_OPERATION_DESCRIPTION,
    CommandData,
    DataModel,
    QUERY_OPERATION_DESCRIPTION,
    QueryData,
    SCHEDULE_OPERATION_DESCRIPTION,
    ScheduleData,
    answer_result_record,
    answer_record,
    convert_command,
    convert_query,
    execute_result_record,
)
from intent_to_schedule.application.query import Summary
from intent_to_schedule.application.translate import (
    ApplyRecord,
    ApplyStep,
    MessageStep,
    QueryStep,
    ScheduleStep,
    Speaker,
    Step,
    StepRecord,
    StepTranslator,
    Utterance,
)


class QueryOutput(DataModel):
    """Structured output for a query step."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={"description": QUERY_OPERATION_DESCRIPTION}
    )

    kind: Literal["query"]
    query: QueryData


class ApplyOutput(DataModel):
    """Structured output for an apply step."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={"description": APPLY_OPERATION_DESCRIPTION}
    )

    kind: Literal["apply"]
    commands: tuple[CommandData, ...]


class ScheduleOutput(ScheduleData):
    """Structured output for a schedule step."""

    model_config: ClassVar[ConfigDict] = ConfigDict(
        json_schema_extra={"description": SCHEDULE_OPERATION_DESCRIPTION}
    )

    kind: Literal["schedule"]


class MessageOutput(DataModel):
    """Structured output for a message step."""

    kind: Literal["message"]
    text: str


class StepOutput(DataModel):
    """Root of the structured output for a translation step."""

    result: QueryOutput | ApplyOutput | ScheduleOutput | MessageOutput


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
        case ScheduleOutput(stability=stability):
            return ScheduleStep(stability)
        case MessageOutput(text=text):
            return MessageStep(text)


_STEP_PROMPT: str = (
    "Choose one next step for the latest user utterance: query, apply, schedule, or message. "
    "The chat turn has a limit of 12 steps. "
    "Context contains the current time, a summary of the stored problem, and results of earlier steps in this turn. "
    "After applying all requested changes, use schedule to create the schedule and end the turn. "
    "Use message to answer a question or ask for clarification and end the turn. "
    "Write every field explicitly. "
    "When query results remain ambiguous, narrow them or use message to ask the user rather than guessing. "
    "Ask the user when required time bounds are missing. "
    "If it is unclear whether existing placements should be rebuilt, use message to ask whether rearranging the whole schedule is acceptable. "
    "A rejected step includes an explanation; use it to correct the next step or ask the user. "
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
            if isinstance(record, ApplyRecord):
                result = execute_result_record(record.step.commands, record.result)
            else:
                result = answer_result_record(record.result)
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
