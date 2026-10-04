from datetime import datetime, timedelta

import pytest

from intent_to_schedule.domain.calendar import TimeInterval
from intent_to_schedule.domain.task import FixedTask, TaskId


@pytest.mark.parametrize(
    "duration", [timedelta(0), timedelta(minutes=31, microseconds=1)]
)
def test_fixed_task_interval_uses_start_and_duration(duration: timedelta) -> None:
    """Expose the fixed task's exact interval."""
    start: datetime = datetime(2026, 10, 5, 9)
    task: FixedTask = FixedTask(
        TaskId("fixed"), "Existing", start, duration, frozenset()
    )
    assert task.interval == TimeInterval(start, start + duration)
