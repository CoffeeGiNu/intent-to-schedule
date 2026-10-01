from datetime import datetime, timedelta

from intent_to_schedule.domain.evaluation import (
    Distance,
    Evaluation,
    Excess,
    Intrusion,
    Shortfall,
)
from intent_to_schedule.domain.measure import (
    AggregateMeasure,
    AggregateQuantity,
    DependencyMeasure,
    IntervalMeasure,
    Measure,
    PointMeasure,
)


def is_supported(measure: Measure, evaluation: Evaluation) -> bool:
    """Check whether a measure and evaluation are compatible."""
    match measure, evaluation:
        case PointMeasure(), Distance(target=datetime()):
            return True
        case IntervalMeasure(), Intrusion():
            return True
        case DependencyMeasure(), Distance(target=timedelta()):
            return True
        case DependencyMeasure(), Shortfall(lower=timedelta()):
            return True
        case AggregateMeasure(quantity=AggregateQuantity.COUNT), Excess(
            upper=int() as upper
        ):
            return not isinstance(upper, bool)
        case AggregateMeasure(quantity=AggregateQuantity.TOTAL_DURATION), Excess(
            upper=timedelta()
        ):
            return True
        case _:
            return False
