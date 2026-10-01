from intent_to_schedule.domain.evaluation import Evaluation
from intent_to_schedule.domain.measure import Measure


def is_supported(measure: Measure, evaluation: Evaluation) -> bool:
    """Check whether a measure and evaluation are compatible."""
    ...
