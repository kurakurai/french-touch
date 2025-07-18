from lighteval.metrics.metrics_sample import ExactMatches
import re
from typing import Callable
import os
from aenum import Enum
import numpy as np
from lighteval.metrics.normalizations import (helm_normalizer)
from lighteval.metrics.metrics import Metrics
from lighteval.metrics.utils.metric_utils import (
    Metric,
    MetricGrouping,
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric
)
from lighteval.utils.utils import as_list


class ExactMatchesThinking(ExactMatches):
    """
    A class to compute exact matches for reasoning tasks.
    This class extends the ExactMatches metric to handle reasoning-specific requirements.
    """

    def __init__(
        self,
        aggregation_function: Callable[[list[float]], float] = max,
        normalize_gold: Callable[[str], str] | None = None,
        normalize_pred: Callable[[str], str] | None = None,
        strip_strings: bool = False,
        type_exact_match: str = "full",
        answer_token: str = os.environ.get("answer_token", ""),  # for reasoning tasks we need to specify the answer token like <answer> or end of thinking token "</thinking>" if no asnwer token is outputted
    ):

        super().__init__(
            aggregation_function=aggregation_function,
            normalize_gold=normalize_gold,
            normalize_pred=normalize_pred,
            strip_strings=strip_strings,
            type_exact_match=type_exact_match,
        )
        self.answer_token = answer_token
    
    
    def compute_one_item(
        self,
        gold: str,
        pred: str,
    ) -> float:

        #extract the answer afte the answer token if it exists
        if self.answer_token:
            if self.answer_token in pred:
                pred = pred.split(self.answer_token, 1)[1]

        return super().compute_one_item(gold, pred)

class MetricsThinking:
    exact_match = SampleLevelMetric(
        metric_name="em",
        sample_level_fn=ExactMatchesThinking(strip_strings=True).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

    quasi_exact_match = SampleLevelMetric(
        metric_name="qem",
        sample_level_fn=ExactMatchesThinking(
            normalize_gold=helm_normalizer,
            normalize_pred=helm_normalizer,
            strip_strings=True,
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

    prefix_exact_match = SampleLevelMetric(
        metric_name="pem",
        sample_level_fn=ExactMatchesThinking(strip_strings=True, type_exact_match="prefix").compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )

    prefix_quasi_exact_match = SampleLevelMetric(
        metric_name="pqem",
        sample_level_fn=ExactMatchesThinking(
            normalize_gold=helm_normalizer,
            normalize_pred=helm_normalizer,
            type_exact_match="prefix",
        ).compute,
        category=MetricCategory.GENERATIVE,
        use_case=MetricUseCase.ACCURACY,
        corpus_level_fn=np.mean,
        higher_is_better=True,
    )




