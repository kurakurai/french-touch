from lighteval.metrics.metrics_sample import (
    PassAtK,
)
from lighteval.utils.language import Language
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    IndicesExtractionConfig,
    LatexExtractionConfig,
    compare_gold_target,
    extract_target_from_pred,
    get_extraction_regexes,
    multilingual_extractive_match_metric,
)
import numpy as np

# Metric for math-hard-fr task
math_pass_fr_at_1_1n = SampleLevelMetric(
    metric_name="math_pass_fr@1:1_samples",
    sample_level_fn=PassAtK(
        k=1,
        n=1,
        strip_strings=True,
        # Extracting mathematical expressions and latex expressions
        normalize_gold=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[
                    ExprExtractionConfig(),
                    LatexExtractionConfig(boxed_match_priority=0),
                ],
                language=Language.FRENCH,
            ),
        ),
        # Extracting mathematical expressions and latex expressions
        normalize_pred=lambda k: extract_target_from_pred(
            k,
            get_extraction_regexes(
                formatted_doc=None,
                target_types=[
                    ExprExtractionConfig(),
                    LatexExtractionConfig(boxed_match_priority=0),
                ],
                language=Language.FRENCH,
            ),
        ),
        # Uses sympy for comparison
        sample_scoring_function=compare_gold_target,
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)

# Metric for GPQA-Diamond-fr task
gpqa_instruct_pass_fr_at_1_1n = SampleLevelMetric(
    metric_name="gpqa_pass_fr@1:1_samples",
    sample_level_fn=PassAtK(
        k=1,
        n=1,
        sample_scoring_function=lambda pred, ref, doc: multilingual_extractive_match_metric(
            language=Language.FRENCH,
            gold_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters")
            ],
            pred_extraction_target=[
                IndicesExtractionConfig(prefix_for_extraction="NativeLetters")
            ],
            precision=6,
        ).sample_level_fn(
            [ref], [pred], doc
        ),
    ).compute,
    category=MetricCategory.GENERATIVE_SAMPLING,
    use_case=MetricUseCase.REASONING,
    corpus_level_fn=np.mean,
    higher_is_better=True,
)
