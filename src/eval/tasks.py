# MIT License

# Copyright (c) 2024 The HuggingFace Team

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

# ruff: noqa: F405, F403, F401
"""
Custom evaluation tasks for lighteval.

This file generally creates just a TASKS_TABLE and TASKS_GROUPS which are then imported by LightEval.

This module implements tasks for the french specific datasets
See : https://huggingface.co/fr-gouv-coordination-ia
"""

from lighteval.metrics.metrics import Metrics
from lighteval.tasks.extended.ifeval.main import ifeval_metrics
from lighteval.tasks.lighteval_task import LightevalTaskConfig
import prompts as prompt
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
)
from lighteval.metrics.metrics_sample import (
    PassAtK,
)
from lighteval.utils.language import Language
from lighteval.metrics.utils.metric_utils import (
    SampleLevelMetric,
)
from lighteval.metrics.dynamic_metrics import (
    ExprExtractionConfig,
    LatexExtractionConfig,
    compare_gold_target,
    extract_target_from_pred,
    get_extraction_regexes,
)
from lighteval.metrics.utils.metric_utils import (
    MetricCategory,
    MetricUseCase,
    SampleLevelMetric,
)
import numpy as np


# IFEVal-fr task
ifeval_fr_task = LightevalTaskConfig(
    name="ifeval-fr",
    prompt_function=prompt.prompt_ifeval_fr,
    suite=["community"],
    hf_repo="jzhang86/fr_ifeval",
    hf_subset="default",
    metric=[ifeval_metrics],
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1280,
    stop_sequence=[],  # no stop sequence, will use eot token
    version="0.1",
)

# GPQA-fr task
gpqa_fr_task = LightevalTaskConfig(
    name="gpqa-fr",
    suite=["community"],
    prompt_function=prompt.prompt_gpqa_fr,
    hf_repo="le-leadboard/gpqa-fr",
    hf_subset="default",
    hf_avail_splits=["train"],
    evaluation_splits=["train"],
    few_shots_split="train",
    few_shots_select="random_sampling",
    generation_size=1,
    metric=[Metrics.loglikelihood_acc],
    stop_sequence=["\n"],
    trust_dataset=True,
    version=0,
)


# Math-Hard-fr task
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

math_hard_fr_task = LightevalTaskConfig(
    name="math-hard-fr",
    suite=["community"],
    prompt_function=prompt.prompt_math_hard_fr,
    hf_repo="le-leadboard/MATH_LVL5_fr",
    hf_subset="default",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=4096,
    metric=[
        math_pass_fr_at_1_1n,
    ],
    trust_dataset=True,
    version=0,
)

# BBH-fr task
bbh_boolean_expressions_community = LightevalTaskConfig(
    name="bbh-fr:expressions_booléennes",
    suite=["community"],
    prompt_function=prompt.bbh_boolean_expressions,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="expressions_booléennes",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_causal_judgment_community = LightevalTaskConfig(
    name="bbh-fr:jugement_causal",
    suite=["community"],
    prompt_function=prompt.bbh_causal_judgment,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="jugement_causal",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_date_understanding_community = LightevalTaskConfig(
    name="bbh-fr:compréhension_de_la_date",
    suite=["community"],
    prompt_function=prompt.bbh_date_understanding,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="compréhension_de_la_date",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_disambiguation_qa_community = LightevalTaskConfig(
    name="bbh-fr:désambiguïsation_qa",
    suite=["community"],
    prompt_function=prompt.bbh_disambiguation_qa,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="désambiguïsation_qa",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_dyck_languages_community = LightevalTaskConfig(
    name="bbh-fr:dyck_languages",
    suite=["community"],
    prompt_function=prompt.bbh_dyck_languages,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="dyck_languages",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_formal_fallacies_community = LightevalTaskConfig(
    name="bbh-fr:sophismes_formels",
    suite=["community"],
    prompt_function=prompt.bbh_formal_fallacies,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="sophismes_formels",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_geometric_shapes_community = LightevalTaskConfig(
    name="bbh-fr:formes_géométriques",
    suite=["community"],
    prompt_function=prompt.bbh_geometric_shapes,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="formes_géométriques",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_hyperbaton_community = LightevalTaskConfig(
    name="bbh-fr:hyperbate",
    suite=["community"],
    prompt_function=prompt.bbh_hyperbaton,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="hyperbate",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_logical_deduction_five_objects_community = LightevalTaskConfig(
    name="bbh-fr:suivi_objets_mélangés_cinq_objets",
    suite=["community"],
    prompt_function=prompt.bbh_logical_deduction_five_objects,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="suivi_objets_mélangés_cinq_objets",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_logical_deduction_seven_objects_community = LightevalTaskConfig(
    name="bbh-fr:déduction_logique_sept_objets",
    suite=["community"],
    prompt_function=prompt.bbh_logical_deduction_seven_objects,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="déduction_logique_sept_objets",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_logical_deduction_three_objects_community = LightevalTaskConfig(
    name="bbh-fr:déduction_logique_trois_objets",
    suite=["community"],
    prompt_function=prompt.bbh_logical_deduction_three_objects,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="déduction_logique_trois_objets",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_movie_recommendation_community = LightevalTaskConfig(
    name="bbh-fr:recommandation_de_film",
    suite=["community"],
    prompt_function=prompt.bbh_movie_recommendation,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="recommandation_de_film",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_multistep_arithmetic_two_community = LightevalTaskConfig(
    name="bbh-fr:multistep_arithmetic_two",
    suite=["community"],
    prompt_function=prompt.bbh_multistep_arithmetic_two,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="multistep_arithmetic_two",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_navigate_community = LightevalTaskConfig(
    name="bbh-fr:naviguer",
    suite=["community"],
    prompt_function=prompt.bbh_navigate,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="naviguer",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_object_counting_community = LightevalTaskConfig(
    name="bbh-fr:comptage_d_objets",
    suite=["community"],
    prompt_function=prompt.bbh_object_counting,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="comptage_d_objets",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_penguins_in_a_table_community = LightevalTaskConfig(
    name="bbh-fr:pingouins_sur_une_table",
    suite=["community"],
    prompt_function=prompt.bbh_penguins_in_a_table,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="pingouins_sur_une_table",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_reasoning_about_colored_objects_community = LightevalTaskConfig(
    name="bbh-fr:raisonnement_sur_les_objets_colorés",
    suite=["community"],
    prompt_function=prompt.bbh_reasoning_about_colored_objects,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="raisonnement_sur_les_objets_colorés",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_ruin_names_community = LightevalTaskConfig(
    name="bbh-fr:noms_de_ruines",
    suite=["community"],
    prompt_function=prompt.bbh_ruin_names,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="noms_de_ruines",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_salient_translation_error_detection_community = LightevalTaskConfig(
    name="bbh-fr:détection_d_erreur_de_traduction_sailante",
    suite=["community"],
    prompt_function=prompt.bbh_salient_translation_error_detection,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="détection_d_erreur_de_traduction_sailante",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_snarks_community = LightevalTaskConfig(
    name="bbh-fr:sarcasmes",
    suite=["community"],
    prompt_function=prompt.bbh_snarks,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="sarcasmes",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_sports_understanding_community = LightevalTaskConfig(
    name="bbh-fr:compréhension_des_sports",
    suite=["community"],
    prompt_function=prompt.bbh_sports_understanding,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="compréhension_des_sports",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_temporal_sequences_community = LightevalTaskConfig(
    name="bbh-fr:séquences_temporelles",
    suite=["community"],
    prompt_function=prompt.bbh_temporal_sequences,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="séquences_temporelles",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_tracking_shuffled_objects_five_objects_community = LightevalTaskConfig(
    name="bbh-fr:suivi_objets_mélangés_cinq_objets",
    suite=["community"],
    prompt_function=prompt.bbh_tracking_shuffled_objects_five_objects,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="suivi_objets_mélangés_cinq_objets",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_tracking_shuffled_objects_seven_objects_community = LightevalTaskConfig(
    name="bbh-fr:suivi_objets_mélangés_sept_objets",
    suite=["community"],
    prompt_function=prompt.bbh_tracking_shuffled_objects_seven_objects,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="suivi_objets_mélangés_sept_objets",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_tracking_shuffled_objects_three_objects_community = LightevalTaskConfig(
    name="bbh-fr:suivi_objets_mélangés_trois_objets",
    suite=["community"],
    prompt_function=prompt.bbh_tracking_shuffled_objects_three_objects,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="suivi_objets_mélangés_trois_objets",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_web_of_lies_community = LightevalTaskConfig(
    name="bbh-fr:toile_de_mensonges",
    suite=["community"],
    prompt_function=prompt.bbh_web_of_lies,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="toile_de_mensonges",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)
bbh_word_sorting_community = LightevalTaskConfig(
    name="bbh-fr:tri_de_mots",
    suite=["community"],
    prompt_function=prompt.bbh_word_sorting,
    hf_repo="le-leadboard/bbh-fr",
    hf_subset="tri_de_mots",
    hf_avail_splits=["test"],
    evaluation_splits=["test"],
    few_shots_split="train",
    few_shots_select="random",
    generation_size=20,
    metric=[
        Metrics.exact_match,
        Metrics.quasi_exact_match,
        Metrics.prefix_exact_match,
        Metrics.prefix_quasi_exact_match,
        Metrics.perfect_exact_match,
    ],
    stop_sequence=["</s>", "Q=", "\n\n"],
    trust_dataset=True,
    version=0,
)

# STORE YOUR EVALS
TASKS_TABLE = [
    ifeval_fr_task,
    gpqa_fr_task,
    math_hard_fr_task,
    bbh_boolean_expressions_community,
    bbh_causal_judgment_community,
    bbh_date_understanding_community,
    bbh_disambiguation_qa_community,
    bbh_dyck_languages_community,
    bbh_formal_fallacies_community,
    bbh_geometric_shapes_community,
    bbh_hyperbaton_community,
    bbh_logical_deduction_five_objects_community,
    bbh_logical_deduction_seven_objects_community,
    bbh_logical_deduction_three_objects_community,
    bbh_movie_recommendation_community,
    bbh_multistep_arithmetic_two_community,
    bbh_navigate_community,
    bbh_object_counting_community,
    bbh_penguins_in_a_table_community,
    bbh_reasoning_about_colored_objects_community,
    bbh_ruin_names_community,
    bbh_salient_translation_error_detection_community,
    bbh_snarks_community,
    bbh_sports_understanding_community,
    bbh_temporal_sequences_community,
    bbh_tracking_shuffled_objects_five_objects_community,
    bbh_tracking_shuffled_objects_seven_objects_community,
    bbh_tracking_shuffled_objects_three_objects_community,
    bbh_web_of_lies_community,
    bbh_word_sorting_community,
]
