from lighteval.metrics.metrics_sample import ExactMatches
import re
from typing import Callable
import os

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
        print("working")
        #extract the answer afte the answer token if it exists
        if self.answer_token:
            if self.answer_token in pred:
                pred = pred.split(self.answer_token, 1)[1]

        return super().compute_one_item(gold, pred)