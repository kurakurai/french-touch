from lighteval.metrics.metrics_sample import ExactMatches



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
        answer_token: str = "",  # for reasoning tasks we need to specify the answer token like <answer> or end of thinking token "</thinking>" if no asnwer token is outputted
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
        if not self.answer_token:
            return super().compute_one_item(gold, pred)
        
