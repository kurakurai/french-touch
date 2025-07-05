import random
from lighteval.tasks.default_prompts import LETTER_INDICES
from lighteval.tasks.requests import Doc


# Ifeval-fr prompt function
def prompt_ifeval_fr(line, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=line["prompt"],
        choices=[""],
        gold_index=0,
        instruction="",
        specific={
            "instructions_id_list": line["instruction_id_list"],
            "kwargs": line["kwargs"],
        },
    )


# gpqa-fr prompt function
def prompt_gpqa_fr(line, task_name: str = None):
    gold_index = random.randint(0, 3)
    choices = [
        line["Réponse incorrecte 1"],
        line["Réponse incorrecte 2"],
        line["Réponse incorrecte 3"],
    ]
    choices.insert(gold_index, line["Réponse correcte"])

    instruction = "Choisissez la réponse correcte aux questions suivantes.\n\n"

    query = f"Question: {line['Question']}\n"
    query += "".join(
        [f"{key}. {choice}\n" for key, choice in zip(LETTER_INDICES, choices)]
    )
    query += "Réponse: "
    return Doc(
        task_name=task_name,
        query=f"{instruction}{query}",
        choices=LETTER_INDICES[: len(choices)],
        gold_index=gold_index,
        instruction=instruction,
    )


# bbh-fr prompt functions
def bbh(line, instruction, choices, task_name: str = None):
    return Doc(
        task_name=task_name,
        query=f"{instruction}Q: {line['input']}\nA:",
        choices=choices,
        gold_index=choices.index(line["target"]),
        instruction=instruction,
    )


def bbh_boolean_expressions(line, task_name: str = None):
    instruction = "Évalue le résultat d'une expression booléenne aléatoire.\n\n"
    choices = ["Incorrect", "Vrai"]
    return bbh(line, instruction, choices, task_name)


def bbh_causal_judgment(line, task_name: str = None):
    instruction = "Réponds à des questions sur l’attribution causale.\n\n"
    choices = ["Oui", "Non"]
    return bbh(line, instruction, choices, task_name)


def bbh_date_understanding(line, task_name: str = None):
    instruction = "Infère la date à partir du contexte.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:6]]
    return bbh(line, instruction, choices, task_name)


def bbh_disambiguation_qa(line, task_name: str = None):
    instruction = "Clarifie le sens des phrases avec des pronoms ambigus.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:3]]
    return bbh(line, instruction, choices, task_name)


def bbh_dyck_languages(line, task_name: str = None):  # Can only be done in generative
    instruction = "Ferme correctement un mot Dyck-n.\n\n"
    choices = [line["target"]]
    return bbh(line, instruction, choices, task_name)


def bbh_formal_fallacies(line, task_name: str = None):
    instruction = (
        "Distingue les arguments déductivement valides des sophismes formels.\n\n"
    )
    choices = ["valide", "invalidee"]
    return bbh(line, instruction, choices, task_name)


def bbh_geometric_shapes(line, task_name: str = None):
    instruction = "Nomme les formes géométriques à partir de leurs chemins SVG.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:11]]
    return bbh(line, instruction, choices, task_name)


def bbh_hyperbaton(line, task_name: str = None):
    instruction = "Ordonne correctement les adjectifs dans des phrases anglaises.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:2]]
    return bbh(line, instruction, choices, task_name)


def bbh_logical_deduction_five_objects(line, task_name: str = None):
    instruction = "Une tâche de déduction logique qui nécessite de déduire l’ordre d’une séquence d’objets.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:5]]
    return bbh(line, instruction, choices, task_name)


def bbh_logical_deduction_seven_objects(line, task_name: str = None):
    instruction = "Une tâche de déduction logique qui nécessite de déduire l’ordre d’une séquence d’objets.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:7]]
    return bbh(line, instruction, choices, task_name)


def bbh_logical_deduction_three_objects(line, task_name: str = None):
    instruction = "Une tâche de déduction logique qui nécessite de déduire l’ordre d’une séquence d’objets.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:3]]
    return bbh(line, instruction, choices, task_name)


def bbh_movie_recommendation(line, task_name: str = None):
    if line["target"] == "Monsters, Inc":  # this line is not correctly formatted
        print(
            "One sample removed from task bbh:movie_recommendation because its line is incorrectly formatted."
        )
        return []
    instruction = "Recommande des films similaires à la liste de films donnée.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:6]]
    return bbh(line, instruction, choices, task_name)


def bbh_multistep_arithmetic_two(line, task_name: str = None):
    instruction = "Résous des problèmes arithmétiques à étapes multiples.\n\n"  # Can only be done in generative
    choices = [line["target"]]
    return bbh(line, instruction, choices, task_name)


def bbh_navigate(line, task_name: str = None):
    instruction = "Étant donné une série d’instructions de navigation, détermine si l’on revient au point de départ.\n\n"
    choices = ["Oui", "Non"]
    return bbh(line, instruction, choices, task_name)


def bbh_object_counting(line, task_name: str = None):
    instruction = "Questions impliquant de dénombrer des objets et demander au modèle de les compter.\n\n"
    choices = [str(i) for i in range(1, 19)]
    return bbh(line, instruction, choices, task_name)


def bbh_penguins_in_a_table(line, task_name: str = None):
    instruction = (
        "Réponds à des questions sur un tableau de pingouins et leurs attributs.\n\n"
    )
    choices = [f"({c})" for c in LETTER_INDICES[:5]]
    return bbh(line, instruction, choices, task_name)


def bbh_reasoning_about_colored_objects(line, task_name: str = None):
    instruction = "Réponds à des questions très simples sur les couleurs des objets sur une surface.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:18]]
    return bbh(line, instruction, choices, task_name)


def bbh_ruin_names(line, task_name: str = None):
    if line["target"] in [
        "dearth, wind, & fire",
        "rita, sue and bob poo",
    ]:  # line not correctly formatted
        print(
            "One sample removed from task bbh:ruin_names because its line is incorrectly formatted."
        )
        return []
    instruction = "Choisis la modification humoristique qui « ruine » le nom du film ou de l’artiste musical donné.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:6]]
    return bbh(line, instruction, choices, task_name)


def bbh_salient_translation_error_detection(line, task_name: str = None):
    instruction = "Détecte le type d’erreur dans une traduction anglaise d’une phrase source allemande.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:6]]
    return bbh(line, instruction, choices, task_name)


def bbh_snarks(line, task_name: str = None):
    instruction = "Détermine laquelle des deux phrases est sarcastique. Selon le dictionnaire de l’université de Cambridge, le sarcasme est « l’utilisation de remarques qui signifient clairement le contraire de ce qu’elles disent, faites pour blesser quelqu’un ou critiquer quelque chose de manière humoristique ». Les phrases sarcastiques contiennent souvent des propos satiriques ou ironiques, des hyperboles, des remarques ambivalentes ou spirituelles.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:2]]
    return bbh(line, instruction, choices, task_name)


def bbh_sports_understanding(line, task_name: str = None):
    instruction = "Détermine si une phrase artificiellement construite liée au sport est plausible ou non.\n\n"
    choices = ["Oui", "Non"]
    return bbh(line, instruction, choices, task_name)


def bbh_temporal_sequences(line, task_name: str = None):
    instruction = "Description de tâche: Réponds à des questions sur les moments auxquels certains événements ont pu se produire.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:4]]
    return bbh(line, instruction, choices, task_name)


def bbh_tracking_shuffled_objects_five_objects(line, task_name: str = None):
    instruction = "Détermine les positions finales d’un ensemble d’objets données leurs positions initiales et la description d’une séquence d’échanges.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:5]]
    return bbh(line, instruction, choices, task_name)


def bbh_tracking_shuffled_objects_seven_objects(line, task_name: str = None):
    instruction = "Détermine les positions finales d’un ensemble d’objets données leurs positions initiales et la description d’une séquence d’échanges.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:7]]
    return bbh(line, instruction, choices, task_name)


def bbh_tracking_shuffled_objects_three_objects(line, task_name: str = None):
    instruction = "Détermine les positions finales d’un ensemble d’objets données leurs positions initiales et la description d’une séquence d’échanges.\n\n"
    choices = [f"({c})" for c in LETTER_INDICES[:3]]
    return bbh(line, instruction, choices, task_name)


def bbh_web_of_lies(line, task_name: str = None):
    instruction = "Évalue une fonction booléenne aléatoire exprimée sous forme de problème en mots.\n\n"
    choices = ["Oui", "Non"]
    return bbh(line, instruction, choices, task_name)


def bbh_word_sorting(line, task_name: str = None):
    instruction = "Trie une liste de mots.\n\n"  # Can only be done in generative
    choices = [line["target"]]
    return bbh(line, instruction, choices, task_name)
