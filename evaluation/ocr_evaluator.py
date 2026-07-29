import json

# How to use:
# evaluate("../database/labels/rg_real.json",
#               "../database/predictions/rg_prediction.json")

def normalize(value):
    """Normaliza valores para comparação."""
    if value is None:
        return None

    if isinstance(value, str):
        return value.strip().lower()

    return value


def compare_dict(label, prediction):
    """Compara dois dicionários (label e prediction) e devolve um dicionário
    com os scores de cada campo. Se o campo for um dicionário, a função
    é chamada recursivamente."""
    scores = {}

    for key, value in label.items():

        if isinstance(value, dict):

            pred_child = prediction.get(key, {})
            if not isinstance(pred_child, dict):
                pred_child = {}

            scores[key] = compare_dict(value, pred_child)

        else:

            if key not in prediction:
                scores[key] = 0
                continue

            if normalize(value) == normalize(prediction[key]):
                scores[key] = 100
            else:
                scores[key] = 0

    return scores


def flatten_scores(scores):
    """Extrai todos os scores da árvore."""
    values = []

    for value in scores.values():
        if isinstance(value, dict):
            values.extend(flatten_scores(value))
        else:
            values.append(value)

    return values


def evaluate(label_json: str, prediction_json: str):
    """
    Avalia a acurácia de um arquivo de predição em relação a um
    arquivo de rótulo. Recebe o caminho do arquivo de rótulo
    (label_json) e o caminho do arquivo de predição (prediction_json).
    Retorna um dicionário com a pontuação global e os scores de
    Tomcada campo.
    """
    with open(label_json, encoding="utf-8") as f:
        label = json.load(f)

    with open(prediction_json, encoding="utf-8") as f:
        prediction = json.load(f)

    score_tree = compare_dict(label, prediction)

    values = flatten_scores(score_tree)

    global_score = sum(values) / len(values) if values else 100

    return {
        "score": round(global_score, 2),
        "fields": score_tree,
    }

