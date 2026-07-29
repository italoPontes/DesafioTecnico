import json
import tempfile

from evaluation.ocr_evaluator import evaluate

def run_test(name, label, prediction, expected_score, expected_fields):
    """Executa um caso de teste."""

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(label, f, ensure_ascii=False)
        label_path = f.name

    with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False, encoding="utf-8") as f:
        json.dump(prediction, f, ensure_ascii=False)
        prediction_path = f.name

    result = evaluate(label_path, prediction_path)

    passed = (
        result["score"] == expected_score
        and result["fields"] == expected_fields
    )

    print(f"\n{name}")
    print("=" * len(name))
    print(f"Expected score : {expected_score}")
    print(f"Obtained score : {result['score']}")
    print(f"Status         : {'PASS' if passed else 'FAIL'}")

    if not passed:
        print("\nExpected fields:")
        print(json.dumps(expected_fields, indent=4, ensure_ascii=False))

        print("\nObtained fields:")
        print(json.dumps(result["fields"], indent=4, ensure_ascii=False))

    return passed


def main():

    tests = [

        {
            "name": "Exact match",
            "label": {"document": {"country": "BR", "state": "SP"}},
            "prediction": {"document": {"country": "BR", "state": "SP"}},
            "score": 100,
            "fields": {
                "document": {
                    "country": 100,
                    "state": 100,
                }
            },
        },

        {
            "name": "Case insensitive",
            "label": {"document": {"country": "BR", "state": "SP"}},
            "prediction": {"document": {"country": "br", "state": "sp"}},
            "score": 100,
            "fields": {
                "document": {
                    "country": 100,
                    "state": 100,
                }
            },
        },

        {
            "name": "Extra field",
            "label": {"document": {"country": "BR", "state": "SP"}},
            "prediction": {
                "document": {
                    "country": "BR",
                    "state": "SP",
                    "age": 10,
                }
            },
            "score": 100,
            "fields": {
                "document": {
                    "country": 100,
                    "state": 100,
                }
            },
        },

        {
            "name": "Wrong value",
            "label": {"document": {"country": "BR", "state": "SP"}},
            "prediction": {"document": {"country": "Brasil", "state": "SP"}},
            "score": 50,
            "fields": {
                "document": {
                    "country": 0,
                    "state": 100,
                }
            },
        },

        {
            "name": "Missing field",
            "label": {"document": {"country": "BR", "state": "SP"}},
            "prediction": {"document": {"state": "SP"}},
            "score": 50,
            "fields": {
                "document": {
                    "country": 0,
                    "state": 100,
                }
            },
        },

        {
            "name": "All wrong",
            "label": {"document": {"country": "BR", "state": "SP"}},
            "prediction": {"document": {"country": "US", "state": "RJ"}},
            "score": 0,
            "fields": {
                "document": {
                    "country": 0,
                    "state": 0,
                }
            },
        },

        {
            "name": "Nested object",
            "label": {
                "document": {
                    "issuer": {
                        "country": "BR",
                        "state": "SP",
                    }
                }
            },
            "prediction": {
                "document": {
                    "issuer": {
                        "country": "BR",
                        "state": "RJ",
                    }
                }
            },
            "score": 50,
            "fields": {
                "document": {
                    "issuer": {
                        "country": 100,
                        "state": 0,
                    }
                }
            },
        },

        {
            "name": "Null value",
            "label": {"document": {"country": None}},
            "prediction": {"document": {"country": None}},
            "score": 100,
            "fields": {
                "document": {
                    "country": 100,
                }
            },
        },

    ]

    total = len(tests)
    passed = 0

    print("=" * 60)
    print("Running evaluation tests")
    print("=" * 60)

    for test in tests:
        if run_test(
            test["name"],
            test["label"],
            test["prediction"],
            test["score"],
            test["fields"],
        ):
            passed += 1

    print("\n" + "=" * 60)
    print(f"Passed {passed}/{total} tests")

    if passed == total:
        print("All tests passed!")
    else:
        print(f"{total - passed} test(s) failed.")


if __name__ == "__main__":
    main()