"""Build the conciseness datasets (ES-at-Scale paper Tables 4 & 5) in the
repo's HF save_to_disk layout.

Train pairs are taken verbatim from the paper's own code
(archive/es_fine-tuning_conciseness.py); eval pairs from paper Table 5.
"""
from datasets import Dataset, DatasetDict

TRAIN_PAIRS = [
    ("Solve: 3 + 5 =", "8"),
    ("If all birds can fly and penguins are birds, can penguins fly?", "No"),
]

EVAL_PAIRS = [
    ("What is the capital of France?", "Paris"),
    ("Calculate: 12×7=", "84"),
    ("Is the statement 'All cats are mammals' true or false?", "True"),
    ("What comes next in the sequence: 2,4,6,8, ?", "10"),
    ("Translate 'Hello' to Spanish:", "Hola"),
    ("What is 15% of 200?", "30"),
    ("Name one primary color:", "Red"),
    ("How many days are in a week?", "7"),
]


def to_ds(pairs):
    return Dataset.from_dict({
        "input": [p for p, _ in pairs],
        "target": [t for _, t in pairs],
    })


if __name__ == "__main__":
    DatasetDict({"train": to_ds(TRAIN_PAIRS)}).save_to_disk(
        "datasets/train/conciseness")
    DatasetDict({"conciseness_eval": to_ds(EVAL_PAIRS)}).save_to_disk(
        "datasets/evaluation_suite/conciseness")
    print("wrote datasets/train/conciseness (2) and "
          "datasets/evaluation_suite/conciseness (8)")
