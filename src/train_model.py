import numpy as np
import evaluate

seqeval = evaluate.load("seqeval")
"""
If this function is used inside a trainer, you'll need to partially fill
the function with functools.partial.
"""


def compute_metrics_for_initial_training(p, label_lst):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    true_predictions = [
        [label_lst[p] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [label_lst[l] for (p, l) in zip(prediction, label) if l != -100]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(predictions=true_predictions, references=true_labels)
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }


def compute_metrics_without_O(p, label_lst):
    predictions, labels = p
    predictions = np.argmax(predictions, axis=2)

    # Determine the index of the "O" label
    O_index = label_lst.index("O")

    # Filter out predictions and labels with -100 and "O" labels
    true_predictions = [
        [
            label_lst[p]
            for (p, l) in zip(prediction, label)
            if l != -100 and p != O_index and l != O_index
        ]
        for prediction, label in zip(predictions, labels)
    ]
    true_labels = [
        [
            label_lst[l]
            for (p, l) in zip(prediction, label)
            if l != -100 and p != O_index and l != O_index
        ]
        for prediction, label in zip(predictions, labels)
    ]

    results = seqeval.compute(
        predictions=true_predictions, references=true_labels, zero_division=0
    )
    return {
        "precision": results["overall_precision"],
        "recall": results["overall_recall"],
        "f1": results["overall_f1"],
        "accuracy": results["overall_accuracy"],
    }
