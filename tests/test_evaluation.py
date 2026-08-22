import numpy as np

from species_classifier.evaluation import classification_metrics


def test_classification_metrics_includes_top5_and_macro_values():
    targets = np.array([0, 0, 1, 1, 2, 2])
    predictions = np.array([0, 1, 1, 1, 2, 0])
    scores = np.array(
        [
            [0.8, 0.1, 0.1],
            [0.3, 0.6, 0.1],
            [0.1, 0.8, 0.1],
            [0.2, 0.7, 0.1],
            [0.1, 0.2, 0.7],
            [0.6, 0.1, 0.3],
        ]
    )

    metrics = classification_metrics(targets, predictions, scores)

    assert metrics["top1_accuracy"] == 4 / 6
    assert metrics["top5_accuracy"] == 1.0
    assert 0 <= metrics["macro_precision"] <= 1
    assert 0 <= metrics["macro_recall"] <= 1
    assert 0 <= metrics["macro_f1"] <= 1

