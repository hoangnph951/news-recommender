import numpy as np
import pytest

from newsrec.metrics import auc_score, evaluate, mrr_score, ndcg_score


def test_perfect_ranking():
    labels, scores = np.array([1, 0, 0, 1]), np.array([0.9, 0.1, 0.2, 0.8])
    assert auc_score(labels, scores) == 1.0
    assert mrr_score(labels, scores) == pytest.approx((1 + 1 / 2) / 2)
    assert ndcg_score(labels, scores, 5) == pytest.approx(1.0)


def test_example_from_slide_l8():
    # Slide L8 trang 22: hit ở vị trí 2, 3, 4 trong top-5, |T| = 3 -> nDCG@5 ~ 0.73
    labels = np.array([0, 1, 1, 1, 0])
    scores = np.array([5, 4, 3, 2, 1], dtype=float)
    assert ndcg_score(labels, scores, 5) == pytest.approx(0.73, abs=0.01)


def test_ties_and_undefined():
    assert auc_score(np.array([1, 0]), np.array([0.5, 0.5])) == 0.5
    assert np.isnan(auc_score(np.array([0, 0]), np.array([0.1, 0.2])))
    metrics = evaluate([[1, 0], [0, 0]], [[0.9, 0.1], [0.3, 0.2]])
    assert metrics["AUC"] == 1.0
