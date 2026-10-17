import pytest

from raglaw.retrieval.fusion import reciprocal_rank_fusion

pytestmark = pytest.mark.unit


# U27: fixed ranked lists, scores worked out by hand with k = 60.
def test_rrf_sums_reciprocal_ranks_across_lists():
    fused = reciprocal_rank_fusion([["a", "b", "c"], ["b", "c", "d"]], k=60)

    assert [doc_id for doc_id, _ in fused] == ["b", "c", "a", "d"]
    assert dict(fused)["b"] == pytest.approx(1 / 62 + 1 / 61)
    assert dict(fused)["d"] == pytest.approx(1 / 63)


def test_k_dampens_the_weight_of_the_top_rank():
    """With a small k, one first place beats two middling places; with k = 60 it doesn't."""
    lists = [["a", "x", "y", "b"], ["c", "z", "w", "b"], ["d", "v", "u", "b"]]

    assert reciprocal_rank_fusion(lists, k=0)[0][0] == "a"
    assert reciprocal_rank_fusion(lists, k=60)[0][0] == "b"


def test_ties_are_broken_by_id_whatever_the_list_order():
    assert reciprocal_rank_fusion([["b"], ["a"]]) == reciprocal_rank_fusion(
        [["a"], ["b"]]
    )
    assert [doc_id for doc_id, _ in reciprocal_rank_fusion([["b"], ["a"]])] == [
        "a",
        "b",
    ]


def test_a_repeated_id_counts_only_at_its_best_rank_in_one_list():
    fused = dict(reciprocal_rank_fusion([["a", "a", "b"]]))

    assert fused["a"] == pytest.approx(1 / 61)
    assert fused["b"] == pytest.approx(1 / 63)  # the others keep their own rank


def test_an_empty_list_contributes_nothing():
    assert reciprocal_rank_fusion([[], ["a"]]) == [("a", pytest.approx(1 / 61))]


def test_a_negative_k_is_rejected():
    with pytest.raises(ValueError, match="k"):
        reciprocal_rank_fusion([["a"]], k=-1)
