import pytest
from einsum_to_eT import get_n_elements, get_comma_separated_dimensions


def test_get_n_elements():
    o2 = get_n_elements("ij")
    assert o2 == "wf%n_o*wf%n_o"
    v4 = get_n_elements("abcd")
    assert v4 == "wf%n_v*wf%n_v*wf%n_v*wf%n_v"
    vovo = get_n_elements("aibj")
    assert vovo == "wf%n_v*wf%n_o*wf%n_v*wf%n_o"
    g2 = get_n_elements("pq")
    assert g2 == "wf%n_mo*wf%n_mo"
    g2 = get_n_elements("pw")
    assert g2 == "wf%n_mo*wf%n_mo"

    with pytest.raises(ValueError):
        get_n_elements("xy")


def test_get_comma_separated_dimensions():
    o2 = get_comma_separated_dimensions("ij")
    assert o2 == "wf%n_o,wf%n_o"
    v4 = get_comma_separated_dimensions("abcd")
    assert v4 == "wf%n_v,wf%n_v,wf%n_v,wf%n_v"
    vovo = get_comma_separated_dimensions("aibj")
    assert vovo == "wf%n_v,wf%n_o,wf%n_v,wf%n_o"
    g2 = get_comma_separated_dimensions("pq")
    assert g2 == "wf%n_mo,wf%n_mo"
    g2 = get_comma_separated_dimensions("pw")
    assert g2 == "wf%n_mo,wf%n_mo"

    with pytest.raises(ValueError):
        get_comma_separated_dimensions("xy")


def test_contains_all_indices():
    from einsum_to_eT import contains_all_indices

    tensor_indices = "aick"
    assert contains_all_indices(tensor_indices, "ck")
    assert contains_all_indices(tensor_indices, "kc")
    assert contains_all_indices(tensor_indices, "ik")
    assert contains_all_indices(tensor_indices, "ki")
    assert contains_all_indices(tensor_indices, "ka")
    assert contains_all_indices(tensor_indices, "ak")
    assert contains_all_indices(tensor_indices, "ai")
    assert contains_all_indices(tensor_indices, "ia")
    assert contains_all_indices(tensor_indices, "ci")
    assert contains_all_indices(tensor_indices, "ic")

    assert contains_all_indices(tensor_indices, "ick")
    assert contains_all_indices(tensor_indices, "iak")

    assert not contains_all_indices(tensor_indices, "bj")
    assert not contains_all_indices(tensor_indices, "pq")
    assert not contains_all_indices(tensor_indices, "bjc")
