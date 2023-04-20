import pytest
from einsum_to_eT import tensor, allocatable_tensor


def test_tensor():
    t2_aibj = tensor("t2", "aibj")
    t2_ckdl = tensor("t2", "ckdl")
    g = tensor("g", "pqrs")
    F_vo = tensor("Fock", "em")
    rank1 = tensor("rank1", "t")
    t2_rank1 = tensor("t2", "p")

    assert t2_aibj == t2_ckdl
    assert t2_aibj != g
    assert F_vo != g
    assert F_vo != t2_aibj
    assert t2_rank1 != t2_aibj
    assert t2_rank1 != t2_ckdl
    assert t2_rank1 != rank1

    assert t2_aibj.symbol == "t2"
    assert t2_aibj.symbol == "t2"
    assert t2_rank1.symbol == "t2"
    assert g.symbol == "g"
    assert F_vo.symbol == "Fock"
    assert rank1.symbol == "rank1"

    assert t2_aibj.indices == "aibj"
    assert t2_ckdl.indices == "ckdl"
    assert g.indices == "pqrs"
    assert F_vo.indices == "em"
    assert rank1.indices == "t"
    assert t2_rank1.indices == "p"

    assert str(t2_aibj) == "t2_aibj"
    assert str(t2_ckdl) == "t2_ckdl"
    assert str(g) == "g_pqrs"
    assert str(F_vo) == "Fock_em"
    assert str(rank1) == "rank1_t"

    assert t2_ckdl.dimensions == "wf%n_v,wf%n_o,wf%n_v,wf%n_o"
    assert g.dimensions == "wf%n_mo,wf%n_mo,wf%n_mo,wf%n_mo"
    assert F_vo.dimensions == "wf%n_v,wf%n_o"
    assert rank1.dimensions == "wf%n_mo"

    assert (
        t2_ckdl.get_declaration()
        == "      real(dp), dimension(wf%n_v,wf%n_o,wf%n_v,wf%n_o) :: t2\n"
    )

    assert (
        t2_ckdl.get_declaration("intent(in)")
        == "      real(dp), dimension(wf%n_v,wf%n_o,wf%n_v,wf%n_o), intent(in) :: t2\n"
    )

    assert (
        t2_ckdl.get_declaration("intent(in)", "contiguous")
        == "      real(dp), dimension(wf%n_v,wf%n_o,wf%n_v,wf%n_o), intent(in), contiguous :: t2\n"
    )

    assert (
        F_vo.get_declaration() == "      real(dp), dimension(wf%n_v,wf%n_o) :: Fock\n"
    )

    assert rank1.get_declaration() == "      real(dp), dimension(wf%n_mo) :: rank1\n"
    assert t2_rank1.get_declaration() == "      real(dp), dimension(wf%n_mo) :: t2\n"


def test_allocatable_tensor():
    t2_abij = allocatable_tensor("t2", "abij")

    assert t2_abij.symbol == "t2_abij"
    assert t2_abij.indices == "abij"
    assert str(t2_abij) == t2_abij.symbol
    assert not t2_abij.allocated

    assert (
        t2_abij.allocate()
        == "      call mem%alloc(t2_abij, wf%n_v,wf%n_v,wf%n_o,wf%n_o)\n"
    )
    assert t2_abij.allocated

    assert t2_abij.deallocate() == "      call mem%dealloc(t2_abij)\n"
    assert not t2_abij.allocated

    assert (
        t2_abij.get_declaration()
        == "      real(dp), dimension(:,:,:,:), allocatable :: t2_abij\n"
    )

    assert (
        t2_abij.get_declaration("intent(inout)")
        == "      real(dp), dimension(:,:,:,:), allocatable, intent(inout) :: t2_abij\n"
    )
