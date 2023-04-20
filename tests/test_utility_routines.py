import pytest
from einsum_to_eT import tensor, allocatable_tensor


def test_get_routine_header():
    from einsum_to_eT import get_routine_header

    t2_cidj = tensor("t2", "cidj")
    g_vvvv = tensor("g_vvvv", "acbd")
    omega2 = tensor("omega2", "aibj")

    name = "construct_ccsd_omega_a2"
    prefactor = 1.0

    header = get_routine_header(name, prefactor, [t2_cidj, g_vvvv, omega2])

    reference = """   subroutine construct_ccsd_omega_a2(wf, t2, g_vvvv, omega2)
!!
!!    Calculates:
!!    omega2_aibj += 1.00 t2_cidj g_vvvv_acbd
!!
      implicit none
!
      class(ccs), intent(in) :: wf
!
      real(dp), dimension(wf%n_v,wf%n_o,wf%n_v,wf%n_o), intent(in) :: t2
      real(dp), dimension(wf%n_v,wf%n_v,wf%n_v,wf%n_v), intent(in) :: g_vvvv
      real(dp), dimension(wf%n_v,wf%n_o,wf%n_v,wf%n_o), intent(inout) :: omega2
!
"""
    assert header == reference


def test_parse_einsum_contraction_string():
    from einsum_to_eT import parse_einsum_contraction_string

    reference = ["cidj", "acbd", "aibj"]
    tensor_indices = parse_einsum_contraction_string("cidj,acbd->aibj")
    assert reference == tensor_indices

    with pytest.raises(ValueError):
        parse_einsum_contraction_string("ci-dj,ac.bd->aibj")


def test_get_reordering():
    from einsum_to_eT import get_reordering

    t2 = tensor("t2", "aibj")
    t2_abij = allocatable_tensor("t2", "abij")
    t2_kcdl = allocatable_tensor("t2", "kcdl")

    sort_aibj_to_abij = get_reordering(t2, t2_abij, (0, 2, 1, 3))
    reference = "sort_to_1324(t2, t2_abij, wf%n_v,wf%n_o,wf%n_v,wf%n_o)\n"
    reference = "      call " + reference
    assert sort_aibj_to_abij == reference

    sort_ckdl_to_kcdl = get_reordering(t2, t2_kcdl, (1, 0, 2, 3))
    reference = "sort_to_2134(t2, t2_kcdl, wf%n_v,wf%n_o,wf%n_v,wf%n_o)\n"
    reference = "      call " + reference
    assert sort_ckdl_to_kcdl == reference

    sort_cdkl_to_kcdl = get_reordering(t2_abij, t2_kcdl, (2, 0, 1, 3))
    reference = "sort_to_3124(t2_abij, t2_kcdl, wf%n_v,wf%n_v,wf%n_o,wf%n_o)\n"
    reference = "      call " + reference
    assert sort_cdkl_to_kcdl == reference


def test_add_array_to_result_daxpy():
    from einsum_to_eT import add_array_to_result

    g_aibj = tensor("g_vovo", "aibj")
    g_aibj_allocatable = allocatable_tensor("g_vovo", "aibj")
    omega2 = tensor("omega2", "aibj")

    daxpy = add_array_to_result(1.0, g_aibj, omega2)

    reference = "      call daxpy(wf%n_v*wf%n_o*wf%n_v*wf%n_o, 1.0d0, g_vovo, 1, omega2, 1)\n!\n"
    assert daxpy == reference

    reference = "      call daxpy(wf%n_v*wf%n_o*wf%n_v*wf%n_o, 1.0d0, g_vovo_aibj, 1, omega2, 1)\n"
    reference += "      call mem%dealloc(g_vovo_aibj)\n!\n"
    daxpy_dealloc = add_array_to_result(1.0, g_aibj_allocatable, omega2)
    assert daxpy_dealloc == reference


def test_add_array_to_result_reorder_and_add():
    from einsum_to_eT import add_array_to_result

    g_vvoo = tensor("g_vvoo", "abij")
    g_ovov = tensor("g_ovov", "iajb")
    g_ovov_allocatable = allocatable_tensor("g_ovov", "jaib")
    omega2 = tensor("omega2", "aibj")

    add_1324 = add_array_to_result(1.0, g_vvoo, omega2)
    reference = "      call add_1324_to_1234(1.0d0, g_vvoo, omega2, wf%n_v,wf%n_o,wf%n_v,wf%n_o)\n!\n"
    assert add_1324 == reference

    add_2143 = add_array_to_result(1.0, g_ovov, omega2)
    reference = "      call add_2143_to_1234(1.0d0, g_ovov, omega2, wf%n_v,wf%n_o,wf%n_v,wf%n_o)\n!\n"
    assert add_2143 == reference

    add_4123 = add_array_to_result(1.0, g_ovov_allocatable, omega2)
    reference = "      call add_4123_to_1234(1.0d0, g_ovov_jaib, omega2, wf%n_v,wf%n_o,wf%n_v,wf%n_o)\n"
    reference += "      call mem%dealloc(g_ovov_jaib)\n!\n"
    assert add_4123 == reference
