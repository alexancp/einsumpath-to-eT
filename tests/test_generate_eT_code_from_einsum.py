import pytest
from einsum_to_eT import generate_eT_code_from_einsum, sanity_checks
from numpy import zeros

n_o = 10
n_v = n_o * 7

g_ovov = zeros([n_o, n_v, n_o, n_v])
g_oooo = zeros([n_o, n_o, n_o, n_o])
t2 = zeros([n_v, n_o, n_v, n_o])
omega2 = zeros([n_v, n_o, n_v, n_o])


def test_sanity_checks():
    contraction = "akbl,kilj->aibj"
    arrays = [t2, g_oooo, omega2]
    symbols = ["t2", "g_oooo", "omega2"]

    sanity_checks(contraction, arrays, symbols)

    with pytest.raises(ValueError):
        sanity_checks(contraction, arrays, symbols[:-1])

    with pytest.raises(ValueError):
        sanity_checks(contraction, arrays[:-1], symbols)

    with pytest.raises(ValueError):
        sanity_checks(contraction, arrays[:-1], symbols[:-1])

    arrays_plus_one = [t2, g_oooo, omega2, g_ovov]
    symbols_plus_one = ["t2", "g_oooo", "omega2", "g_ovov"]

    with pytest.raises(ValueError):
        sanity_checks(contraction, arrays_plus_one, symbols_plus_one[:-1])

    sanity_checks(contraction, arrays_plus_one[:-1], symbols_plus_one)


def test_generate_eT_code_from_einsum():
    # CCSD B2 term 3: Omega_aibj = sum_kl sum_cd t_akbl t_cidj g_kcld
    routine = generate_eT_code_from_einsum(
        routine_name="construct_ccsd_b2_term3",
        prefactor=1.0,
        contraction_string="akbl,cidj,kcld->aibj",
        arrays=[t2, t2, g_ovov, omega2],
        symbols=["t2", "t2", "g_ovov", "omega2"],
    )

    reference = """   subroutine construct_ccsd_b2_term3(wf, t2, g_ovov, omega2)
!!
!!    Calculates:
!!    omega2_aibj += 1.00 t2_akbl t2_cidj g_ovov_kcld
!!
      implicit none
!
      class(ccs), intent(in) :: wf
!
      real(dp), dimension(wf%n_v,wf%n_o,wf%n_v,wf%n_o), intent(in) :: t2
      real(dp), dimension(wf%n_o,wf%n_v,wf%n_o,wf%n_v), intent(in) :: g_ovov
      real(dp), dimension(wf%n_v,wf%n_o,wf%n_v,wf%n_o), intent(inout) :: omega2
!
      real(dp), dimension(:,:,:,:), allocatable :: g_ovov_klcd
      real(dp), dimension(:,:,:,:), allocatable :: t2_cdij
      real(dp), dimension(:,:,:,:), allocatable :: intermediate_0_klij
      real(dp), dimension(:,:,:,:), allocatable :: t2_klab
      real(dp), dimension(:,:,:,:), allocatable :: intermediate_1_ijab
!
      call mem%alloc(g_ovov_klcd, wf%n_o,wf%n_o,wf%n_v,wf%n_v)
      call sort_to_1324(g_ovov, g_ovov_klcd, wf%n_o,wf%n_v,wf%n_o,wf%n_v)
!
      call mem%alloc(t2_cdij, wf%n_v,wf%n_v,wf%n_o,wf%n_o)
      call sort_to_1324(t2, t2_cdij, wf%n_v,wf%n_o,wf%n_v,wf%n_o)
!
      call mem%alloc(intermediate_0_klij, wf%n_o,wf%n_o,wf%n_o,wf%n_o)
!
!
      call dgemm('N', 'N', &
                 wf%n_o*wf%n_o, &
                 wf%n_o*wf%n_o, &
                 wf%n_v*wf%n_v, &
                 one, &
                 g_ovov_klcd, &
                 wf%n_o*wf%n_o, &
                 t2_cdij, &
                 wf%n_v*wf%n_v, &
                 zero, &
                 intermediate_0_klij, &
                 wf%n_o*wf%n_o)
!
!
      call mem%dealloc(g_ovov_klcd)
      call mem%dealloc(t2_cdij)
      call mem%alloc(t2_klab, wf%n_o,wf%n_o,wf%n_v,wf%n_v)
      call sort_to_2413(t2, t2_klab, wf%n_v,wf%n_o,wf%n_v,wf%n_o)
!
      call mem%alloc(intermediate_1_ijab, wf%n_o,wf%n_o,wf%n_v,wf%n_v)
!
!
      call dgemm('T', 'N', &
                 wf%n_o*wf%n_o, &
                 wf%n_v*wf%n_v, &
                 wf%n_o*wf%n_o, &
                 one, &
                 intermediate_0_klij, &
                 wf%n_o*wf%n_o, &
                 t2_klab, &
                 wf%n_o*wf%n_o, &
                 zero, &
                 intermediate_1_ijab, &
                 wf%n_o*wf%n_o)
!
!
      call mem%dealloc(intermediate_0_klij)
      call mem%dealloc(t2_klab)
!
      call add_2413_to_1234(1.0d0, intermediate_1_ijab, omega2, wf%n_v,wf%n_o,wf%n_v,wf%n_o)
      call mem%dealloc(intermediate_1_ijab)
!
   end subroutine construct_ccsd_b2_term3
"""

    assert routine == reference
