from einsum_to_eT import generate_eT_code_from_einsum
from numpy import zeros


def generate_omega_ccsd_b2_code():
    """CCSD B2 term: Omega_aibj = sum_kl sum_cd t_akbl t_cidj g_kcld"""
    n_o = 25
    n_v = n_o * 7

    g_oooo = zeros([n_o, n_o, n_o, n_o])
    g_vovo = zeros([n_v, n_o, n_v, n_o])
    g_ovov = zeros([n_o, n_v, n_o, n_v])
    t2 = zeros([n_v, n_o, n_v, n_o])
    omega2 = zeros([n_v, n_o, n_v, n_o])

    routines = ""
    routines = generate_eT_code_from_einsum(
        routine_name="construct_ccsd_b2_term1",
        prefactor=1.0,
        contraction_string="aibj->aibj",
        arrays=[g_vovo, omega2],
        symbols=["g_vovo", "omega2"],
    )

    routines += "!\n!\n"

    routines += generate_eT_code_from_einsum(
        routine_name="construct_ccsd_b2_term2",
        prefactor=1.0,
        contraction_string="akbl,kilj->aibj",
        arrays=[t2, g_oooo, omega2],
        symbols=["t2", "g_oooo", "omega2"],
    )

    routines += "!\n!\n"

    routines += generate_eT_code_from_einsum(
        routine_name="construct_ccsd_b2_term3",
        prefactor=1.0,
        contraction_string="akbl,cidj,kcld->aibj",
        arrays=[t2, t2, g_ovov, omega2],
        symbols=["t2", "t2", "g_ovov", "omega2"],
    )

    return routines


if __name__ == "__main__":
    code = generate_omega_ccsd_b2_code()
    print(code)


# julia> foreach(x -> print_code(x, "Ω"), omega_ai.terms)
# Ω_pq += +1.00000000e+00 * np.einsum("pq->pq", F_vo, optimize="optimal");
# Ω_pq += +2.00000000e+00 * np.einsum("rs,pqsr->pq", F_ov, t_vovo, optimize="optimal");
# Ω_pq += -1.00000000e+00 * np.einsum("rs,prsq->pq", F_ov, t_vovo, optimize="optimal");
# Ω_pq += +2.00000000e+00 * np.einsum("prst,rqts->pq", g_vvov, t_vovo, optimize="optimal");
# Ω_pq += -1.00000000e+00 * np.einsum("prst,rstq->pq", g_vvov, t_vovo, optimize="optimal");
# Ω_pq += -2.00000000e+00 * np.einsum("rqst,prts->pq", g_ooov, t_vovo, optimize="optimal");
# Ω_pq += +1.00000000e+00 * np.einsum("rqst,pstr->pq", g_ooov, t_vovo, optimize="optimal");
