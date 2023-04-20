import pytest
from einsum_to_eT import tensor, allocatable_tensor
from einsum_to_eT import determine_contraction, get_array_for_contraction, write_dgemm


class TestContractionCode:

    t2 = tensor("t2", "cidj")
    g_vvvv = tensor("g_vvvv", "acbd")

    t2_cdij = allocatable_tensor("t2", "cdij")
    g_abcd = allocatable_tensor("g_vvvv", "abcd")

    t2_ijcd = allocatable_tensor("t2", "ijcd")
    g_cdab = allocatable_tensor("g_vvvv", "cdab")

    c_t2_ijcd, c_g_cdab = determine_contraction(t2_ijcd, g_cdab, "cd")
    c_t2_cdij, c_g_abcd = determine_contraction(t2_cdij, g_abcd, "cd")
    c_t2, c_g_vvvv = determine_contraction(t2, g_vvvv, "cd")

    result_ijab = allocatable_tensor("result", "ijab")

    def test_determine_contraction(self):
        """t2_ijcd g_cdab -> aibj"""
        reference_t2 = {
            "transpose": False,
            "reorder": (),
            "remaining string": "ij",
        }
        reference_g_vvvv = {
            "transpose": False,
            "reorder": (),
            "remaining string": "ab",
        }

        assert reference_t2 == self.c_t2_ijcd
        assert reference_g_vvvv == self.c_g_cdab

    def test_determine_contraction_transpose(self):
        """t2_cdij g_abcd -> aibj"""
        reference_t2 = {
            "transpose": True,
            "reorder": (),
            "remaining string": "ij",
        }
        reference_g_vvvv = {
            "transpose": True,
            "reorder": (),
            "remaining string": "ab",
        }

        assert reference_t2 == self.c_t2_cdij
        assert reference_g_vvvv == self.c_g_abcd

    def test_determine_contraction_reordering(self):
        """t2_cidj g_acbd -> aibj"""
        reference_t2 = {
            "transpose": False,
            "reorder": (1, 3, 0, 2),
            "remaining string": "ij",
        }
        reference_g_vvvv = {
            "transpose": False,
            "reorder": (1, 3, 0, 2),
            "remaining string": "ab",
        }

        assert reference_t2 == self.c_t2
        assert reference_g_vvvv == self.c_g_vvvv

    def test_dgemm_NN(self):

        A, declaration_A, implementation_A = get_array_for_contraction(
            self.t2_ijcd, self.c_t2_ijcd
        )
        assert declaration_A == ""
        assert implementation_A == ""
        assert A == self.t2_ijcd

        B, declaration_B, implementation_B = get_array_for_contraction(
            self.g_cdab, self.c_g_cdab
        )
        assert declaration_B == ""
        assert implementation_B == ""
        assert B == self.g_cdab

        dgemm = write_dgemm(
            A,
            self.c_t2_ijcd,
            B,
            self.c_g_cdab,
            "cd",
            self.result_ijab,
        )

        reference_dgemm = """      call dgemm('N', 'N', &
                 wf%n_o*wf%n_o, &
                 wf%n_v*wf%n_v, &
                 wf%n_v*wf%n_v, &
                 one, &
                 t2_ijcd, &
                 wf%n_o*wf%n_o, &
                 g_vvvv_cdab, &
                 wf%n_v*wf%n_v, &
                 zero, &
                 result_ijab, &
                 wf%n_o*wf%n_o)\n"""
        reference_dgemm += "!\n!\n"
        reference_dgemm += "      call mem%dealloc(t2_ijcd)\n"
        reference_dgemm += "      call mem%dealloc(g_vvvv_cdab)\n"

        assert dgemm == reference_dgemm

    def test_dgemm_NT(self):

        A, declaration_A, implementation_A = get_array_for_contraction(
            self.t2_ijcd, self.c_t2_ijcd
        )
        assert declaration_A == ""
        assert implementation_A == ""
        assert A == self.t2_ijcd

        B, declaration_B, implementation_B = get_array_for_contraction(
            self.g_abcd, self.c_g_abcd
        )
        assert declaration_B == ""
        assert implementation_B == ""
        assert B == self.g_abcd

        dgemm = write_dgemm(
            A,
            self.c_t2_ijcd,
            B,
            self.c_g_abcd,
            "cd",
            self.result_ijab,
        )

        reference_dgemm = """      call dgemm('N', 'T', &
                 wf%n_o*wf%n_o, &
                 wf%n_v*wf%n_v, &
                 wf%n_v*wf%n_v, &
                 one, &
                 t2_ijcd, &
                 wf%n_o*wf%n_o, &
                 g_vvvv_abcd, &
                 wf%n_v*wf%n_v, &
                 zero, &
                 result_ijab, &
                 wf%n_o*wf%n_o)\n"""
        reference_dgemm += "!\n!\n"
        reference_dgemm += "      call mem%dealloc(t2_ijcd)\n"
        reference_dgemm += "      call mem%dealloc(g_vvvv_abcd)\n"

        assert dgemm == reference_dgemm

    def test_dgemm_TN(self):

        A, declaration_A, implementation_A = get_array_for_contraction(
            self.t2_cdij, self.c_t2_cdij
        )
        assert declaration_A == ""
        assert implementation_A == ""
        assert A == self.t2_cdij

        B, declaration_B, implementation_B = get_array_for_contraction(
            self.g_cdab, self.c_g_cdab
        )
        assert declaration_B == ""
        assert implementation_B == ""
        assert B == self.g_cdab

        dgemm = write_dgemm(
            A,
            self.c_t2_cdij,
            B,
            self.c_g_cdab,
            "cd",
            self.result_ijab,
        )

        reference_dgemm = """      call dgemm('T', 'N', &
                 wf%n_o*wf%n_o, &
                 wf%n_v*wf%n_v, &
                 wf%n_v*wf%n_v, &
                 one, &
                 t2_cdij, &
                 wf%n_v*wf%n_v, &
                 g_vvvv_cdab, &
                 wf%n_v*wf%n_v, &
                 zero, &
                 result_ijab, &
                 wf%n_o*wf%n_o)\n"""
        reference_dgemm += "!\n!\n"
        reference_dgemm += "      call mem%dealloc(t2_cdij)\n"
        reference_dgemm += "      call mem%dealloc(g_vvvv_cdab)\n"

        assert dgemm == reference_dgemm

    def test_dgemm_TT(self):

        A, declaration_A, implementation_A = get_array_for_contraction(
            self.t2_cdij, self.c_t2_cdij
        )
        assert declaration_A == ""
        assert implementation_A == ""
        assert A == self.t2_cdij

        B, declaration_B, implementation_B = get_array_for_contraction(
            self.g_abcd, self.c_g_abcd
        )
        assert declaration_B == ""
        assert implementation_B == ""
        assert B == self.g_abcd

        dgemm = write_dgemm(
            A,
            self.c_t2_cdij,
            B,
            self.c_g_abcd,
            "cd",
            self.result_ijab,
        )

        reference_dgemm = """      call dgemm('T', 'T', &
                 wf%n_o*wf%n_o, &
                 wf%n_v*wf%n_v, &
                 wf%n_v*wf%n_v, &
                 one, &
                 t2_cdij, &
                 wf%n_v*wf%n_v, &
                 g_vvvv_abcd, &
                 wf%n_v*wf%n_v, &
                 zero, &
                 result_ijab, &
                 wf%n_o*wf%n_o)\n"""
        reference_dgemm += "!\n!\n"
        reference_dgemm += "      call mem%dealloc(t2_cdij)\n"
        reference_dgemm += "      call mem%dealloc(g_vvvv_abcd)\n"

        assert dgemm == reference_dgemm

    def test_dgemm_NN_reordering(self):

        A, declaration_A, implementation_A = get_array_for_contraction(
            self.t2, self.c_t2
        )

        decl = "      real(dp), dimension(:,:,:,:), allocatable :: t2_ijcd\n"
        assert declaration_A == decl

        impl = "      call mem%alloc(t2_ijcd, wf%n_o,wf%n_o,wf%n_v,wf%n_v)\n"
        impl += "      call sort_to_2413(t2, t2_ijcd, wf%n_v,wf%n_o,wf%n_v,wf%n_o)"
        impl += "\n!\n"
        assert implementation_A == impl
        assert A == self.t2_ijcd

        B, declaration_B, implementation_B = get_array_for_contraction(
            self.g_vvvv, self.c_g_vvvv
        )

        decl = "      real(dp), dimension(:,:,:,:), allocatable :: g_vvvv_cdab\n"
        assert declaration_B == decl

        impl = "      call mem%alloc(g_vvvv_cdab, wf%n_v,wf%n_v,wf%n_v,wf%n_v)\n"
        impl += (
            "      call sort_to_2413(g_vvvv, g_vvvv_cdab, wf%n_v,wf%n_v,wf%n_v,wf%n_v)"
        )
        impl += "\n!\n"
        assert implementation_B == impl
        assert B == self.g_cdab

        dgemm = write_dgemm(
            A,
            self.c_t2_ijcd,
            B,
            self.c_g_vvvv,
            "cd",
            self.result_ijab,
        )

        reference_dgemm = """      call dgemm('N', 'N', &
                 wf%n_o*wf%n_o, &
                 wf%n_v*wf%n_v, &
                 wf%n_v*wf%n_v, &
                 one, &
                 t2_ijcd, &
                 wf%n_o*wf%n_o, &
                 g_vvvv_cdab, &
                 wf%n_v*wf%n_v, &
                 zero, &
                 result_ijab, &
                 wf%n_o*wf%n_o)\n"""
        reference_dgemm += "!\n!\n"
        reference_dgemm += "      call mem%dealloc(t2_ijcd)\n"
        reference_dgemm += "      call mem%dealloc(g_vvvv_cdab)\n"

        assert dgemm == reference_dgemm

    def test_implement_contraction_NT(self):
        from einsum_to_eT import implement_contraction

        g_abcd = tensor("g_abcd", "abcd")

        declarations, code = implement_contraction(
            self.t2, self.c_t2, g_abcd, self.c_g_abcd, self.result_ijab, "cd"
        )

        decl = "      real(dp), dimension(:,:,:,:), allocatable :: t2_ijcd\n"
        decl += "      real(dp), dimension(:,:,:,:), allocatable :: result_ijab\n"
        assert declarations == decl

        ref = "      call mem%alloc(t2_ijcd, wf%n_o,wf%n_o,wf%n_v,wf%n_v)\n"
        ref += "      call sort_to_2413(t2, t2_ijcd, wf%n_v,wf%n_o,wf%n_v,wf%n_o)\n"
        ref += "!\n"
        ref += "      call mem%alloc(result_ijab, wf%n_o,wf%n_o,wf%n_v,wf%n_v)\n"
        ref += "!\n!\n"

        ref += """      call dgemm('N', 'T', &
                 wf%n_o*wf%n_o, &
                 wf%n_v*wf%n_v, &
                 wf%n_v*wf%n_v, &
                 one, &
                 t2_ijcd, &
                 wf%n_o*wf%n_o, &
                 g_abcd, &
                 wf%n_v*wf%n_v, &
                 zero, &
                 result_ijab, &
                 wf%n_o*wf%n_o)\n"""
        ref += "!\n!\n"
        ref += "      call mem%dealloc(t2_ijcd)\n"

        assert code == ref
