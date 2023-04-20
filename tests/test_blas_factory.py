from blas_factory import *


class TestIsBLASRoutine:
    outer = ["ai", "bj", ""]
    outer_transpose1 = ["ia", "bj", ""]
    outer_transpose2 = ["ai", "jb", ""]
    outer_transpose_both = ["ia", "jb", ""]

    matrix_matrix = ["ajbk", "jbki", "jbk"]
    matrix_matrix_permute1 = ["jakb", "bjik", "jbk"]
    matrix_matrix_permute2 = ["ajbk", "jikb", "jbk"]

    matrix_vector = ["ljck", "ck", "ck"]
    matrix_vector_transpose2 = ["ljck", "kc", "ck"]
    matrix_vector_transpose3 = ["ljck", "ck", "kc"]
    matrix_vector_permutation = ["lcjk", "ck", "kc"]

    scalar = ["ai", "ai", "ai"]
    scalar_transpose1 = ["ia", "ai", "ai"]
    scalar_transpose2 = ["ai", "ia", "ai"]
    scalar_transpose3 = ["ai", "ai", "ia"]

    def test_is_outer_product(self):
        assert is_outer_product("")
        assert not is_outer_product("c")
        assert not is_outer_product("k")
        assert not is_outer_product("ck")

    def test_is_scalar_product(self) -> bool:
        assert not is_scalar_product(*self.outer)
        assert not is_scalar_product(*self.outer_transpose1)
        assert not is_scalar_product(*self.outer_transpose2)
        assert not is_scalar_product(*self.outer_transpose_both)

        assert not is_scalar_product(*self.matrix_matrix)
        assert not is_scalar_product(*self.matrix_matrix_permute1)
        assert not is_scalar_product(*self.matrix_matrix_permute2)

        assert not is_scalar_product(*self.matrix_vector)
        assert not is_scalar_product(*self.matrix_vector_transpose2)
        assert not is_scalar_product(*self.matrix_vector_transpose3)
        assert not is_scalar_product(*self.matrix_vector_permutation)

        assert is_scalar_product(*self.scalar)
        assert is_scalar_product(*self.scalar_transpose1)
        assert is_scalar_product(*self.scalar_transpose2)
        assert is_scalar_product(*self.scalar_transpose3)

    def test_is_matrix_vector_product(self):
        assert not is_matrix_vector_product(*self.outer)
        assert not is_matrix_vector_product(*self.outer_transpose1)
        assert not is_matrix_vector_product(*self.outer_transpose2)
        assert not is_matrix_vector_product(*self.outer_transpose_both)

        assert not is_matrix_vector_product(*self.matrix_matrix)
        assert not is_matrix_vector_product(*self.matrix_matrix_permute1)
        assert not is_matrix_vector_product(*self.matrix_matrix_permute2)

        assert is_matrix_vector_product(*self.matrix_vector)
        assert is_matrix_vector_product(*self.matrix_vector_transpose2)
        assert is_matrix_vector_product(*self.matrix_vector_transpose3)
        assert is_matrix_vector_product(*self.matrix_vector_permutation)

        assert not is_matrix_vector_product(*self.scalar)
        assert not is_matrix_vector_product(*self.scalar_transpose1)
        assert not is_matrix_vector_product(*self.scalar_transpose2)
        assert not is_matrix_vector_product(*self.scalar_transpose3)

    def test_is_matrix_matrix_product(self):
        assert not is_matrix_matrix_product(*self.outer)
        assert not is_matrix_matrix_product(*self.outer_transpose1)
        assert not is_matrix_matrix_product(*self.outer_transpose2)
        assert not is_matrix_matrix_product(*self.outer_transpose_both)

        assert is_matrix_matrix_product(*self.matrix_matrix)
        assert is_matrix_matrix_product(*self.matrix_matrix_permute1)
        assert is_matrix_matrix_product(*self.matrix_matrix_permute2)

        assert not is_matrix_matrix_product(*self.matrix_vector)
        assert not is_matrix_matrix_product(*self.matrix_vector_transpose2)
        assert not is_matrix_matrix_product(*self.matrix_vector_transpose3)
        assert not is_matrix_matrix_product(*self.matrix_vector_permutation)

        assert not is_matrix_matrix_product(*self.scalar)
        assert not is_matrix_matrix_product(*self.scalar_transpose1)
        assert not is_matrix_matrix_product(*self.scalar_transpose2)
        assert not is_matrix_matrix_product(*self.scalar_transpose3)
