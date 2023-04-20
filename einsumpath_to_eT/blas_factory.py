from einsum_to_eT import parse_einsum_contraction_string
from einsum_to_eT import contains_all_indices


def is_outer_product(contraction_indices) -> bool:
    return not contraction_indices


def is_matrix_matrix_product(
    indices_t1: str, indices_t2: str, contraction_indices: str
) -> bool:
    if not contraction_indices:
        return False
    return (
        contains_all_indices(indices_t1, contraction_indices)
        and len(indices_t1) > len(contraction_indices)
        and contains_all_indices(indices_t2, contraction_indices)
        and len(indices_t2) > len(contraction_indices)
    )


def is_scalar_product(
    indices_t1: str, indices_t2: str, contraction_indices: str
) -> bool:
    sorted_indices_t1 = sorted(indices_t1)
    sorted_indices_t2 = sorted(indices_t2)
    sorted_contraction = sorted(contraction_indices)
    return (
        sorted_indices_t1 == sorted_contraction
        and sorted_indices_t2 == sorted_contraction
    )


def is_matrix_vector_product(
    indices_t1: str, indices_t2: str, contraction_indices: str
) -> bool:
    sorted_indices_t1 = sorted(indices_t1)
    sorted_indices_t2 = sorted(indices_t2)
    sorted_contraction = sorted(contraction_indices)
    return (sorted_indices_t1 == sorted_contraction) != (
        sorted_indices_t2 == sorted_contraction
    )


def determine_blas_routine(contraction: tuple):
    """
    Receives tuple from einsum_path(..., einsum_call=True)
    determining the contraction and determines which routine to implement
    """
    (
        tensor_indices,
        contraction_indices,
        contraction_string,
        remaining_tensor_indices,
        is_blas,
    ) = contraction

    contraction_indices = "".join(sorted(list(contraction_indices)))
    indices_t1, indices_t2, _ = parse_einsum_contraction_string(contraction_string)
    blas_routine = ""
    if is_matrix_matrix_product(indices_t1, indices_t2, contraction_indices):
        blas_routine = "dgemm"
    elif is_matrix_vector_product(indices_t1, indices_t2, contraction_indices):
        blas_routine = "dgemv"
    elif is_outer_product(contraction_indices):
        blas_routine = "dger"
    elif is_scalar_product(indices_t1, indices_t2, contraction_indices):
        blas_routine = "ddot"
    else:
        NotImplementedError("Could not recognize blas routine to implement.")

    return tensor_indices, contraction_indices, blas_routine
