import numpy as np

"""TODO:
1. Handle which type of wf
2. Handle indentation without hardcoding it
3. Handle more blas routines (blas_factory -> call write_blas_code)
4. Test if cholesky vectors could be used (capital J)
5. Could try to avoid allocating an additional intermediate
    for the result in the last contraction
6. Make index_to_dimension settable from outside
7. Make result a separate array again?
8. Make dger, dgemv, ddot ("bj,bj")
"""

index_to_dimension = {
    "ijklmno": "wf%n_o",
    "abcdefgh": "wf%n_v",
    "pqrstuvw": "wf%n_mo",
}


def contains_all_indices(string: str, indices: str):
    return all([(index in string) for index in indices])


def convert_index_string_to_dimensions(indices: str) -> list[str]:
    if indices == "":
        return ["1"]

    index_dimensions = [
        index_to_dimension[key]
        for i in indices
        for key in index_to_dimension
        if i in key
    ]

    if not index_dimensions:
        raise ValueError(
            "Expecting string containing only characters from this list: "
            + f"{', '.join(index_to_dimension.keys())}"
        )
    return index_dimensions


def get_n_elements(indices: str) -> str:
    index_dimensions = convert_index_string_to_dimensions(indices)
    return "*".join(index_dimensions)


def get_comma_separated_dimensions(indices: str) -> str:
    index_dimensions = convert_index_string_to_dimensions(indices)
    return ",".join(index_dimensions)


class tensor:
    def __init__(self, symbol: str, indices: str):
        self.symbol = symbol
        self.indices = indices

        self.dimensions = get_comma_separated_dimensions(self.indices)

    def __str__(self) -> str:
        return f"{self.symbol}_{self.indices}"

    def get_declaration(self, *attributes) -> str:
        declaration = f"      real(dp), dimension({self.dimensions})"
        if attributes:
            declaration += ", "
        declaration += ", ".join(attributes)
        declaration += f" :: {self.symbol}\n"

        return declaration

    # hash and eq implemented to use sets/dicts for removing duplicate tensors
    def __eq__(self, other):
        return self.symbol == other.symbol and len(self.indices) == len(other.indices)

    def __hash__(self) -> int:
        return hash(self.symbol)


class allocatable_tensor(tensor):
    def __init__(self, symbol: str, indices: str):
        super().__init__(f"{symbol}_{indices}", indices)
        self.allocated = False

    def __str__(self) -> str:
        return f"{self.symbol}"

    def allocate(self) -> str:
        alloc = f"      call mem%alloc({self.symbol}, {self.dimensions})\n"
        self.allocated = True
        return alloc

    def deallocate(self) -> str:
        dealloc = f"      call mem%dealloc({self.symbol})\n"
        self.allocated = False
        return dealloc

    def get_declaration(self, *attributes) -> str:
        rank = ",".join([":" for _ in self.indices])
        declaration = f"      real(dp), dimension({rank}), allocatable"
        if attributes:
            declaration += ", "
        declaration += ", ".join(attributes)
        declaration += f" :: {self.symbol}\n"

        return declaration


def get_routine_header(
    name: str,
    prefactor: float,
    tensors: list[tensor],
) -> str:
    """
    Writes subroutine like this:
    It is assumed that the last tensor is the output tensor

    subroutine construct_ccsd_b2(wf, t2, g_ovov, omega2)
    !!
    !!    Calculates:
    !!    omega2_aibj += 1.00 t2_akbl t2_cidj g_ovov_kcld omega2_aibj
    !!
        implicit none
    !
        class(ccs), intent(in) :: wf
    !
        real(dp), dimension(wf%n_v,wf%n_o,wf%n_v,wf%n_o), intent(in) :: t2
        real(dp), dimension(wf%n_o,wf%n_v,wf%n_o,wf%n_v), intent(in) :: g_ovov
        real(dp), dimension(wf%n_v,wf%n_o,wf%n_v,wf%n_o), intent(inout) :: omega2
    """

    # since 3.7 dicts preserve the insertion order
    unique_tensors = list(dict.fromkeys(tensors))
    parameters = ", ".join([t.symbol for t in unique_tensors])

    routine = f"   subroutine {name}(wf, {parameters})\n"
    routine += "!!\n"
    routine += "!!    Calculates:\n"
    routine += f"!!    {str(tensors[-1])} += {prefactor:.2f} " + " ".join(
        [str(t) for t in tensors[:-1]]
    )
    routine += "\n!!\n"
    routine += "      implicit none\n!\n"
    routine += "      class(ccs), intent(in) :: wf\n"
    routine += "!\n"

    for t in unique_tensors[:-1]:  # skip result tensor
        routine += t.get_declaration("intent(in)")

    routine += tensors[-1].get_declaration("intent(inout)")
    routine += "!\n"

    return routine


def parse_einsum_contraction_string(einsum_string: str) -> list[str]:
    from re import findall

    lhs, rhs = einsum_string.split("->")
    lhs = lhs.split(",")
    tensor_indices = lhs + [rhs]
    tensor_indices = [i.strip() for i in tensor_indices]

    matches = [findall(r"[^a-zA-Z ]", indices) for indices in lhs]

    if any(matches):
        raise ValueError("The einsum_string may only contain letters, commas and '->'")

    return tensor_indices


def determine_contraction(
    tensor1: tensor,
    tensor2: tensor,
    contraction_indices: str,
) -> tuple[dict(), dict()]:
    # Get order of contraction indices
    n_indices = len(contraction_indices)

    t1_last_indices = tensor1.indices[-n_indices:]
    t1_first_indices = tensor1.indices[:n_indices]

    t2_last_indices = tensor2.indices[-n_indices:]
    t2_first_indices = tensor2.indices[:n_indices]

    index_order = contraction_indices
    if index_order == "":
        index_order = contraction_indices
    elif contains_all_indices(t1_last_indices, contraction_indices):
        index_order = t1_last_indices
    elif contains_all_indices(t1_first_indices, contraction_indices):
        index_order = t1_first_indices
    elif contains_all_indices(t2_first_indices, contraction_indices):
        index_order = t2_first_indices
    elif contains_all_indices(t2_last_indices, contraction_indices):
        index_order = t2_last_indices

    # Get contraction result
    tensor1_indices = {"contracted": [], "remaining": [], "remaining string": ""}
    tensor1_indices["contracted"] = [tensor1.indices.find(i) for i in index_order]
    for i, str_i in enumerate(tensor1.indices):
        if str_i not in contraction_indices:
            tensor1_indices["remaining"].append(i)
            tensor1_indices["remaining string"] += str_i

    tensor2_indices = {"contracted": [], "remaining": [], "remaining string": ""}
    tensor2_indices["contracted"] = [tensor2.indices.find(i) for i in index_order]
    for i, str_i in enumerate(tensor2.indices):
        if str_i not in contraction_indices:
            tensor2_indices["remaining"].append(i)
            tensor2_indices["remaining string"] += str_i

    # Create contraction containing info how to handle the 2 tensors
    # Tensor 1:
    if index_order == t1_last_indices:
        contraction_t1 = {
            "transpose": False,
            "reorder": tuple(),
            "remaining string": tensor1_indices["remaining string"],
        }
    elif index_order == t1_first_indices:
        contraction_t1 = {
            "transpose": True,
            "reorder": tuple(),
            "remaining string": tensor1_indices["remaining string"],
        }
    else:
        reorder_to = tuple(tensor1_indices["remaining"] + tensor1_indices["contracted"])
        contraction_t1 = {
            "transpose": False,
            "reorder": reorder_to,
            "remaining string": tensor1_indices["remaining string"],
        }

    # Tensor 2:
    if index_order == t2_first_indices:
        contraction_t2 = {
            "transpose": False,
            "reorder": tuple(),
            "remaining string": tensor2_indices["remaining string"],
        }
    elif index_order == t2_last_indices:
        contraction_t2 = {
            "transpose": True,
            "reorder": tuple(),
            "remaining string": tensor2_indices["remaining string"],
        }
    else:
        reorder_to = tuple(tensor2_indices["contracted"] + tensor2_indices["remaining"])
        contraction_t2 = {
            "transpose": False,
            "reorder": reorder_to,
            "remaining string": tensor2_indices["remaining string"],
        }

    return contraction_t1, contraction_t2


def get_reordering(array: tensor, reordered: tensor, permutation: tuple[int]) -> str:
    routine_name = "sort_to_" + "".join(str(i + 1) for i in permutation)
    reordering = f"      call {routine_name}"
    reordering += f"({array.symbol}, {reordered.symbol}, {array.dimensions})\n"
    return reordering


def get_array_for_contraction(
    array: tensor,
    contraction: dict(),
) -> tuple[tensor, str, str]:
    """
    Analyzes contraction dict and determines if the array can be contracted as is,
    otherwise allocates new tensor and reorders `array` into the new tensor
    """

    if not contraction["reorder"]:
        return array, "", ""

    reordered_indices = [array.indices[i] for i in contraction["reorder"]]

    reordered_array = allocatable_tensor(f"{array.symbol}", "".join(reordered_indices))

    declarations = reordered_array.get_declaration()

    implementation = reordered_array.allocate()
    implementation += get_reordering(array, reordered_array, contraction["reorder"])

    A = reordered_array

    if isinstance(array, allocatable_tensor):
        implementation += array.deallocate()

    implementation += "!\n"

    return A, declarations, implementation


def write_dgemm(A, contraction_A, B, contraction_B, contraction_indices, result):
    M = get_n_elements(contraction_A["remaining string"])
    N = get_n_elements(contraction_B["remaining string"])
    K = get_n_elements(contraction_indices)

    if contraction_A["transpose"]:
        trans_1 = "T"
        lda = K
    else:
        trans_1 = "N"
        lda = M

    if contraction_B["transpose"]:
        trans_2 = "T"
        ldb = N
    else:
        trans_2 = "N"
        ldb = K

    dgemm = f"      call dgemm('{trans_1}', '{trans_2}', &\n"
    dgemm += f"                 {M}, &\n"
    dgemm += f"                 {N}, &\n"
    dgemm += f"                 {K}, &\n"
    dgemm += f"                 one, &\n"
    dgemm += f"                 {A.symbol}, &\n"
    dgemm += f"                 {lda}, &\n"
    dgemm += f"                 {B.symbol}, &\n"
    dgemm += f"                 {ldb}, &\n"
    dgemm += f"                 zero, &\n"
    dgemm += f"                 {result.symbol}, &\n"
    dgemm += f"                 {M})\n"

    implementation = dgemm + "!\n!\n"

    if isinstance(A, allocatable_tensor):
        implementation += A.deallocate()

    if isinstance(B, allocatable_tensor):
        implementation += B.deallocate()

    return implementation


def implement_contraction(
    tensor_1: tensor,
    contraction_t1: dict(),
    tensor_2: tensor,
    contraction_t2: dict(),
    result: tensor,
    contraction_indices: str,
) -> tuple[str, str]:

    A, declarations, implementation = get_array_for_contraction(
        tensor_1, contraction_t1
    )

    B, declarations_2, implementation_2 = get_array_for_contraction(
        tensor_2, contraction_t2
    )

    declarations += declarations_2
    implementation += implementation_2

    declarations += result.get_declaration()
    implementation += result.allocate()
    implementation += "!\n!\n"

    implementation += write_dgemm(
        A,
        contraction_t1,
        B,
        contraction_t2,
        contraction_indices,
        result,
    )

    return declarations, implementation


def create_contraction_code(
    variable_definitions: str,
    contraction_code: str,
    n_intermediates: int,
    tensors: list[tensor],
    contraction: tuple,
):
    from blas_factory import determine_blas_routine

    tensor_indices, contraction_indices, routine = determine_blas_routine(contraction)

    if routine not in ["dgemm", "dgemv", "dger"]:
        raise NotImplementedError(
            "Only contractions that can be represented as dgemm (dgemv, and dger) can be implmented."
        )

    index_tensor_1, index_tensor_2 = tensor_indices

    contraction_t1, contraction_t2 = determine_contraction(
        tensors[index_tensor_1],
        tensors[index_tensor_2],
        contraction_indices,
    )

    result_indices = (
        contraction_t1["remaining string"] + contraction_t2["remaining string"]
    )

    result_tensor = allocatable_tensor(
        f"intermediate_{n_intermediates}", result_indices
    )
    n_intermediates += 1

    definitions, implementation = implement_contraction(
        tensors[index_tensor_1],
        contraction_t1,
        tensors[index_tensor_2],
        contraction_t2,
        result_tensor,
        contraction_indices,
    )

    variable_definitions += definitions
    contraction_code += implementation

    for i in tensor_indices:
        tensors.pop(i)

    tensors.insert(-1, result_tensor)

    return variable_definitions, contraction_code, n_intermediates, tensors


def sanity_checks(
    contraction_string: str, arrays: list[np.ndarray], symbols: list[str]
):
    """Performs checks on the correctnes of the input parameters"""

    if len(arrays) > len(symbols):
        raise ValueError("More arrays than symbols specified.")
    elif len(arrays) < len(symbols):
        print(
            f"More symbols than arrays specified, choosing first {len(arrays)} symbols."
        )

    array_indices = parse_einsum_contraction_string(contraction_string)

    if len(arrays) < len(array_indices):
        raise ValueError(
            f"Not enough arrays ({len(arrays)}) for the given contraction: {contraction_string}"
        )
    elif len(arrays) > len(array_indices):
        raise ValueError(
            f"Too many arrays ({len(arrays)}) for the given contraction: {contraction_string}"
        )

    if len(symbols) < len(array_indices):
        raise ValueError(
            f"{len(symbols)} symbols not enough for the given contraction: {contraction_string}"
        )


def add_array_to_result(prefactor: float, array: tensor, result: tensor) -> str:
    size = get_n_elements(array.indices)

    if array.indices == result.indices:
        code = f"      call daxpy({size}, {prefactor}d0, {array.symbol}, 1, {result.symbol}, 1)\n"
    else:
        index_to_number = {
            character: str(i + 1) for i, character in enumerate(result.indices)
        }
        from_ = "".join([index_to_number[i] for i in array.indices])
        to_ = "".join([index_to_number[i] for i in result.indices])
        code = f"      call add_{from_}_to_{to_}"
        code += (
            f"({prefactor}d0, {array.symbol}, {result.symbol}, {result.dimensions})\n"
        )

    if isinstance(array, allocatable_tensor):
        code += array.deallocate()
    code += "!\n"

    return code


def generate_eT_code_from_einsum(
    routine_name: str,
    prefactor: float,
    contraction_string: str,
    arrays: list[np.ndarray],
    symbols: list[str],
) -> str:
    """
    Example call:
    code = generate_eT_code_from_einsum(
        routine_name="construct_ccsd_b2",
        prefactor=1.0,
        contraction_string="akbl,cidj,kcld->aibj",
        arrays=[t2, t2, g_ovov],
        symbols=["t2", "t2", "g_ovov"]
    )

    Parameters:
    - prefactor: prefactor to multiply the entire term with.
    - contraction_string:
      String indicating the contraction to be evaluated.
      Same input as for numpy.einsum or numpy.einsum_path
      Note:
      * ellipsis are not supported

    - arrays: (list of numpy arrays)
      The arrays to apply einsum on.

      Important:
      * For finding the optimal reduction path np.einsum(...,optimize='optimal') is used.
      * The arrays should have a relastic size for optimal code generation.

    - symbols: (list of string)
      Symbols/Variable routine_names for the arrays in the code produced.

      * temp_ and result are reserved for internal arrays
    """

    sanity_checks(contraction_string, arrays, symbols)

    code = ""

    n_intermediates = 0

    """
    Due to einsum_call=True, we get a list containing the optimal contractions.
    Each element of the list is a tuple containing:
    1. Indices specifying the 2 arrays to be contracted in "arrays"-list
    2. Set of the names of the indices we contract over
    3. Einsum contraction string for current intermediate
    4. List of index strings for the remaining arrays in the full contraction and the new intermediate
    5. Logical if the contraction can easily represeneted as BLAS routine
    """
    _, contraction_list = np.einsum_path(
        contraction_string, *arrays[:-1], optimize="optimal", einsum_call=True
    )

    array_indices = parse_einsum_contraction_string(contraction_string)
    tensors = [
        tensor(symbol, indices) for symbol, indices in zip(symbols, array_indices)
    ]
    code = get_routine_header(routine_name, prefactor, tensors)

    if len(tensors) > 2:
        variable_definitions = ""
        contraction_code = ""

        for contraction in contraction_list:
            (
                variable_definitions,
                contraction_code,
                n_intermediates,
                tensors,
            ) = create_contraction_code(
                variable_definitions,
                contraction_code,
                n_intermediates,
                tensors,
                contraction,
            )

        code += variable_definitions + f"!\n"
        code += contraction_code + f"!\n"

        if len(tensors) != 2:
            raise RuntimeError(
                "Number of tensors is different from 2 after creating contraction code."
            )

    code += add_array_to_result(prefactor, tensors[0], tensors[1])

    code += f"   end subroutine {routine_name}\n"

    return code
