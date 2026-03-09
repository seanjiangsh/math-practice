"""Linear algebra utilities for math practice notebooks."""

from __future__ import annotations

from numbers import Number
from typing import Iterable, List, Literal, Optional, Sequence, Tuple, TypedDict, Union

import sympy as sp
MatrixLike = Union[sp.MatrixBase, Sequence[Sequence[Number]]]
VectorLike = Union[sp.MatrixBase, Sequence[Number]]
PermLike = Sequence[Union[int, Tuple[int, int], Sequence[int]]]


RefRrefDetails = TypedDict(
	"RefRrefDetails",
	{
		"A": sp.Matrix,
		"REF": sp.Matrix,
		"RREF": sp.Matrix,
		"rank(A)": int,
		"pivots": Tuple[int, ...],
		"PA_LU": Optional[Tuple[sp.Matrix, sp.Matrix, sp.Matrix]],
		"b": sp.Matrix,
		"[A|b]": sp.Matrix,
		"RREF([A|b])": sp.Matrix,
		"rank([A|b])": int,
		"solution_type": Literal["no solution", "unique", "infinite"],
		"solution_dimension": Optional[int],
		"aug_pivots": Tuple[int, ...],
	},
	total=False,
)


def _to_matrix(data: MatrixLike) -> sp.Matrix:
	if isinstance(data, sp.MatrixBase):
		return sp.Matrix(data)
	return sp.Matrix(data)


def _to_column_vector(data: VectorLike, rows: int) -> sp.Matrix:
	vec = _to_matrix(data)
	if vec.cols == 1 and vec.rows == rows:
		return vec
	if vec.rows == 1 and vec.cols == rows:
		return vec.T
	if vec.rows == rows and vec.cols == 1:
		return vec
	raise ValueError("b must be a vector compatible with A's row count")


def _perm_to_matrix(perm: PermLike, size: int) -> sp.Matrix:
	if not perm:
		return sp.eye(size)
	if all(isinstance(item, int) for item in perm) and len(perm) == size:
		p = sp.zeros(size)
		for i, j in enumerate(perm):
			p[i, j] = 1
		return p
	p = sp.eye(size)
	for swap in perm:
		if isinstance(swap, (tuple, list)) and len(swap) == 2:
			i, j = swap
			p.row_swap(i, j)
	return p


def _lu_steps_augmented(
	A: sp.Matrix,
	b: Optional[sp.Matrix],
) -> List[Tuple[List[str], Optional[sp.Matrix], sp.Matrix, sp.Matrix]]:
	mat = A
	if b is not None:
		mat = A.row_join(b)

	m = A.rows
	n = A.cols
	L = sp.eye(m)
	U = mat.copy()
	P = sp.eye(m)
	steps: List[Tuple[List[str], Optional[sp.Matrix], sp.Matrix, sp.Matrix]] = []

	for k in range(min(m, n)):
		if U[k, k] == 0:
			swap_row = None
			for i in range(k + 1, m):
				if U[i, k] != 0:
					swap_row = i
					break
			if swap_row is not None:
				U.row_swap(k, swap_row)
				P.row_swap(k, swap_row)
				if k > 0:
					L.row_swap(k, swap_row)
				steps.append(
					(
						[f"P: swap R{k + 1} <-> R{swap_row + 1}"],
						P.copy(),
						L.copy(),
						U.copy(),
					)
				)
		pivot = U[k, k]
		if pivot == 0:
			continue
		for i in range(k + 1, m):
			mult = sp.simplify(U[i, k] / pivot)
			if mult == 0:
				continue
			L[i, k] = mult
			U[i, :] = U[i, :] - mult * U[k, :]
			steps.append(
				(
					[f"m_{i + 1}{k + 1} = {mult}"],
					None,
					L.copy(),
					U.copy(),
				)
			)

	return steps


def _rref_steps_from_ref(
	mat: sp.Matrix,
	*,
	pivot_cols: Optional[int] = None,
) -> List[Tuple[str, sp.Matrix]]:
	steps: List[Tuple[str, sp.Matrix]] = []
	A = mat.copy()
	rows, cols = A.rows, A.cols
	if pivot_cols is None or pivot_cols > cols:
		pivot_cols = cols
	pivots: List[Tuple[int, int]] = []

	for r in range(rows):
		pivot_col = None
		for c in range(pivot_cols):
			if A[r, c] != 0:
				pivot_col = c
				break
		if pivot_col is None:
			continue
		pivots.append((r, pivot_col))

	# Step 1: Scale all pivot rows to have pivot = 1
	for r, c in pivots:
		pivot = A[r, c]
		if pivot != 1:
			A.row_op(r, lambda v, _: v / pivot)
			steps.append((f"R{r + 1} <- (1/{pivot}) R{r + 1}", A.copy()))

	# Step 2: Eliminate above pivots, from right to left, top to bottom
	for r, c in reversed(pivots):
		for i in range(r):
			factor = A[i, c]
			if factor != 0:
				A.row_op(i, lambda v, j: v - factor * A[r, j])
				steps.append(
					(f"R{i + 1} <- R{i + 1} - ({factor}) R{r + 1}", A.copy())
				)

	return steps


def _rref_from_ref(mat: sp.Matrix, pivot_cols: int) -> sp.Matrix:
	A = mat.copy()
	rows = A.rows
	pivots: List[Tuple[int, int]] = []

	for r in range(rows):
		pivot_col = None
		for c in range(pivot_cols):
			if A[r, c] != 0:
				pivot_col = c
				break
		if pivot_col is None:
			continue
		pivots.append((r, pivot_col))

	for r, c in pivots:
		pivot = A[r, c]
		if pivot != 1:
			A.row_op(r, lambda v, _: v / pivot)

	for r, c in reversed(pivots):
		for i in range(r):
			factor = A[i, c]
			if factor != 0:
				A.row_op(i, lambda v, j: v - factor * A[r, j])

	return A


def matrix_ref_rref_details(
	A: MatrixLike,
	b: Optional[VectorLike] = None,
	*,
	show: bool = True,
	show_steps: bool = False,
	stop_before_aug_pivot: Optional[bool] = None,
	symbol: str = "A",
) -> RefRrefDetails:
	"""Compute REF/RREF and report rank and solution details.

	Returns a dictionary with matrices and metadata. When show=True, prints a
	formatted report.
	"""

	mat_a = _to_matrix(A)
	vec_b: Optional[sp.Matrix] = None
	ref_a = mat_a.echelon_form()
	rref_a, pivots_a = mat_a.rref()
	rank_a = mat_a.rank()

	lu_data: Optional[Tuple[sp.Matrix, sp.Matrix, sp.Matrix]] = None
	if mat_a.rows == mat_a.cols:
		try:
			l_mat, u_mat, perm = mat_a.LUdecomposition()
			p_mat = _perm_to_matrix(perm, mat_a.rows)
			lu_data = (p_mat, l_mat, u_mat)
		except Exception:
			lu_data = None

	details: RefRrefDetails = {
		"A": mat_a,
		"REF": ref_a,
		"RREF": rref_a,
		"rank(A)": rank_a,
		"pivots": pivots_a,
		"PA_LU": lu_data,
	}

	if b is not None:
		vec_b = _to_column_vector(b, mat_a.rows)
		aug = mat_a.row_join(vec_b)
		rref_aug, pivots_aug = aug.rref()
		rank_aug = aug.rank()

		if rank_a != rank_aug:
			solution_type = "no solution"
			dim_solution = None
		elif rank_a == mat_a.cols:
			solution_type = "unique"
			dim_solution = 0
		else:
			solution_type = "infinite"
			dim_solution = mat_a.cols - rank_a

		rref_aug_display = rref_aug
		if stop_before_aug_pivot is None and solution_type == "no solution":
			stop_before_aug_pivot = True
		if stop_before_aug_pivot:
			ref_aug = aug
			lu_steps_aug = _lu_steps_augmented(mat_a, vec_b)
			if lu_steps_aug:
				ref_aug = lu_steps_aug[-1][3]
			rref_aug_display = _rref_from_ref(ref_aug, mat_a.cols)

		details.update(
			{
				"b": vec_b,
				"[A|b]": aug,
				"RREF([A|b])": rref_aug_display,
				"rank([A|b])": rank_aug,
				"solution_type": solution_type,
				"solution_dimension": dim_solution,
				"aug_pivots": pivots_aug,
			}
		)

	if show:
		print(f"{symbol} =")
		sp.pprint(mat_a)
		print("\nREF(A) =")
		sp.pprint(ref_a)

		if show_steps:
			print("\nREF steps (LU style):")
			lu_steps = _lu_steps_augmented(
				mat_a,
				vec_b if b is not None else None,
			)
			for idx, (notes, p_mat, l_mat, u_mat) in enumerate(lu_steps, start=1):
				print(f"Step {idx}:")
				for note in notes:
					print(note)
				if p_mat is not None:
					print("P =")
					sp.pprint(p_mat)
				print("L =")
				sp.pprint(l_mat)
				print("U =")
				sp.pprint(u_mat)

		if not show_steps:
			if lu_data is not None:
				p_mat, l_mat, u_mat = lu_data
				print("\nPA = LU (REF shown as U):")
				print("P =")
				sp.pprint(p_mat)
				print("L =")
				sp.pprint(l_mat)
				print("U =")
				sp.pprint(u_mat)
			else:
				print("\nPA = LU not available (matrix not square or LU failed).")

		print("\nRREF(A) =")
		sp.pprint(rref_a)
		print(f"\nrank(A) = {rank_a}")

		if show_steps:
			print("\nRREF steps:")
			base_mat = mat_a
			if b is not None:
				base_mat = (
					details["RREF([A|b])"]
					if stop_before_aug_pivot
					else mat_a.row_join(vec_b)
				)
			if "lu_steps" in locals() and lu_steps and not stop_before_aug_pivot:
				base_mat = lu_steps[-1][3]
			step_pivot_cols = mat_a.cols if stop_before_aug_pivot else None
			for op, step_mat in _rref_steps_from_ref(
				base_mat,
				pivot_cols=step_pivot_cols,
			):
				print(op)
				sp.pprint(step_mat)

		if b is not None:
			print("\n[A|b] =")
			sp.pprint(aug)
			print("\nRREF([A|b]) =")
			sp.pprint(details["RREF([A|b])"])
			if stop_before_aug_pivot:
				print(
					"Note: augmented column stops before an augmented pivot "
					"for an inconsistent system."
				)
			print(f"\nrank([A|b]) = {details['rank([A|b])']}")
			print(f"solution type = {details['solution_type']}")
			print(
				"dimension of solution set = "
				f"{details['solution_dimension']}"
			)

	return details


__all__ = ["matrix_ref_rref_details"]
