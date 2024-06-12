import torch

def linear(
	a: torch.Tensor,	# [a, b]
	w: torch.Tensor		# [c, b]
) -> torch.Tensor:		# [a, c]
	# pylint: disable=not-callable
	# NOTE. It seems that torch.nn.functional.linear automatically select
	# the best implementation for the given input shapes (GEMM or GEMV) while
	# torch.matmul always uses GEMM. So, we use torch.nn.functional.linear here
	# to get the best performance.
	return torch.nn.functional.linear(a, w)
