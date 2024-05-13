import torch

def linear(
	a: torch.Tensor,	# [a, b]
	w: torch.Tensor		# [c, b]
) -> torch.Tensor:		# [a, c]
	# pylint: disable=not-callable
	return torch.nn.functional.linear(a, w)
