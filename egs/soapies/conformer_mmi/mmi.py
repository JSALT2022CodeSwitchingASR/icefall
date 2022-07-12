
import matgraph as mg
import torch
from torch import nn

from typing import List

class SeqLogProb(torch.autograd.Function):
    """Compute the log-marginal probabily of a sequence given a graph."""

    @staticmethod
    def forward(ctx, input, seqlengths, fsm) -> torch.Tensor:
        """
        Args:
          input:
            Input to the loss.
          seglengths:
            Length of each sequence of the batch.
          fsm:
            FSM to use for the marginalization.

        Returns:
          Return a scalar.
        """
        posts, logprob = mg.pdfposteriors(fsm, input, seqlengths.numpy())
        ctx.save_for_backward(posts)
        return sum(logprob)

    @staticmethod
    def backward(ctx, f_grad):
        input_grad, = ctx.saved_tensors
        return torch.mul(input_grad, f_grad), None, None


class LFMMILoss(nn.Module):

    def __init__(self, denfsms, numfsms, den_scale=1.0):
        super().__init__()
        self.denfsms = denfsms
        self.numfsms = numfsms
        self.den_scale = den_scale

    def forward(self, input, seqlengths):
        num_llh = SeqLogProb.apply(input, seqlengths, self.numfsms)
        den_llh = SeqLogProb.apply(input, seqlengths, self.denfsms)
        return -(num_llh - self.den_scale * den_llh)

