"""
Token Learner PyTorch implementation adapted from the Jax implementation: https://github.com/google-deepmind/open_x_embodiment/blob/main/models/token_learner.py

"""

from typing import Callable

import torch
import torch.nn as nn


class MlpBlock(nn.Module):
    """Transformer MLP / feed-forward block."""

    def __init__(self,
                 in_dim: int,
                 mlp_dim: int,
                 out_dim: int,
                 dropout_rate: float = 0.1,
                 use_bias: bool = True,
                 activation_fn: Callable[[torch.tensor], torch.tensor] = nn.GELU,
                 dtype: torch.dtype = torch.float32):
        super().__init__()

        self.linear_1 = nn.Linear(in_dim, mlp_dim, bias=use_bias, dtype=dtype)
        self.activation_1 = activation_fn()
        self.dropout_1 = nn.Dropout(dropout_rate)

        self.linear_2 = nn.Linear(mlp_dim, out_dim, bias=use_bias, dtype=dtype)
        self.dropout_2 = nn.Dropout(dropout_rate)


    def __call__(self, inputs: torch.tensor, **kwargs) -> torch.tensor:
        """Applies Transformer MlpBlock module."""
        x = self.linear_1(inputs)
        x = self.activation_1(x)
        x = self.dropout_1(x)

        output = self.linear_2(x)
        output = self.dropout_2(output)

        return output



class TokenLearnerModuleV11(nn.Module):
    """TokenLearner module Version 1.1, using slightly different conv. layers.

    Instead of using 4 conv. layers with small channels to implement spatial
    attention, this version uses a MLP with gelu inbetween. It also uses softmax
    instead of sigmoid. We confirmed that this version works better in general.

    Attributes:
    num_tokens: Number of tokens.
    bottleneck_dim: The size of hidden units in the MLP for spatial attention.
    dropout_rate: Dropout rate.
    """
    def __init__(self,
                in_tokens_dim: int,
                num_out_tokens: int,
                bottleneck_dim: int = 64,
                dropout_rate: float = 0.0):
        super().__init__()
        self.num_tokens = num_out_tokens

        self.layer_norm = nn.LayerNorm(in_tokens_dim)
        self.mlp = MlpBlock(in_tokens_dim, bottleneck_dim, num_out_tokens, dropout_rate, activation_fn=nn.GELU)

    def __call__(self, inputs: torch.tensor, **kwargs) -> torch.tensor:
        """Applies learnable tokenization to the 2D inputs.

        Args:
            inputs: Inputs of shape `[bs, h, w, c]`.
            deterministic: Weather we are in the deterministic mode (e.g inference
            time) or not.

        Returns:
            Output of shape `[bs, n_token, c]`.
        """
        if inputs.ndim == 4:
            n, h, w, c = inputs.shape
            inputs = torch.reshape(inputs, [n, h * w, c])

        feature_shape = inputs.shape
        selected = inputs

        selected = self.layer_norm(selected)
        selected = self.mlp(selected)

        selected = torch.reshape(selected, [feature_shape[0], -1, self.num_tokens])         # Shape: [bs, h*w, n_token]
        selected = torch.transpose(selected, 2, 1)                                          # Shape: [bs, n_token, h*w]
        selected = torch.nn.functional.softmax(selected, dim=-1)

        feat = inputs
        feat = torch.reshape(feat, [feature_shape[0], -1, feature_shape[-1]])               # Shape: [bs, h*w, c]

        feat = torch.einsum('...si,...id->...sd', selected, feat)

        return feat