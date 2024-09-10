"""
Film-Conditioning PyTorch implementation adapted from the Jax implementation: https://github.com/google-deepmind/open_x_embodiment/blob/main/models/film_conditioning.py

"""

import torch
import torch.nn as nn


class FilmConditioning(nn.Module):
    """FiLM conditioning layer."""

    def __init__(self, num_channels: int, context_size: int):
        super().__init__()
        self.film_linear_add = nn.Linear(context_size, num_channels)
        self.film_linear_mul = nn.Linear(context_size, num_channels)

        self.apply(self.weights_init)
    
    def weights_init(self, m):
        if isinstance(m, nn.Linear):
            nn.init.zeros_(m.weight.data)
            nn.init.zeros_(m.bias.data)

    def forward(self, conv_filters, context):
        """Applies FiLM conditioning to the input.

        Args:
          conv_filters: array of shape (B, C, H, W), usually an output conv feature
            map.
          context: array of shape (B, context_size).

        Returns:
          array of shape (B, C, H, W) with the FiLM conditioning applied.
        """
        project_cond_add = self.film_linear_add(context)
        project_cond_mul = self.film_linear_mul(context)

        project_cond_add = project_cond_add[:, None, None, :]
        project_cond_mul = project_cond_mul[:, None, None, :]

        conv_filters = torch.permute(conv_filters, (0, 2, 3, 1))                        # (B, C, H, W) -> (B, H, W, C)

        result = (1 + project_cond_mul) * conv_filters + project_cond_add

        result = torch.permute(result, (0, 3, 1, 2))                                    # (B, H, W, C) -> (B, C, H, W)

        return result               