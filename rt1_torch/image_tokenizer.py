import torch
from torch import nn, Tensor

from torchvision.models import WeightsEnum
from typing import Callable

from rt1_torch.efficientnet import efficientnet_b3, EfficientNet_B3_Weights
from rt1_torch.film_conditioning import FilmConditioning
from rt1_torch.token_learner import TokenLearnerModuleV11


class ImageTokenizer(nn.Module):
    """
    Image Tokenizer module that takes a batch of image sequences and returns a a batch of token sequences.
    Arguments: 
    embedding_output_dim: int - The output dimension of the image tokenizer. It will be dthe dimension of the token embeddings.
    vision_model_factory: Callable - A callable that returns a vision model.
    vision_model_weights: WeightsEnum - The weights to be used for the vision model.
    use_film: bool - Whether to use FiLM conditioning.
    film_context_size: int - The size of the FiLM context arrays. Only used if use_film is True.
    use_token_learner: bool - Whether to use the Token Learner module to reduce the number of tokens.
    num_output_tokens: int - The number of tokens per output sequence. Only used if use_token_learner is True.
    """
    def __init__(self, 
                 embedding_output_dim: int, 
                 vision_model_factory: Callable = efficientnet_b3,
                 vision_model_weights: WeightsEnum = EfficientNet_B3_Weights.DEFAULT,
                 use_film: bool = True,
                 film_context_size: int = 768,
                 use_token_learner: bool = True, 
                 num_output_tokens: int = 8):
        super().__init__()
        self.use_film = use_film
        self.use_token_learner = use_token_learner
        self.vision_model_weights = vision_model_weights

        self.vision_model = vision_model_factory(weights=vision_model_weights, 
                                                 num_classes=None,
                                                 film_context_size=film_context_size if use_film else None)
        
        self.conv1x1 = nn.Conv2d(self.vision_model.lastconv_output_channels,
                                 embedding_output_dim,
                                 kernel_size=1,
                                 padding='same',
                                 bias=False)

        if use_film:
            self.final_film = FilmConditioning(embedding_output_dim, context_size=film_context_size)

        if use_token_learner:
            self.token_learner = TokenLearnerModuleV11(embedding_output_dim, num_output_tokens)


    def forward(self, images: Tensor, context: Tensor = None):
        B, T, C, H, W = images.shape
        images = torch.reshape(images, [B * T, C, H, W])                     # Fold time into batch dimension
        context = torch.reshape(context, [B * T, -1])

        if self.use_film:
            last_feature_maps = self.vision_model(images, context)
            features = self.conv1x1(last_feature_maps)
            features = self.final_film(features, context)
        else:
            last_feature_maps = self.vision_model(images)
            features = self.conv1x1(last_feature_maps)

        tokens = torch.permute(features, (0, 2, 3, 1))                       # (B*T, C, H, W) -> (B*T, H, W, C)

        if self.use_token_learner:
            tokens = self.token_learner(tokens)
        else:
            n, h, w, c = tokens.shape
            tokens = torch.reshape(tokens, (n, h * w, c))

        tokens = torch.reshape(tokens, (B, T, tokens.shape[1], -1))          # Unfold time from batch dimension

        return tokens