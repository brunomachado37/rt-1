import torch
import torch.nn as nn
import lightning as L

from typing import Callable, Optional, Literal
from torchvision.models import WeightsEnum

import rt1_torch.efficientnet as efficientnet
from rt1_torch.image_tokenizer import ImageTokenizer


class RT1(nn.Module):
    def __init__(self,
                 num_layers: int = 8,
                 token_dim: int = 512,
                 num_heads: int = 8,
                 dropout_rate: float = 0.1,
                 attention_mode: Literal["causal", "timestep_causal"] = "timestep_causal",
                 vision_model: str = "efficientnet_b3",
                 vision_model_weights: str = "EfficientNet_B3_Weights",
                 language_embedding_dim: int = 768,
                 num_tokens_per_image: int = 8,
                 action_space: Literal["discrete", "continuous"] = "continuous",
                 action_dim: int = 1,
                 action_readout_tokens: int = 8):
        """
        Arguments:
        num_layers: int - Number of transformer layers.
        token_dim: int - Dimension of the tokens.
        num_heads: int - Number of attention heads.
        dropout_rate: float - Dropout rate.
        attention_mode: Literal["causal", "timestep_causal"] - The mode of the attention mask.
            If causal, a traditional causal mask is used, where a token can only attend to itself and all previous tokens in the sequence.
                RT-1 uses this mode, so not all tokens coming from the same image can't attend to each other.
            If timestep_causal, a mask is used where a token can attend to all tokens in the current and previous timesteps, but not to future tokens.
        vision_model: Callable - A callable that returns a vision model.
        vision_model_weights: WeightsEnum - The weights to be used for the vision model.
        language_embedding_dim: int - Dimension of the language embeddings.
        num_tokens_per_image: int - Number of tokens to be outputted by the image tokenizer.
        action_space: Literal["discrete", "continuous"] - The mode of the action space.
            If discrete, the input actions are spected to be discrete and the action prediction head will be a classification head.
            If continuous, the input actions are spected to be continuous and the action prediction head will be a regression head.
        action_dim: int - Output dimension of the action prediction head.
            If using action_space == "discrete", action_dim should be the number of discretization bins (e.g. for RT-1, action_dim = 256 | number of "classes").
            If using action_space == "continuous", action_dim should be equal to the dimension of the action space when action_readout_tokens == 1 and
            should be 1 when using action_readout_tokens equal to the dimension of the action space. Note that other combinations are possible.
        action_readout_tokens: int - Number of tokens added to the model input for the action prediction.
            If using action_space == "discrete", action_readout_tokens should be equal to the dimension of the action space (e.g. in RT-1, action_readout_tokens = 11).
            If using action_space == "continuous", you can either add a single readout token for all action dimensions (e.g. action_readout_tokens = 1 and action_dim = 11), 
            or add a readout token for each action dimension (e.g. action_readout_tokens = 11 and action_dim = 1). Note that other combinations are possible.
        """
        super().__init__()

        self.action_readout_tokens = action_readout_tokens
        self.attention_mode = attention_mode
        self.action_space = action_space
        
        self.image_tokenizer = ImageTokenizer(
                embedding_output_dim=token_dim, 
                vision_model_factory=getattr(efficientnet, vision_model),
                vision_model_weights=getattr(efficientnet, vision_model_weights).DEFAULT,
                use_film=True,
                film_context_size=language_embedding_dim,
                use_token_learner=True, 
                num_output_tokens=num_tokens_per_image
        )
                
        self.transformer = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=token_dim,
                    nhead=num_heads,
                    dim_feedforward=4*token_dim,
                    dropout=dropout_rate,
                    activation='gelu',
                    batch_first=True,
                    norm_first=True,
                ),
                num_layers=num_layers,
                norm=nn.LayerNorm(token_dim)
        )

        self.action_prediction_head = nn.Linear(token_dim, action_dim)


    def _create_mask(self, tokens_per_timestep: int, sequence_length: int, action_indexes: torch.Tensor, dtype: torch.dtype = torch.float32):
        """
        Create a causal mask in which a token can attend to all tokens in the current and previous timesteps
        This differs from the original implementation, where a traditional causal mask is used, but for the action tokens, which can attend to all previous image tokens.
        Arguments:
        tokens_per_timestep: int - Number of tokens per timestep
        sequence_length: int - Sequence length
        action_indexes: torch.Tensor - Indexes of the readout action tokens, which shouldn't be attended to
        dtype: torch.dtype - Data type of the mask (to allow mixed precision training)
        """
        mask = torch.full((sequence_length, sequence_length), float("-inf"), dtype=dtype)
        
        for i in range(0, sequence_length, tokens_per_timestep):
            end = min(sequence_length, i + tokens_per_timestep)
            mask[i:end, :end] = 0

        mask[:, action_indexes] = float("-inf")

        return mask
    

    def _create_causal_mask(self, sequence_length: int, action_indexes: torch.Tensor, dtype: torch.dtype = torch.float32):
        """
        Create a traditional causal mask, where a token can only attend to itself and all previous tokens in the sequence.
        Ignore all the action tokens.
        Arguments:
        sequence_length: int - Sequence length
        action_indexes: torch.Tensor - Indexes of the readout action tokens, which shouldn't be attended to
        dtype: torch.dtype - Data type of the mask (to allow mixed precision training)
        """
        mask = torch.triu(torch.full((sequence_length, sequence_length), float("-inf"), dtype=dtype), diagonal=1)
        mask[:, action_indexes] = float("-inf")

        return mask


    def forward(self, 
                images: torch.Tensor,
                language_embeddings: Optional[torch.Tensor] = None,
                ):
        """
        Arguments:
        images: torch.Tensor - A batch of images of shape (B, T, C, H, W).
        language_embeddings: torch.Tensor - A batch of language embeddings of shape (B, L_D).
        """       
        bs, t, c, h, w = images.shape
        image_tokens = self.image_tokenizer(images, language_embeddings)                                                # (B, T, num_output_tokens, token_dim)

        # Create a sequence of dummy action tokens, adding action_readout_tokens for each action
        action_tokens = torch.zeros(bs, t, self.action_readout_tokens, image_tokens.shape[-1]).to(images.device)        # (B, T, self.action_readout_tokens, token_dim)

        # Add it to the image tokens sequence to create the input sequence for the transformer
        input_tokens = torch.cat([image_tokens, action_tokens], dim=2)                                                  # (B, T, num_output_tokens + 1, token_dim)            
        input_sequences = torch.reshape(input_tokens, (bs, -1, image_tokens.shape[-1]))                                 # (B, T * (num_output_tokens + 1), token_dim)

        # The attention mask assumes that the action indexes are the same for all the batches, which should be the case
        # TODO: Compare it with the Jax Open-X implementation (print both masks and compare)
        action_indexes = (input_sequences[0] == 0).all(-1).nonzero(as_tuple=True)[0]

        mask = self._create_causal_mask(input_sequences.shape[1], action_indexes, images.dtype).to(images.device) if self.attention_mode == "causal" \
               else self._create_mask(input_tokens.shape[2], input_sequences.shape[1], action_indexes, images.dtype).to(images.device) 

        output_embeddings = self.transformer(input_sequences, mask=mask)
        action_embeddings = output_embeddings[:, action_indexes, :]
        action_logits = self.action_prediction_head(action_embeddings)

        return {"action_logits": action_logits, "logits": output_embeddings}
    

class BCTraining(L.LightningModule):
    def __init__(self,  
                 model: nn.Module, 
                 learning_rate: float = 1e-3, 
                 weight_decay: float = 0.01,
                 epochs: int = 30,
                ):
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.epochs = epochs

    def training_step(self, batch, batch_idx):
        """
        batch is composed of:
        images: torch.Tensor - A batch of images of shape (B, T, C, H, W).
        language_embeddings: torch.Tensor - A batch of language embeddings of shape (B, L_D).
        actions: torch.Tensor - A batch of actions of shape (B, T, A_D). 
            The action dimension A_D represents the dimension of each action. 
                When using action_space == "continuous":
                    - A_D should match action_dim * action_readout_tokens.
                    - Actions should be floats between -1 and 1.
                When using action_space == "discrete":
                    - A_D should match action_readout_tokens.
                    - Actions should be integers between 0 and action_dim - 1. 
        """   
        bs = batch["images"].shape[0]
        actions = batch.pop("actions")

        output = self.model(**batch)
        action_logits = output["action_logits"]

        if self.model.action_space == "continuous":
            action_preds = nn.functional.tanh(action_logits)
            loss = nn.functional.l1_loss(action_preds.view(bs, -1), actions.view(bs, -1))

        elif self.model.action_space == "discrete":
            loss = nn.functional.cross_entropy(action_logits.view(-1, action_logits.shape[-1]), actions.view(-1))

        else:
            raise ValueError("Invalid action space")

        self.log("train_loss", loss)
        return loss
    
    def validation_step(self, batch, batch_idx): 
        bs = batch["images"].shape[0]
        actions = batch.pop("actions")

        output = self.model(**batch)
        action_logits = output["action_logits"]

        if self.model.action_space == "continuous":
            action_preds = nn.functional.tanh(action_logits)
            loss = nn.functional.l1_loss(action_preds.view(bs, -1), actions.view(bs, -1))

        elif self.model.action_space == "discrete":
            loss = nn.functional.cross_entropy(action_logits.view(-1, action_logits.shape[-1]), actions.view(-1))

        else:
            raise ValueError("Invalid action space")

        self.log("val_loss", loss)
        return loss

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.epochs)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]