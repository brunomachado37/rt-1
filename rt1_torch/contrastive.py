import torch
import torch.nn as nn
import lightning as L


class ActionContrativeTraining(L.LightningModule):
    def __init__(self,  
                 model: nn.Module, 
                 learning_rate: float = 1e-3, 
                 weight_decay: float = 0.01,
                 temperature: float = 0.07,
                 steps: int = 200_000,
                ):
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logit_scale = 1.0 / temperature
        self.steps = steps

    def training_step(self, batch, batch_idx):
        """
        batch is composed of:
        images: torch.Tensor - A batch of images of shape (B, T, C, H, W).
        language_embeddings: torch.Tensor - A batch of language embeddings of shape (B, L_D).
        actions: torch.Tensor - A batch of actions of shape (B, T, A_D). 
        """   
        actions = batch.pop("actions")          # (bs, T, action_dim)

        output = self.model(**batch)            # (bs, T, token_per_image, embedding_dim)

        # Take image representations (1 token per image)
        image_logits = output.squeeze((1, 2))   # (bs, embedding_dim)

        # Compute cosine similarity between all actions
        actions = actions.squeeze(1)            # (bs, action_dim)
        norm_actions = actions / actions.norm(dim=-1, keepdim=True)
        action_sim = norm_actions @ norm_actions.T
        action_sim = (action_sim + 1.0) / 2.0   # Shift the interval from [-1 and 1] to [0 and 1]

        self.log("action_sim_mean", action_sim.mean())
        self.log("action_sim_max", action_sim.max())
        self.log("action_sim_min", action_sim.min())
        
        # Compute cosine similarity between all image representations
        norm_image_logits = image_logits / image_logits.norm(dim=-1, keepdim=True)
        image_sim = self.logit_scale * norm_image_logits @ norm_image_logits.T

        # Minimize similarity between images based on action similarity
        loss = torch.nn.functional.cross_entropy(image_sim, action_sim)

        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]


class CameraContrativeTraining(L.LightningModule):
    def __init__(self,  
                 model: nn.Module, 
                 learning_rate: float = 1e-3, 
                 weight_decay: float = 0.01,
                 temperature: float = 0.07,
                 steps: int = 200_000,
                ):
        super().__init__()
        self.save_hyperparameters(ignore="model")

        self.model = model
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.logit_scale = 1.0 / temperature
        self.steps = steps

    def training_step(self, batch, batch_idx):
        """
        batch is composed of:
        view_1: torch.Tensor - A batch of images of shape (B, T, C, H, W) from camera 1.
        view_2: torch.Tensor - A batch of images of shape (B, T, C, H, W) from camera 2.
        language_embeddings: torch.Tensor - A batch of language embeddings of shape (B, L_D).
        actions: torch.Tensor - A batch of actions of shape (B, T, A_D). 
        """   
        actions = batch.pop("actions")          # (bs, T, action_dim)
        view_2 = batch.pop("view_2")            
        batch["images"] = batch.pop("view_1")

        output_1 = self.model(**batch)          # (bs, T, token_per_image, embedding_dim)
        
        batch["images"] = view_2
        output_2 = self.model(**batch)          # (bs, T, token_per_image, embedding_dim)

        # Take image representations (1 token per image)
        image_logits_1 = output_1.squeeze((1, 2))   # (bs, embedding_dim)
        image_logits_2 = output_2.squeeze((1, 2))   # (bs, embedding_dim)
        
        # Compute cosine similarity between all image representations
        norm_image_logits_1 = image_logits_1 / image_logits_1.norm(dim=-1, keepdim=True)
        norm_image_logits_2 = image_logits_2 / image_logits_2.norm(dim=-1, keepdim=True)

        image_sim = self.logit_scale * norm_image_logits_1 @ norm_image_logits_2.T

        labels = torch.arange(image_sim.size(0), device=image_sim.device)
        loss = torch.nn.functional.cross_entropy(image_sim, labels)

        self.log("train_loss", loss)
        return loss
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=self.steps)

        return [optimizer], [{'scheduler': scheduler, 'interval': 'step', 'frequency': 1}]