import torch
import hydra

from torch.utils.data import DataLoader, random_split
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import seed_everything

from rt1_torch.image_tokenizer import ImageTokenizer
from rt1_torch.contrastive import CameraContrativeTraining
import rt1_torch.efficientnet as efficientnet

from data.robocasa.dataset import MultiViewContrastiveBatchTransform
from data.robocasa.train_utils import load_data_for_training
from data.robocasa.lang_utils import language_encoder_factory


@hydra.main(version_base=None, config_path="conf/camera_contrastive", config_name="01_config")
def train(config):
    seed_everything(42, workers=True)

    model_config = dict(config.model)
    backbone = getattr(efficientnet, model_config.pop('vision_model'))
    pretrained_weights = getattr(efficientnet, model_config.pop('vision_model_weights')).DEFAULT
    model = ImageTokenizer(**model_config, 
                           vision_model_factory=backbone, 
                           vision_model_weights=pretrained_weights)
    
    batch_transform = MultiViewContrastiveBatchTransform(model.vision_model_weights.transforms())
    lang_encoder = language_encoder_factory(model=config.data.language_encoder, device="cuda")
    training_set, validation_set = load_data_for_training(config.data, 
                                                          obs_keys=config.data.all_obs_keys, 
                                                          lang_encoder=lang_encoder, 
                                                          batch_transform=batch_transform)   

    max_num_steps = (len(training_set) // config.dataloader.batch_size) * config.trainer.max_epochs
    lightning_model = CameraContrativeTraining(model, steps=max_num_steps, **config.training)

    # Release memory used by language encoder
    print(f"Torch CUDA memory allocated: {torch.cuda.memory_allocated()/1_000_000:.0f} Mb")
    if hasattr(training_set, "lang_encoder"):
        del training_set.lang_encoder
    else:
        for i in range(len(training_set.datasets)):
            del training_set.datasets[i].lang_encoder 
    del lang_encoder
    print(f"Torch CUDA memory allocated: {torch.cuda.memory_allocated()/1_000_000:.0f} Mb\n")

    if config.data.validate:
        validation_set_size = int(len(training_set) * config.dataloader.validation_percentage)
        training_set_size = len(training_set) - validation_set_size

        training_set, validation_set = random_split(training_set, [training_set_size, validation_set_size])

        val_dataloader = DataLoader(validation_set, 
                   batch_size=config.dataloader.batch_size, 
                   num_workers=config.dataloader.num_workers,
                   collate_fn=None, 
                   shuffle=True, 
                   drop_last=True)

    train_dataloader = DataLoader(training_set, 
                   batch_size=config.dataloader.batch_size, 
                   num_workers=config.dataloader.num_workers,
                   collate_fn=None, 
                   shuffle=True, 
                   drop_last=True)

    logger = WandbLogger(**config.logger)
    lr_monitor = LearningRateMonitor(logging_interval='step')
    checkpoint_callback = ModelCheckpoint(every_n_train_steps=50_000)

    trainer = Trainer(logger=logger, callbacks=[lr_monitor, checkpoint_callback], **config.trainer)
    trainer.fit(lightning_model, train_dataloader, val_dataloader) if config.data.validate else trainer.fit(lightning_model, train_dataloader)

if __name__ == "__main__":
    train()