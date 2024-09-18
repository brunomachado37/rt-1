import torch
import hydra

from torch.utils.data import DataLoader, random_split
from lightning import Trainer
from lightning.pytorch.loggers import WandbLogger
from lightning.pytorch.callbacks import LearningRateMonitor, ModelCheckpoint
from lightning.pytorch import seed_everything
from types import SimpleNamespace

from rt1_torch.rt1 import RT1, BCTraining
from data.robocasa.train_utils import load_data_for_training
from data.robocasa.lang_utils import language_encoder_factory


@hydra.main(version_base=None, config_path="conf", config_name="config_robocasa")
def train(config):
    seed_everything(42, workers=True)

    model = RT1(**config.model)
    
    data_config = SimpleNamespace(**config.data)
    data_config.image_transform = model.image_tokenizer.vision_model_weights.transforms()
    lang_encoder = language_encoder_factory(model=data_config.language_encoder, device="cuda")
    training_set, validation_set = load_data_for_training(data_config, obs_keys=data_config.all_obs_keys, lang_encoder=lang_encoder)

    max_num_steps = (len(training_set) // config.dataloader.batch_size) * config.trainer.max_epochs
    lightning_model = BCTraining(model, steps=max_num_steps, **config.training)

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