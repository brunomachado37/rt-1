"""
materialize.py

Factory class for initializing Open-X RLDS-backed datasets, given specified data mixture parameters; provides and
exports individual functions for clear control flow.
"""

from pathlib import Path
from typing import Tuple
from types import SimpleNamespace

import json
from torch.utils.data import Dataset
from transformers import PreTrainedTokenizerBase

from data.tfds import EpisodicRLDSDataset, RLDSBatchTransform, RLDSDataset, ImageTransform, PaddedCollatorForActionPrediction
from data.robocasa.train_utils import load_data_for_training


def get_dataset_and_collator_tfds(
    data_root_dir: Path,
    data_mix: str,
    image_transform: ImageTransform,
    tokenizer: PreTrainedTokenizerBase,
    default_image_resolution: Tuple[int, int, int],
    padding_side: str = "right",
    predict_stop_token: bool = True,
    shuffle_buffer_size: int = 100_000,
    train: bool = True,
    episodic: bool = False,
    image_aug: bool = False,
    window_size: int = 1,
) -> Tuple[Dataset, PaddedCollatorForActionPrediction]:
    """Initialize RLDS Dataset (wraps TFDS) and initialize transform/collation functions."""
    batch_transform = RLDSBatchTransform(
        tokenizer, image_transform, predict_stop_token=predict_stop_token
    )
    collator = PaddedCollatorForActionPrediction(
        tokenizer.model_max_length, tokenizer.pad_token_id, padding_side=padding_side
    )

    # Build RLDS Iterable Dataset
    cls = RLDSDataset if not episodic else EpisodicRLDSDataset
    dataset = cls(
        data_root_dir,
        data_mix,
        batch_transform,
        resize_resolution=default_image_resolution[1:],
        shuffle_buffer_size=shuffle_buffer_size,
        train=train,
        image_aug=image_aug,
        window_size=window_size,
    )

    return dataset, collator
