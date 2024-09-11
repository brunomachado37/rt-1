"""
This file contains several utility functions used to define the main training loop. It 
mainly consists of functions to assist with logging, rollouts, and the @run_epoch function,
which is the core training logic for models in this repository.
"""
import os
import h5py
import numpy as np
from copy import deepcopy

from data.robocasa.dataset import InterleavedDataset, R2D2Dataset, MetaDataset


def load_data_for_training(config, obs_keys, lang_encoder=None):
    """
    Data loading at the start of an algorithm.

    Args:
        config (BaseConfig instance): config object
        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

    Returns:
        train_dataset (SequenceDataset instance): train dataset object
        valid_dataset (SequenceDataset instance): valid dataset object (only if using validation)
    """

    # config can contain an attribute to filter on
    train_filter_by_attribute = config.hdf5_filter_key
    valid_filter_by_attribute = config.hdf5_validation_filter_key
    if valid_filter_by_attribute is not None:
        assert config.validate, "specified validation filter key {}, but configexperiment.validate is not set".format(valid_filter_by_attribute)

    # load the dataset into memory
    if config.validate:
        assert not config.hdf5_normalize_obs, "no support for observation normalization with validation data yet"
        assert (train_filter_by_attribute is not None) and (valid_filter_by_attribute is not None), \
            "did not specify filter keys corresponding to train and valid split in dataset" \
            " - please fill config.hdf5_filter_key and config.hdf5_validation_filter_key"
        # train_demo_keys = _get_demos_for_filter_key(
        #     hdf5_path=os.path.expanduser(config.data),
        #     filter_key=train_filter_by_attribute,
        # )
        # valid_demo_keys = _get_demos_for_filter_key(
        #     hdf5_path=os.path.expanduser(config.data),
        #     filter_key=valid_filter_by_attribute,
        # )
        # assert set(train_demo_keys).isdisjoint(set(valid_demo_keys)), "training demonstrations overlap with " \
        #     "validation demonstrations!"
        train_dataset = dataset_factory(
            config, obs_keys,
            filter_by_attribute=train_filter_by_attribute,
            lang_encoder=lang_encoder,
        )
        valid_dataset = dataset_factory(
            config, obs_keys,
            filter_by_attribute=valid_filter_by_attribute,
            lang_encoder=lang_encoder,
        )
    else:
        train_dataset = dataset_factory(
            config, obs_keys,
            filter_by_attribute=train_filter_by_attribute,
            lang_encoder=lang_encoder,
        )
        valid_dataset = None

    return train_dataset, valid_dataset


def dataset_factory(config, obs_keys, filter_by_attribute=None, dataset_path=None, lang_encoder=None):
    """
    Create a SequenceDataset instance to pass to a torch DataLoader.

    Args:
        config (BaseConfig instance): config object

        obs_keys (list): list of observation modalities that are required for
            training (this will inform the dataloader on what modalities to load)

        filter_by_attribute (str): if provided, use the provided filter key
            to select a subset of demonstration trajectories to load

        dataset_path (str): if provided, the SequenceDataset instance should load
            data from this dataset path. Defaults to config.data.

    Returns:
        dataset (SequenceDataset instance): dataset object
    """
    if dataset_path is None:
        dataset_path = config.data

    ds_kwargs = dict(
        hdf5_path=dataset_path,
        obs_keys=obs_keys,
        action_keys=config.action_keys,
        dataset_keys=config.dataset_keys,
        action_config=config.action_config,
        load_next_obs=config.hdf5_load_next_obs, # whether to load next observations (s') from dataset
        frame_stack=config.frame_stack,
        seq_length=config.seq_length,
        pad_frame_stack=config.pad_frame_stack,
        pad_seq_length=config.pad_seq_length,
        get_pad_mask=True,
        goal_mode=config.goal_mode,
        hdf5_cache_mode=config.hdf5_cache_mode,
        hdf5_use_swmr=config.hdf5_use_swmr,
        hdf5_normalize_obs=config.hdf5_normalize_obs,
        filter_by_attribute=filter_by_attribute,
        shuffled_obs_key_groups=config.shuffled_obs_key_groups,
        lang_encoder=lang_encoder,
        image_transform=config.image_transform,
    )

    ds_kwargs["hdf5_path"] = [ds_cfg["path"] for ds_cfg in config.data]
    ds_kwargs["filter_by_attribute"] = [ds_cfg.get("filter_key", filter_by_attribute) for ds_cfg in config.data]
    ds_weights = [ds_cfg.get("weight", 1.0) for ds_cfg in config.data]
    ds_langs = [ds_cfg.get("lang", None) for ds_cfg in config.data]

    meta_ds_kwargs = dict()

    dataset = get_dataset(
        ds_class=R2D2Dataset if config.data_format == "r2d2" else InterleavedDataset,
        ds_kwargs=ds_kwargs,
        ds_weights=ds_weights,
        ds_langs=ds_langs,
        normalize_weights_by_ds_size=False,
        meta_ds_class=MetaDataset,
        meta_ds_kwargs=meta_ds_kwargs,
    )

    return dataset


def get_dataset(
    ds_class,
    ds_kwargs,
    ds_weights,
    ds_langs,
    normalize_weights_by_ds_size,
    meta_ds_class=MetaDataset,
    meta_ds_kwargs=None,
):
    ds_list = []
    for i in range(len(ds_weights)):
        
        ds_kwargs_copy = deepcopy(ds_kwargs)

        keys = ["hdf5_path", "filter_by_attribute"]

        for k in keys:
            ds_kwargs_copy[k] = ds_kwargs[k][i]

        ds_kwargs_copy["dataset_lang"] = ds_langs[i]
        
        ds_list.append(ds_class(**ds_kwargs_copy))
    
    if len(ds_weights) == 1:
        ds = ds_list[0]
    else:
        if meta_ds_kwargs is None:
            meta_ds_kwargs = dict()
        ds = meta_ds_class(
            datasets=ds_list,
            ds_weights=ds_weights,
            normalize_weights_by_ds_size=normalize_weights_by_ds_size,
            **meta_ds_kwargs
        )

    return ds


def _get_demos_for_filter_key(hdf5_path, filter_key):
    """
    Gets demo keys that correspond to a particular filter key.

    Args:
        hdf5_path (str): path to hdf5 file
        filter_key (str): name of filter key

    Returns:
        demo_keys ([str]): list of demonstration keys that
            correspond to this filter key. For example, ["demo_0", 
            "demo_1"].
    """
    f = h5py.File(hdf5_path, "r")
    demo_keys = [elem.decode("utf-8") for elem in np.array(f["mask/{}".format(filter_key)][:])]
    f.close()
    return demo_keys