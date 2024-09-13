"""
Set of classes to load ManiSkill datasets.

The datasets are stored in h5 files and are loaded using the h5py library.
"""
from typing import Union

import h5py
import numpy as np
import os
from PIL import Image
from nerv.utils import read_img, load_obj
import collections

from mani_skill.utils.io_utils import load_json

import torch
from torch.utils.data import Dataset
import cv2
import numpy as np

import torch
import torchvision.transforms as transforms
from torchvision.transforms import functional as F

def compact(l):
    return list(filter(None, l))


class BaseTransforms(object):
    """Data pre-processing steps."""

    def __init__(self, resolution, mean=(0.5, ), std=(0.5, )):
        self.transforms = transforms.Compose([
            transforms.ToTensor(),  # [3, H, W]
            transforms.Normalize(mean, std),  # [-1, 1]
            transforms.Resize(resolution, antialias=True),
        ])
        self.resolution = resolution

    def process_mask(self, mask):
        if isinstance(mask, np.ndarray):
            mask = torch.from_numpy(mask).long()
        if len(mask.shape) == 2:
            mask = mask.unsqueeze(0)
            mask = F.resize(
                mask,
                self.resolution,
                interpolation=transforms.InterpolationMode.NEAREST)[0]
        else:
            mask = F.resize(
                mask,
                self.resolution,
                interpolation=transforms.InterpolationMode.NEAREST)
        return mask

    def __call__(self, input):
        return self.transforms(input)


# loads h5 data into memory for faster access
def load_h5_data(data):
    out = dict()
    for k in data.keys():
        if isinstance(data[k], h5py.Dataset):
            out[k] = data[k][:]
        else:
            out[k] = load_h5_data(data[k])
    return out
def load_h5_data_filtered(data, keys):
    out = dict()
    for k in data.keys():
        if k not in keys:
            if isinstance(data[k], h5py.Dataset):
                out[k] = data[k][:]
            else:
                out[k] = load_h5_data_filtered(data[k], keys)
    return out

class ManiSkillVisionDataset(Dataset):
    """
    Dataset loading frames extracted from ManiSkill trajectories.
    The frames are stored in a 'videos' folder in the dataset folder.
    Two subfolders 'train' and 'val' are expected to be present in the 
    'videos' and contain the frames for the training and validation sets respectively.
    """
    def __init__(
        self, 
        data_root: str, 
        transform,
        split='train',
        n_sample_frames=6,
        frame_offset=None,
        filter_enter=False,
    ) -> None:
        self.data_root = data_root
        self.split = split
        assert self.split in ['train', 'val']
        # Load videos
        self.video_path = os.path.join(data_root, 'videos', self.split)
        
        self.transform = transform
        self.video_lens = []
        self.n_sample_frames = n_sample_frames
        self.frame_offset =  frame_offset

        # all video paths
        self.files = self._get_files()
        self.num_videos = len(self.files)
        self.filter_enter = filter_enter
        if self.filter_enter:
            self.valid_idx = self._get_filtered_sample_idx()
        else:
            self.valid_idx = self._get_sample_idx()

        # by default, we load small video clips
        self.load_video = False

    def pad_video(self, video, max_len):
        """Pad video to max_len."""
        if len(video) >= max_len:
            return video
        dup_video = torch.stack(
            [video[-1]] * (max_len - video.shape[0]), dim=0)
        return torch.cat([video, dup_video], dim=0)
    
    def _get_files(self):
        """Get path for all videos."""
        video_paths = [os.path.join(self.video_path, f) \
                    for f in os.listdir(self.video_path) \
                    if os.path.isdir(os.path.join(self.video_path, f))]
        return sorted(compact(video_paths))

    def get_video(self, video_idx):
        video_path = self.files[video_idx]
        if len(os.listdir(video_path)) != self.video_lens[video_idx]+ 1:
            raise ValueError
        max_len = max(int(len_video // self.frame_offset) for len_video  in self.video_lens)
        frames = [
            read_img(
                os.path.join(
                    video_path,
                    f'{n:04d}.jpg'))
            for n in range(1, self.video_lens[video_idx], self.frame_offset)
        ]
        if any(frame is None for frame in frames):
            raise ValueError
        video = [
            self.transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)))
            for img in frames
        ]  # [T, C, H, W]
        video = torch.stack(video, dim=0)
        video = self.pad_video(video, max_len)
        return {
            'video': video,
            'error_flag': False,
            'data_idx': video_idx,
        }

    def _get_sample_idx(self):
        """Get (video_idx, start_frame_idx) pairs as a list."""
        valid_idx = []
        video_lens = {}
        self.video_lens = []
        drop_ids = []
        for video_idx in range(len(self.files)):
            video_path = self.files[video_idx]
            if len(os.listdir(video_path)) - 1 > 800:
                drop_ids.append(video_idx)
        self.files = [f for i, f in enumerate(self.files) if i not in drop_ids]
        for video_idx in range(len(self.files)):
            video_path = self.files[video_idx]
            video_lens[video_idx] = len(os.listdir(video_path)) - 1
            # simply use random uniform sampling
            max_start_idx = video_lens[video_idx] - \
                (self.n_sample_frames -1)* self.frame_offset
            if self.split == 'train':
                valid_idx += [(video_idx, idx) for idx in range(1, max_start_idx)]
            else:
                size = self.n_sample_frames * self.frame_offset
                start_idx = []
                for idx in range(1, video_lens[video_idx] - size + 1, size):
                    start_idx += [i + idx for i in range(self.frame_offset)]
                valid_idx += [(video_idx, idx) for idx in start_idx]
        video_lens = collections.OrderedDict(sorted(video_lens.items()))
        for k, v in video_lens.items():
            self.video_lens.append(v)
        return valid_idx

    def _get_filtered_sample_idx(self):
        """Get (video_idx, start_frame_idx) pairs as a list."""
        valid_idx = []
        video_lens = {}
        self.video_lens = []
        drop_ids = []
        for video_idx in range(len(self.files)):
            video_path = self.files[video_idx]
            if len(os.listdir(video_path)) - 1 > 800:
                drop_ids.append(video_idx)
        self.files = [f for i, f in enumerate(self.files) if i not in drop_ids]
        for video_idx in range(len(self.files)):
            video_path = self.files[video_idx]
            video_len = len(os.listdir(video_path)) - 1
            video_lens[video_idx] = video_len
            max_start_idx = video_len - \
                (self.n_sample_frames - 1) * self.frame_offset
            if self.split == 'train':
                valid_idx += [(video_idx, idx) for idx in range(1, max_start_idx)]
            else:
                size = (self.n_sample_frames - 1) * self.frame_offset
                interval = size // 2
                for idx in range(1, video_len - size, interval):
                    max_idx = min(idx + interval, video_len - size)
                    for sub_idx in range(idx, max_idx, 1):
                        valid_idx.append((video_idx, sub_idx))
                        break
        video_lens = collections.OrderedDict(sorted(video_lens.items()))
        for k, v in video_lens.items():
            self.video_lens.append(v)
        print(f"Number of valid samples: {len(valid_idx)}")
        return valid_idx
    
    def __len__(self):
        return len(self.valid_idx)

    def _read_frames(self, idx):
        """Read video frames. Directly read from jpg images."""
        video_idx, start_idx = self._get_video_start_idx(idx)
        video_path = self.files[video_idx]

        # wrong video length !
        if len(os.listdir(video_path)) != self.video_lens[video_idx] + 1:
            raise ValueError
        frames = [
            read_img(
                os.path.join(
                    video_path,
                    f'{(start_idx + n * self.frame_offset):04d}.jpg'))
            for n in range(self.n_sample_frames)
        ]
        if any(frame is None for frame in frames):
            raise ValueError
        frames = [
            self.transform(Image.fromarray(cv2.cvtColor(img, cv2.COLOR_RGB2BGR)))
            for img in frames
        ]  # [T, C, H, W]
        return torch.stack(frames, dim=0).float()
    
    def _rand_another(self, is_video=False):
        """Random get another sample when encountering loading error."""
        if is_video:
            another_idx = np.random.choice(self.num_videos)
            return self.get_video(another_idx)
        another_idx = np.random.choice(len(self))
        return self.__getitem__(another_idx)

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - images: [T, 3, H, W]
            - error_flag: whether loading `idx` causes error and _rand_another
        """
        try:
            frames = self._read_frames(idx)
            data_dict = {
                'data_idx': idx,
                'images': frames,
                'error_flag': False,
            }
        # corrupted video
        except ValueError:
            data_dict = self._rand_another()
            data_dict['error_flag'] = True
            return data_dict
        return data_dict
    
    def _get_video_start_idx(self, idx):
        return self.valid_idx[idx]

def build_maniskill_dataset(params, val_only=False, test_set=False):
    """Build ManiSkill video dataset."""
    args = dict(
        data_root=params.data_root,
        split='val',
        transform=BaseTransforms(params.resolution),
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
    )
    val_dataset = ManiSkillVisionDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = ManiSkillVisionDataset(**args)
    return train_dataset, val_dataset

class ManiSkillTrajDataset(ManiSkillVisionDataset):
    """
    Dataset loading all the trajectories from ManiSkill.
    The frames are stored in a 'videos' folder in the dataset folder.
    Two subfolders 'train' and 'val' are expected to be present in the 
    'videos' and contain the frames for the training and validation sets respectively.
    """
    def __init__(
        self, 
        data_root: str, 
        transform,
        split='train',
        n_sample_frames=6 + 10,
        frame_offset=None,
        filter_enter=True,
        rt1_format=False,
        load_img=True,
    ) -> None:
        # Load the data
        trajectory_file = data_root + "trajectories.h5"
        self.data = h5py.File(trajectory_file, "r")
        self.trajectories = load_h5_data_filtered(self.data, ["obs", "terminated", "rewards", "success"])
        json_path = trajectory_file.replace(".h5", ".json")
        # Load the corresponding metadata
        self.json_data = load_json(json_path)
        self.episodes = self.json_data["episodes"]
        self.env_info = self.json_data["env_info"]
        self.env_id = self.env_info["env_id"]
        self.env_kwargs = self.env_info["env_kwargs"]
        self.load_img = load_img
        super().__init__(
            data_root=data_root, 
            transform=transform,
            split=split,
            n_sample_frames=n_sample_frames,
            frame_offset=frame_offset,
            filter_enter=filter_enter,
        )
        self.rt1_format = rt1_format
    
    def _get_eps_idx(self, video_id):
        return int(self.files[video_id].split("_")[-1])

    def _read_traj(self, idx):
        """Read video frames slots."""
        video_id, start_idx = self._get_video_start_idx(idx)
        eps_id = self._get_eps_idx(video_id)
        
        # Load episodes
        trajectory = self.trajectories["traj_" + str(eps_id)]
        actions = [
            trajectory["actions"][start_idx + n * self.frame_offset]
            for n in range(self.n_sample_frames)
        ]
        actions = np.stack(actions, axis=0).astype(np.float32)
        if self.load_img:
            img = super()._read_frames(idx)
            return img, actions
        return actions

    def _read_full_traj(self, idx):
        """Read video frames slots."""
        video_idx, _ = self._get_video_start_idx(idx)
        # Load episodes
        eps_id = self._get_eps_idx(video_idx)
        trajectory = self.trajectories["traj_" + str(eps_id)]
        actions = trajectory["actions"]
        
        video = super().get_video(video_idx)
        if self.rt1_format:
            terminated =  trajectory["terminated"]
            actions = {
                "world_vector": actions[:, :3],
                "rotation": actions[:, 3:-1],
                "gripper": actions[:, -1],
                "terminated": terminated
            }

        return video["video"], actions
    
    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - images: [T, 3, H, W]
            - slots: [T, N, C] slots extracted from CLEVRER video frames
            - error_flag: whether loading `idx` causes error and _rand_another
            - mask: [T, H, W], 0 is background
            - pres_mask: [T, max_num_obj], valid index of objects
            - bbox: [T, max_num_obj, 4], object bbox padded to `max_num_obj`
        """
        try :
            if self.load_img:
                img, actions = self._read_traj(idx)
                data_dict = {
                    'data_idx': idx,
                    'error_flag': False,
                    'actions': actions,
                    'images': img,
                }
            else:
                actions = self._read_traj(idx)
                data_dict = {
                    'data_idx': idx,
                    'error_flag': False,
                    'actions': actions,
                }
        except ValueError:
            print(f"Error while loading {idx}, getting another sample")
            data_dict = self._rand_another()
            data_dict['error_flag'] = True
            return
        return data_dict

def build_maniskill_traj_dataset(params, val_only=False, test_set=False):
    """Build ManiSkill slot dataset with pre-computed slots."""
    params.resolution = (params.resolution, params.resolution)
    args = dict(
        data_root=params.data_root,
        transform=BaseTransforms(params.resolution),
        split='val',
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        filter_enter=params.filter_enter,
    )
    val_dataset = ManiSkillTrajDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    train_dataset = ManiSkillTrajDataset(**args)
    return train_dataset, val_dataset


### The following datasets require pre-computed latents or slots
class ManiSkillLatentDataset(ManiSkillTrajDataset):
    def __init__(
        self, 
        data_root: str, 
        video_latents,
        transform,
        split='train',
        n_sample_frames=6 + 10,
        frame_offset=None,
        filter_enter=True,
        load_img=False,
    ) -> None:
        super().__init__(
            data_root=data_root, 
            transform=transform,
            split=split,
            n_sample_frames=n_sample_frames,
            frame_offset=frame_offset,
            filter_enter=filter_enter,
            load_img=load_img,
        )

        # pre-computed latents
        self.video_latents = video_latents

    def _read_latents(self, idx):
        """Read video frames latents."""
        video_idx, start_idx = self._get_video_start_idx(idx)
        video_path = self.files[video_idx]
        video_len = self.video_lens[video_idx]
        try:
            latents = self.video_latents[os.path.basename(video_path)][:video_len]  # [T, N, C]
        except KeyError:
            raise ValueError
        latents = [
            latents[start_idx + n * self.frame_offset]
            for n in range(self.n_sample_frames)
        ]
        return np.stack(latents, axis=0).astype(np.float32)

    def __getitem__(self, idx):
        """Data dict:
            - data_idx: int
            - images: [T, 3, H, W]
            - latents: [T, D] latents extracted from maniskill video frames
            - actions: [T, A] actions extracted from maniskill video frames
            - error_flag: whether loading `idx` causes error and _rand_another
        """
        try :
            data_dict = {
                'data_idx': idx,
                'error_flag': False,
                'latents': self._read_latents(idx),
                'actions': self._read_traj(idx),
            }
            if self.load_img:
                data_dict['images'] = self._read_frames(idx)
        except ValueError:
            print(f"Error while loading {idx}, getting another sample")
            data_dict = self._rand_another()
            data_dict['error_flag'] = True
            return
        return data_dict

def build_maniskill_latents_dataset(params, val_only=False, test_set=False):
    """Build ManiSkill dataset with pre-computed latents (either classical latents or slots)"""
    latents = load_obj(params.latents_root)
    args = dict(
        data_root=params.data_root,
        video_latents=latents['val'],
        transform=BaseTransforms(params.resolution),
        split='val',
        n_sample_frames=params.n_sample_frames,
        frame_offset=params.frame_offset,
        load_img=True,
        filter_enter=params.filter_enter,
    )
    val_dataset = ManiSkillLatentDataset(**args)
    if val_only:
        return val_dataset
    args['split'] = 'train'
    args['load_img'] = params.load_img
    args['video_latents'] = latents['train']
    train_dataset = ManiSkillLatentDataset(**args)
    return train_dataset , val_dataset
