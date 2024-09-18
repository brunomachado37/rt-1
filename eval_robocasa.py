import os
import json
import yaml
import argparse
import torch
import numpy as np
import torch

from rt1_torch.rt1 import RT1

import robomimic.utils.train_utils as TrainUtils
import robomimic.utils.lang_utils as LangUtils
import robomimic.utils.env_utils as EnvUtils
import robomimic.utils.obs_utils as ObsUtils

from robomimic.config import config_factory
from robomimic.algo import RolloutPolicy



class PolicyTransformerWrapper():
    """
    Lightweight Wrapper around the PolicyTransformerModel to be used in the RoboCasa environment.
    """
    
    def __init__(self, policy, image_preprocess, global_config, device):
        self.policy = policy
        self.image_preprocess = image_preprocess
        self.global_config = global_config
        self.device = device

    def set_eval(self):
        self.policy.eval()

    def reset(self):
        pass

    def __repr__(self):
        return self.policy.__repr__()
    
    def get_action(self, obs_dict, goal_dict=None):
        language_embeddings = obs_dict["lang_emb"]
        actions = obs_dict["actions"][:, :7]
        imgs = obs_dict["robot0_agentview_left_image"]

        B, T, C, H, W = imgs.shape
        imgs = torch.reshape(imgs, [B * T, C, H, W])                                        # Fold time into batch dimension       
        images = self.image_preprocess(imgs)
        images = torch.reshape(images, [B, T, C, images.shape[-2], images.shape[-1]])       # Unfold time dimension

        output = self.policy(images=images, language_embeddings=language_embeddings)

        # Take only last prediction
        action = output['action_logits'][0, -1, :]

        # Add actions relative to the base movement, which the model does not predict
        action = torch.cat((action, torch.zeros(5, device=output['action_logits'].device))).unsqueeze(0)
      
        return action


def env_iterator(config, eval_env_meta_list, eval_shape_meta_list):
    for (env_meta, shape_meta) in zip(eval_env_meta_list, eval_shape_meta_list):
        def create_env_helper(env_i=0):
            env_kwargs = dict(
                env_meta=env_meta,
                env_name=env_meta['env_name'],
                render=False,
                render_offscreen=config.experiment.render_video,
                use_image_obs=shape_meta["use_images"],
                seed=config.train.seed * 1000 + env_i,
            )
            env = EnvUtils.create_env_from_metadata(**env_kwargs)
            # handle environment wrappers
            env = EnvUtils.wrap_env_from_config(env, config=config)  # apply environment wrapper, if applicable

            return env

        if config.experiment.rollout.batched:
            from tianshou.env import SubprocVectorEnv
            env_fns = [lambda env_i=i: create_env_helper(env_i) for i in range(config.experiment.rollout.num_batch_envs)]
            env = SubprocVectorEnv(env_fns)
            # env_name = env.get_env_attr(key="name", id=0)[0]
        else:
            env = create_env_helper()
            # env_name = env.name
        print(env)
        yield env


def eval(args):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ext_cfg = json.load(open(os.path.join(args.eval_config, "config.json")))
    config = config_factory(ext_cfg["algo_name"])

    with config.values_unlocked():
        config.update(ext_cfg)
        config['experiment']['name'] = args.model_config.removeprefix("conf/").removesuffix(".yaml")

    ObsUtils.initialize_obs_utils_with_config(config)

    env_config = json.load(open(os.path.join(args.eval_config, "env_config.json")))
    env_config['action_normalization_stats']['actions'] = {k: np.array(v) for k, v in env_config['action_normalization_stats']['actions'].items()}

    with open(args.model_config, 'r') as file:
        model_config = yaml.safe_load(file)

    policy_model = RT1(**model_config["model"]).to(device)
    weights = torch.load(args.checkpoint_path, map_location=device)
    policy_model.load_state_dict({k.removeprefix("model."): v for k, v in weights['state_dict'].items()})
    policy_model.eval()

    image_preprocess = policy_model.image_tokenizer.vision_model_weights.transforms()
    model = PolicyTransformerWrapper(policy=policy_model, image_preprocess=image_preprocess, global_config=config, device=device)

    lang_encoder = LangUtils.language_encoder_factory(model=model_config['data']['language_encoder'], device=device)

    rollout_model = RolloutPolicy(
        model,
        obs_normalization_stats=env_config['obs_normalization_stats'],
        action_normalization_stats=env_config['action_normalization_stats'],
        lang_encoder=lang_encoder,
    )

    video_path = os.path.join(args.output_dir, config.experiment.name, "videos")
    if not os.path.exists(video_path):
        os.makedirs(video_path)

    all_rollout_logs, video_paths = TrainUtils.rollout_with_stats(
        policy=rollout_model,
        envs=env_iterator(config, env_config['env_meta_list'], env_config['shape_meta_list']),
        horizon=[env['horizon'] for env in config.train.data],
        use_goals=config.use_goals,
        num_episodes=args.evaluations_per_task,
        render=False,
        video_dir=video_path,
        epoch=args.checkpoint_path.split('epoch=')[-1].removesuffix('.ckpt'),
        video_skip=config.experiment.get("video_skip", 5),
        terminate_on_success=config.experiment.rollout.terminate_on_success,
        del_envs_after_rollouts=True,
        data_logger=None,
    )

    for task, ret in all_rollout_logs.items():
        print(f"\n{task} success rate: {ret['Success_Rate']:.2f}\n")

    all_rollout_logs["rollouts_per_task"] = args.evaluations_per_task
    eval_results_path = os.path.join(args.output_dir, config.experiment.name, f"eval_results_epoch_{args.checkpoint_path.split('epoch=')[-1].removesuffix('.ckpt')}.json")

    with open(eval_results_path, "w") as f:
        json.dump(all_rollout_logs, f, indent=4)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--model_config",
        type=str,
        default="conf/10_config_robocasa_lr-1e-4.yaml",
        help="Path to the model configuration file",
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="runs/RT-1-RoboCasa/r6q08cgi/checkpoints/epoch=9-step=11520.ckpt",
        help="Path to the run foler containing the model to evaluate",
    )

    parser.add_argument(
        "--eval_config",
        type=str,
        default="conf/eval",
        help="Path to the evaluation configuration file",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default='eval',
        help="Directory to save evaluation results",
    )

    parser.add_argument(
        "--evaluations_per_task",
        type=int,
        default=10,
        help="Number of rollouts to perform per task",
    )

    args = parser.parse_args()
    eval(args)
