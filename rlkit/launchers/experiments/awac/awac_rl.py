from gym.spaces import Dict

import rlkit.torch.pytorch_util as ptu
from rlkit.data_management.env_replay_buffer import EnvReplayBuffer
from rlkit.data_management.split_buffer import SplitReplayBuffer
from rlkit.envs.make_env import make
from rlkit.envs.wrappers import StackObservationEnv
from rlkit.exploration_strategies.base import PolicyWrappedWithExplorationStrategy
from rlkit.exploration_strategies.gaussian_and_epsilon_strategy import GaussianAndEpsilonStrategy
from rlkit.exploration_strategies.ou_strategy import OUStrategy
from rlkit.samplers.data_collector import MdpPathCollector
from rlkit.samplers.data_collector.step_collector import MdpStepCollector
from rlkit.torch.networks import ConcatMlp
from rlkit.torch.sac.awac_trainer import AWACTrainer
from rlkit.torch.sac.policies import TanhGaussianPolicy, MakeDeterministic
from rlkit.torch.torch_rl_algorithm import (
    TorchBatchRLAlgorithm,
    TorchOnlineRLAlgorithm,
)
from rlkit.util.io import load_local_or_remote_file


def experiment(variant):

    normalize_env = variant.get('normalize_env', True)
    env_id = variant.get('env_id', None)
    env_class = variant.get('env_class', None)
    env_kwargs = variant.get('env_kwargs', {})

    expl_env = make(env_id, env_class, env_kwargs, normalize_env)
    eval_env = make(env_id, env_class, env_kwargs, normalize_env)

    if variant.get('add_env_demos', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_demo_path"])

    if variant.get('add_env_offpolicy_data', False):
        variant["path_loader_kwargs"]["demo_paths"].append(variant["env_offpolicy_data_path"])

    path_loader_kwargs = variant.get("path_loader_kwargs", {})
    stack_obs = path_loader_kwargs.get("stack_obs", 1)
    if stack_obs > 1:
        expl_env = StackObservationEnv(expl_env, stack_obs=stack_obs)
        eval_env = StackObservationEnv(eval_env, stack_obs=stack_obs)

    if isinstance(expl_env.observation_space, Dict):
        obs_dim = sum(v.low.size for v in expl_env.observation_space.spaces.values())
    else:
        obs_dim = expl_env.observation_space.low.size

    action_dim = eval_env.action_space.low.size

    qf_kwargs = variant.get("qf_kwargs", {})
    qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )

    qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )

    target_qf1 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )

    target_qf2 = ConcatMlp(
        input_size=obs_dim + action_dim,
        output_size=1,
        **qf_kwargs
    )

    policy_class = variant.get("policy_class", TanhGaussianPolicy)
    policy_kwargs = variant['policy_kwargs']
    policy_path = variant.get("policy_path", False)
    policy = build_policy(action_dim, obs_dim, policy_class, policy_kwargs, policy_path)

    buffer_policy_class = variant.get("buffer_policy_class", policy_class)
    buffer_policy_kwargs = variant.get("buffer_policy_kwargs", policy_kwargs)
    buffer_policy_path = variant.get("buffer_policy_path", False)
    buffer_policy = build_policy(action_dim, obs_dim, buffer_policy_class, buffer_policy_kwargs, buffer_policy_path)

    eval_policy = MakeDeterministic(policy)

    exploration_kwargs = variant.get('exploration_kwargs', {})
    if exploration_kwargs and exploration_kwargs.get("deterministic_exploration", False):
        expl_policy = MakeDeterministic(policy)
    else:
        expl_policy = build_exploration_policy(expl_env, policy, exploration_kwargs)

    eval_path_collector = MdpPathCollector(
        eval_env,
        eval_policy,
    )

    replay_buffer_kwargs = dict(
        max_replay_buffer_size=variant['replay_buffer_size'],
        env=expl_env,
    )

    replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
        **replay_buffer_kwargs,
    )

    if variant.get('use_validation_buffer', False):
        validation_replay_buffer = variant.get('replay_buffer_class', EnvReplayBuffer)(
            **replay_buffer_kwargs,
        )
        replay_buffer = SplitReplayBuffer(replay_buffer, validation_replay_buffer, 0.9)

    trainer_class = variant.get("trainer_class", AWACTrainer)
    trainer = trainer_class(
        env=eval_env,
        policy=policy,
        qf1=qf1,
        qf2=qf2,
        target_qf1=target_qf1,
        target_qf2=target_qf2,
        buffer_policy=buffer_policy,
        **variant['trainer_kwargs']
    )

    algorithm = build_exploration_algorithm(eval_env,
                                            eval_path_collector,
                                            expl_env,
                                            expl_policy,
                                            policy,
                                            replay_buffer,
                                            trainer,
                                            variant)

    demo_train_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )
    demo_test_buffer = EnvReplayBuffer(
        **replay_buffer_kwargs,
    )

    if variant.get('pretrain_buffer_policy', False):
        trainer.pretrain_policy_with_bc(
            buffer_policy,
            replay_buffer.train_replay_buffer,
            replay_buffer.validation_replay_buffer,
            10000,
            label="buffer",
        )

    if variant.get('pretrain_policy', False):
        trainer.pretrain_policy_with_bc(
            policy,
            demo_train_buffer,
            demo_test_buffer,
            trainer.bc_num_pretrain_steps,
        )

    if variant.get('pretrain_rl', False):
        trainer.pretrain_q_with_bc_data()

    if variant.get('train_rl', True):
        algorithm.train()


def build_exploration_algorithm(eval_env,
                                eval_path_collector,
                                expl_env,
                                expl_policy,
                                policy,
                                replay_buffer,
                                trainer,
                                variant):

    if variant['collection_mode'] == 'online':
        algorithm = TorchOnlineRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=MdpStepCollector(
                expl_env,
                policy,
            ),
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=variant['max_path_length'],
            batch_size=variant['batch_size'],
            num_epochs=variant['num_epochs'],
            num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
            num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
            num_trains_per_train_loop=variant['num_trains_per_train_loop'],
            min_num_steps_before_training=variant['min_num_steps_before_training'],
        )

    else:
        algorithm = TorchBatchRLAlgorithm(
            trainer=trainer,
            exploration_env=expl_env,
            evaluation_env=eval_env,
            exploration_data_collector=MdpPathCollector(
                expl_env,
                expl_policy,
            ),
            evaluation_data_collector=eval_path_collector,
            replay_buffer=replay_buffer,
            max_path_length=variant['max_path_length'],
            batch_size=variant['batch_size'],
            num_epochs=variant['num_epochs'],
            num_eval_steps_per_epoch=variant['num_eval_steps_per_epoch'],
            num_expl_steps_per_train_loop=variant['num_expl_steps_per_train_loop'],
            num_trains_per_train_loop=variant['num_trains_per_train_loop'],
            min_num_steps_before_training=variant['min_num_steps_before_training'],
        )

    algorithm.to(ptu.device)
    return algorithm


def build_policy(action_dim, obs_dim, policy_class, policy_kwargs, policy_path=None):
    if policy_path:
        policy = load_local_or_remote_file(policy_path)
    else:
        policy = policy_class(
            obs_dim=obs_dim,
            action_dim=action_dim,
            **policy_kwargs,
        )
    return policy


def build_exploration_policy(expl_env, policy, exploration_kwargs):

    exploration_strategy = exploration_kwargs.get("strategy", None)
    if exploration_strategy is None:
        return policy

    elif exploration_strategy == 'ou':
        es = OUStrategy(
            action_space=expl_env.action_space,
            max_sigma=exploration_kwargs['noise'],
            min_sigma=exploration_kwargs['noise'],
        )
        expl_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=policy
        )

    elif exploration_strategy == 'gauss_eps':
        es = GaussianAndEpsilonStrategy(
            action_space=expl_env.action_space,
            max_sigma=exploration_kwargs['noise'],
            min_sigma=exploration_kwargs['noise'],  # constant sigma
            epsilon=0,
        )
        expl_policy = PolicyWrappedWithExplorationStrategy(
            exploration_strategy=es,
            policy=policy
        )
    else:
        raise ValueError('exploration_strategy')

    return expl_policy
