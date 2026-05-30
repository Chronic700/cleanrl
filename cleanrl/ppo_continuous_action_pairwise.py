# docs and experiment results can be found at https://docs.cleanrl.dev/rl-algorithms/ppo/#ppo_continuous_actionpy
import csv
import os
import random
import sys
import time
from collections import deque
from dataclasses import dataclass
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter

ROOT_DIR = Path(__file__).resolve().parents[3]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from pairwise_advice.cleanrl.preference import (  # noqa: E402
    ComparisonBuffer,
    TrajectoryRecord,
    build_general_comparison,
    sample_partner_for_general_comparison,
    select_teacher_advice_pair,
    train_potential_from_preferences,
)


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "HalfCheetah-v4"
    """the id of the environment"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    num_envs: int = 1
    """the number of parallel game environments"""
    num_steps: int = 2048
    """the number of steps to run in each environment per policy rollout"""
    anneal_lr: bool = True
    """Toggle learning rate annealing for policy and value networks"""
    gamma: float = 0.99
    """the discount factor gamma"""
    gae_lambda: float = 0.95
    """the lambda for the general advantage estimation"""
    num_minibatches: int = 32
    """the number of mini-batches"""
    update_epochs: int = 10
    """the K epochs to update the policy"""
    norm_adv: bool = True
    """Toggles advantages normalization"""
    clip_coef: float = 0.2
    """the surrogate clipping coefficient"""
    clip_vloss: bool = True
    """Toggles whether or not to use a clipped loss for the value function, as per the paper."""
    ent_coef: float = 0.0
    """coefficient of the entropy"""
    vf_coef: float = 0.5
    """coefficient of the value function"""
    max_grad_norm: float = 0.5
    """the maximum norm for the gradient clipping"""
    target_kl: float = None
    """the target KL divergence threshold"""

    # Pairwise-advice shaping
    use_pref_shaping: bool = True
    """enable potential-based shaping from pairwise preferences"""
    teacher_model_path: str = None
    """optional teacher checkpoint path"""
    advice_budget: int = 100
    """number of teacher advice queries"""
    advice_window_episodes: int = 5
    """interval for teacher advice sampling (episodes)"""
    nonzero_priority_prob: float = 0.5
    """prioritize non-zero return trajectories when sampling general comparisons"""
    pref_buffer_size: int = 500
    """max preference buffer size"""
    pref_pairs_per_update: int = 64
    """pairs per potential update"""
    pref_batch_size: int = 16
    """preference minibatch size"""
    pref_updates_per_episode: int = 1
    """preference updates per episode"""
    teacher_pair_weight: float = 5.0
    """relative weight for teacher pairs"""
    teacher_batch_fraction: float = 0.2
    """fraction of training pairs from teacher buffer"""
    potential_lr: float = 1e-3
    """learning rate for potential network"""
    potential_target_sync_episodes: int = 20
    """sync target potential every N episodes"""

    # to be filled in runtime
    batch_size: int = 0
    """the batch size (computed in runtime)"""
    minibatch_size: int = 0
    """the mini-batch size (computed in runtime)"""
    num_iterations: int = 0
    """the number of iterations (computed in runtime)"""


def make_env(env_id, idx, capture_video, run_name, gamma):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.FlattenObservation(env)  # deal with dm_control's Dict observation space
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = gym.wrappers.ClipAction(env)
        env = gym.wrappers.NormalizeObservation(env)
        env = gym.wrappers.TransformObservation(env, lambda obs: np.clip(obs, -10, 10))
        env = gym.wrappers.NormalizeReward(env, gamma=gamma)
        env = gym.wrappers.TransformReward(env, lambda reward: np.clip(reward, -10, 10))
        return env

    return thunk


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs):
        super().__init__()
        self.critic = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(np.array(envs.single_observation_space.shape).prod(), 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, np.prod(envs.single_action_space.shape)), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, np.prod(envs.single_action_space.shape)))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_logstd = self.actor_logstd.expand_as(action_mean)
        action_std = torch.exp(action_logstd)
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)


class PotentialNet(nn.Module):
    def __init__(self, obs_dim: int, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


def discounted_return(rewards, gamma):
    ret = 0.0
    for i, r in enumerate(rewards):
        ret += (gamma ** i) * float(r)
    return float(ret)


def compute_discounted_student_advantage_sum(trajectory, agent, device, gamma):
    if not trajectory:
        return 0.0

    obs = torch.tensor(
        np.stack([step["obs"] for step in trajectory]),
        dtype=torch.float32,
        device=device,
    )
    next_obs = torch.tensor(
        np.stack([step["next_obs"] for step in trajectory]),
        dtype=torch.float32,
        device=device,
    )
    rewards = torch.tensor(
        [step["reward"] for step in trajectory],
        dtype=torch.float32,
        device=device,
    )
    dones = torch.tensor(
        [1.0 if step["done"] else 0.0 for step in trajectory],
        dtype=torch.float32,
        device=device,
    )
    discounts = torch.pow(
        torch.tensor(float(gamma), dtype=torch.float32, device=device),
        torch.arange(len(trajectory), dtype=torch.float32, device=device),
    )

    with torch.no_grad():
        v_s = agent.get_value(obs).view(-1)
        v_sp = agent.get_value(next_obs).view(-1)
        advantages = rewards + gamma * (1.0 - dones) * v_sp - v_s
        discounted_adv_sum = torch.sum(discounts * advantages)

    return float(discounted_adv_sum.item())


def compute_teacher_final_value(final_obs, teacher_model, device):
    if teacher_model is None:
        return 0.0
    obs_t = torch.tensor(final_obs, dtype=torch.float32, device=device).unsqueeze(0)
    with torch.no_grad():
        value = teacher_model.get_value(obs_t).view(-1)[0]
    return float(value.item())


def load_teacher_model(model_path, envs, device):
    teacher = Agent(envs).to(device)
    payload = torch.load(model_path, map_location=device)
    state_dict = payload.get("student_state_dict", payload)
    teacher.load_state_dict(state_dict)
    teacher.eval()
    for p in teacher.parameters():
        p.requires_grad_(False)
    return teacher


if __name__ == "__main__":
    args = tyro.cli(Args)
    args.batch_size = int(args.num_envs * args.num_steps)
    args.minibatch_size = int(args.batch_size // args.num_minibatches)
    args.num_iterations = args.total_timesteps // args.batch_size
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"
    if args.track:
        import wandb

        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )
    run_dir = os.path.join("runs", run_name)
    writer = SummaryWriter(run_dir)
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{key}|{value}|" for key, value in vars(args).items()])),
    )

    os.makedirs(run_dir, exist_ok=True)
    stats_path = os.path.join(run_dir, "stats.csv")
    stats_file = open(stats_path, mode="w", newline="")
    stats_writer = csv.writer(stats_file)
    stats_writer.writerow(
        [
            "event",
            "episode",
            "global_step",
            "discounted_return",
            "pref_loss",
            "pref_acc",
            "pref_pairs",
            "advice_budget_remaining",
        ]
    )
    stats_file.flush()

    # TRY NOT TO MODIFY: seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # env setup
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, i, args.capture_video, run_name, args.gamma) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Box), "only continuous action space is supported"

    agent = Agent(envs).to(device)
    optimizer = optim.Adam(agent.parameters(), lr=args.learning_rate, eps=1e-5)

    potential_net = PotentialNet(np.array(envs.single_observation_space.shape).prod()).to(device)
    potential_target = PotentialNet(np.array(envs.single_observation_space.shape).prod()).to(device)
    potential_target.load_state_dict(potential_net.state_dict())
    potential_opt = optim.Adam(potential_net.parameters(), lr=args.potential_lr, eps=1e-5)

    teacher_model = None
    if args.use_pref_shaping and args.teacher_model_path:
        teacher_model = load_teacher_model(args.teacher_model_path, envs, device)

    general_buffer = ComparisonBuffer(max_size=args.pref_buffer_size)
    teacher_buffer = ComparisonBuffer(max_size=args.pref_buffer_size)
    recent_trajectories = deque(maxlen=args.pref_buffer_size)
    nonzero_trajectories = deque(maxlen=args.pref_buffer_size)
    advice_budget_remaining = int(args.advice_budget)
    total_episodes = 0

    trajectories = [[] for _ in range(args.num_envs)]
    rewards_per_env = [[] for _ in range(args.num_envs)]

    # ALGO Logic: Storage setup
    obs = torch.zeros((args.num_steps, args.num_envs) + envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) + envs.single_action_space.shape).to(device)
    logprobs = torch.zeros((args.num_steps, args.num_envs)).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # TRY NOT TO MODIFY: start the game
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)

    for iteration in range(1, args.num_iterations + 1):
        # Annealing the rate if instructed to do so.
        if args.anneal_lr:
            frac = 1.0 - (iteration - 1.0) / args.num_iterations
            lrnow = frac * args.learning_rate
            optimizer.param_groups[0]["lr"] = lrnow

        for step in range(0, args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            # ALGO LOGIC: action logic
            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action
            logprobs[step] = logprob

            # TRY NOT TO MODIFY: execute the game and log data.
            current_obs = next_obs
            next_obs, reward, terminations, truncations, infos = envs.step(action.cpu().numpy())
            next_done = np.logical_or(terminations, truncations)
            reward_t = torch.tensor(reward).to(device).view(-1)
            next_obs, next_done = torch.Tensor(next_obs).to(device), torch.Tensor(next_done).to(device)

            if args.use_pref_shaping:
                with torch.no_grad():
                    phi_s = potential_target(current_obs).view(-1)
                    phi_sp = potential_target(next_obs).view(-1)
                shaped_reward = reward_t + args.gamma * (1.0 - next_done) * phi_sp - phi_s
                rewards[step] = shaped_reward
            else:
                rewards[step] = reward_t

            for env_idx in range(args.num_envs):
                trajectories[env_idx].append(
                    {
                        "obs": current_obs[env_idx].detach().cpu().numpy(),
                        "next_obs": next_obs[env_idx].detach().cpu().numpy(),
                        "reward": float(reward[env_idx]),
                        "done": bool(next_done[env_idx].item()),
                    }
                )
                rewards_per_env[env_idx].append(float(reward[env_idx]))

                if bool(next_done[env_idx].item()):
                    total_episodes += 1
                    ep_discounted_return = discounted_return(rewards_per_env[env_idx], args.gamma)
                    student_adv_sum = 0.0
                    pref_stats = None
                    if args.use_pref_shaping:
                        student_adv_sum = compute_discounted_student_advantage_sum(
                            trajectories[env_idx], agent, device, args.gamma
                        )

                    final_obs = trajectories[env_idx][-1]["next_obs"]
                    traj_record = TrajectoryRecord(
                        episode=total_episodes,
                        discounted_return=ep_discounted_return,
                        student_adv_sum=student_adv_sum,
                        final_obs=final_obs,
                        trajectory=trajectories[env_idx].copy(),
                    )

                    recent_trajectories.append(traj_record)
                    if abs(ep_discounted_return) > 1e-12:
                        nonzero_trajectories.append(traj_record)

                    if args.use_pref_shaping:
                        partner = sample_partner_for_general_comparison(
                            current_traj=traj_record,
                            recent_trajectories=list(recent_trajectories),
                            nonzero_trajectories=list(nonzero_trajectories),
                            nonzero_priority_prob=args.nonzero_priority_prob,
                        )
                        if partner is not None:
                            partner_record, _ = partner
                            general_cmp = build_general_comparison(traj_record, partner_record)
                            if general_cmp is not None:
                                general_buffer.add(**general_cmp)

                        if (
                            advice_budget_remaining > 0
                            and total_episodes % int(args.advice_window_episodes) == 0
                        ):
                            window_start = max(
                                1, total_episodes - int(args.advice_window_episodes) + 1
                            )
                            window_candidates = [
                                t for t in recent_trajectories if t.episode >= window_start
                            ]
                            if len(window_candidates) >= 2:
                                selected_pair = select_teacher_advice_pair(window_candidates)
                                traj_a, traj_b = selected_pair
                                score_a = compute_teacher_final_value(
                                    traj_a.final_obs, teacher_model, device
                                )
                                score_b = compute_teacher_final_value(
                                    traj_b.final_obs, teacher_model, device
                                )
                                prob_a, gap = (0.5, 0.0)
                                if teacher_model is not None:
                                    prob_a = float(1.0 / (1.0 + np.exp(-(score_a - score_b))))
                                    gap = float(abs(score_a - score_b))
                                advice_budget_remaining -= 1
                                if gap > 1e-12:
                                    teacher_buffer.add(
                                        obs_a=traj_a.final_obs,
                                        obs_b=traj_b.final_obs,
                                        label=prob_a,
                                        gap=gap,
                                        source="final_state_value_teacher_advice",
                                    )

                        pref_stats = train_potential_from_preferences(
                            potential_net=potential_net,
                            optimizer=potential_opt,
                            general_buffer=general_buffer,
                            teacher_buffer=teacher_buffer,
                            device=device,
                            n_pairs=args.pref_pairs_per_update,
                            batch_size=args.pref_batch_size,
                            updates=args.pref_updates_per_episode,
                            teacher_pair_weight=args.teacher_pair_weight,
                            teacher_batch_fraction=args.teacher_batch_fraction,
                        )

                        if total_episodes % int(args.potential_target_sync_episodes) == 0:
                            potential_target.load_state_dict(potential_net.state_dict())

                        if pref_stats is not None:
                            writer.add_scalar("pairwise/pref_loss", pref_stats["pref/loss"], global_step)
                            writer.add_scalar("pairwise/pref_acc", pref_stats["pref/acc"], global_step)
                            writer.add_scalar(
                                "pairwise/pref_pairs",
                                len(general_buffer) + len(teacher_buffer),
                                global_step,
                            )

                    stats_writer.writerow(
                        [
                            "episode",
                            total_episodes,
                            global_step,
                            f"{ep_discounted_return:.6f}",
                            "" if pref_stats is None else f"{pref_stats['pref/loss']:.6f}",
                            "" if pref_stats is None else f"{pref_stats['pref/acc']:.6f}",
                            len(general_buffer) + len(teacher_buffer),
                            advice_budget_remaining,
                        ]
                    )
                    stats_file.flush()

                    trajectories[env_idx] = []
                    rewards_per_env[env_idx] = []

            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info and "episode" in info:
                        print(f"global_step={global_step}, episodic_return={info['episode']['r']}")
                        writer.add_scalar("charts/episodic_return", info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length", info["episode"]["l"], global_step)

        # bootstrap value if not done
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)
            advantages = torch.zeros_like(rewards).to(device)
            lastgaelam = 0
            for t in reversed(range(args.num_steps)):
                if t == args.num_steps - 1:
                    nextnonterminal = 1.0 - next_done
                    nextvalues = next_value
                else:
                    nextnonterminal = 1.0 - dones[t + 1]
                    nextvalues = values[t + 1]
                delta = rewards[t] + args.gamma * nextvalues * nextnonterminal - values[t]
                advantages[t] = lastgaelam = delta + args.gamma * args.gae_lambda * nextnonterminal * lastgaelam
            returns = advantages + values

        # flatten the batch
        b_obs = obs.reshape((-1,) + envs.single_observation_space.shape)
        b_logprobs = logprobs.reshape(-1)
        b_actions = actions.reshape((-1,) + envs.single_action_space.shape)
        b_advantages = advantages.reshape(-1)
        b_returns = returns.reshape(-1)
        b_values = values.reshape(-1)

        # Optimizing the policy and value network
        b_inds = np.arange(args.batch_size)
        clipfracs = []
        for epoch in range(args.update_epochs):
            np.random.shuffle(b_inds)
            for start in range(0, args.batch_size, args.minibatch_size):
                end = start + args.minibatch_size
                mb_inds = b_inds[start:end]

                _, newlogprob, entropy, newvalue = agent.get_action_and_value(b_obs[mb_inds], b_actions[mb_inds])
                logratio = newlogprob - b_logprobs[mb_inds]
                ratio = logratio.exp()

                with torch.no_grad():
                    # calculate approx_kl http://joschu.net/blog/kl-approx.html
                    old_approx_kl = (-logratio).mean()
                    approx_kl = ((ratio - 1) - logratio).mean()
                    clipfracs += [((ratio - 1.0).abs() > args.clip_coef).float().mean().item()]

                mb_advantages = b_advantages[mb_inds]
                if args.norm_adv:
                    mb_advantages = (mb_advantages - mb_advantages.mean()) / (mb_advantages.std() + 1e-8)

                # Policy loss
                pg_loss1 = -mb_advantages * ratio
                pg_loss2 = -mb_advantages * torch.clamp(ratio, 1 - args.clip_coef, 1 + args.clip_coef)
                pg_loss = torch.max(pg_loss1, pg_loss2).mean()

                # Value loss
                newvalue = newvalue.view(-1)
                if args.clip_vloss:
                    v_loss_unclipped = (newvalue - b_returns[mb_inds]) ** 2
                    v_clipped = b_values[mb_inds] + torch.clamp(
                        newvalue - b_values[mb_inds],
                        -args.clip_coef,
                        args.clip_coef,
                    )
                    v_loss_clipped = (v_clipped - b_returns[mb_inds]) ** 2
                    v_loss_max = torch.max(v_loss_unclipped, v_loss_clipped)
                    v_loss = 0.5 * v_loss_max.mean()
                else:
                    v_loss = 0.5 * ((newvalue - b_returns[mb_inds]) ** 2).mean()

                entropy_loss = entropy.mean()
                loss = pg_loss - args.ent_coef * entropy_loss + v_loss * args.vf_coef

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(agent.parameters(), args.max_grad_norm)
                optimizer.step()

            if args.target_kl is not None and approx_kl > args.target_kl:
                break

        y_pred, y_true = b_values.cpu().numpy(), b_returns.cpu().numpy()
        var_y = np.var(y_true)
        explained_var = np.nan if var_y == 0 else 1 - np.var(y_true - y_pred) / var_y

        # TRY NOT TO MODIFY: record rewards for plotting purposes
        writer.add_scalar("charts/learning_rate", optimizer.param_groups[0]["lr"], global_step)
        writer.add_scalar("losses/value_loss", v_loss.item(), global_step)
        writer.add_scalar("losses/policy_loss", pg_loss.item(), global_step)
        writer.add_scalar("losses/entropy", entropy_loss.item(), global_step)
        writer.add_scalar("losses/old_approx_kl", old_approx_kl.item(), global_step)
        writer.add_scalar("losses/approx_kl", approx_kl.item(), global_step)
        writer.add_scalar("losses/clipfrac", np.mean(clipfracs), global_step)
        writer.add_scalar("losses/explained_variance", explained_var, global_step)
        print("SPS:", int(global_step / (time.time() - start_time)))
        writer.add_scalar("charts/SPS", int(global_step / (time.time() - start_time)), global_step)

    if args.save_model:
        model_path = f"runs/{run_name}/{args.exp_name}.cleanrl_model"
        torch.save(agent.state_dict(), model_path)
        print(f"model saved to {model_path}")
        from cleanrl_utils.evals.ppo_eval import evaluate

        episodic_returns = evaluate(
            model_path,
            make_env,
            args.env_id,
            eval_episodes=10,
            run_name=f"{run_name}-eval",
            Model=Agent,
            device=device,
            gamma=args.gamma,
        )
        for idx, episodic_return in enumerate(episodic_returns):
            writer.add_scalar("eval/episodic_return", episodic_return, idx)

        if args.upload_model:
            from cleanrl_utils.huggingface import push_to_hub

            repo_name = f"{args.env_id}-{args.exp_name}-seed{args.seed}"
            repo_id = f"{args.hf_entity}/{repo_name}" if args.hf_entity else repo_name
            push_to_hub(args, episodic_returns, repo_id, "PPO", f"runs/{run_name}", f"videos/{run_name}-eval")

    envs.close()
    stats_file.close()
    writer.close()
