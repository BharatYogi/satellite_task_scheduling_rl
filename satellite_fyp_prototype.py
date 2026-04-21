from __future__ import annotations

import argparse
import csv
import math
import os
import random
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
except Exception as exc:  # pragma: no cover
    raise RuntimeError(
        "PyTorch is required for this script. Install torch before running."
    ) from exc


# ============================================================
# Data model
# ============================================================


@dataclass
class SatelliteTask:
    task_id: int
    window_start: int
    window_end: int
    duration: int
    priority: int
    energy_cost: float
    storage_cost: float
    reward_value: float


@dataclass
class ScheduleEntry:
    method: str
    task_id: int
    start_time: int
    end_time: int
    priority: int
    reward_value: float
    battery_after: float
    storage_after: float


# ============================================================
# Task generation
# ============================================================


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


class TaskGenerator:
    """Generates a synthetic one-day satellite scheduling dataset."""

    def __init__(self, horizon: int = 96):
        self.horizon = horizon

    def generate(self, n_tasks: int, seed: int) -> List[SatelliteTask]:
        rng = random.Random(seed)
        tasks: List[SatelliteTask] = []

        for task_id in range(n_tasks):
            duration = rng.randint(1, 6)

            # Mix of narrow, medium and wide windows
            window_type = rng.random()
            if window_type < 0.35:
                window_length = rng.randint(2, 6)
            elif window_type < 0.75:
                window_length = rng.randint(7, 14)
            else:
                window_length = rng.randint(15, 28)

            window_start = rng.randint(0, max(0, self.horizon - window_length - 1))
            window_end = min(self.horizon - 1, window_start + window_length)

            # Skew priority so there are fewer high-priority tasks
            priority_roll = rng.random()
            if priority_roll < 0.40:
                priority = 1
            elif priority_roll < 0.65:
                priority = 2
            elif priority_roll < 0.82:
                priority = 3
            elif priority_roll < 0.94:
                priority = 4
            else:
                priority = 5

            # More variation so heuristics behave differently
            reward_value = priority * 12 + rng.uniform(-2.0, 12.0)
            energy_cost = rng.uniform(5.0, 18.0) + priority * rng.uniform(0.5, 1.8)
            storage_cost = rng.uniform(4.0, 16.0)

            tasks.append(
                SatelliteTask(
                    task_id=task_id,
                    window_start=window_start,
                    window_end=window_end,
                    duration=duration,
                    priority=priority,
                    energy_cost=round(energy_cost, 2),
                    storage_cost=round(storage_cost, 2),
                    reward_value=round(reward_value, 2),
                )
            )

        tasks.sort(key=lambda t: (t.window_start, t.window_end, -t.priority))
        return tasks

    @staticmethod
    def to_dataframe(tasks: Sequence[SatelliteTask]) -> pd.DataFrame:
        return pd.DataFrame([asdict(t) for t in tasks])


# ============================================================
# Environment
# ============================================================


class SatelliteSchedulingEnv:
    """
    Discrete-time satellite task scheduling environment.

    State representation (compact, report-friendly and easy to justify):
    - normalised current time
    - battery level
    - storage usage
    - mean urgency of top candidate tasks
    - mean priority of top candidate tasks
    - ratio of remaining tasks
    - number of currently feasible tasks

    Actions:
    - 0..K-1 : select one of the top-K candidate tasks presented by the environment
    - K      : downlink (clear storage when ground station available)
    - K+1    : idle/advance time by one step
    """

    def __init__(
        self,
        tasks: Sequence[SatelliteTask],
        horizon: int = 96,
        battery_capacity: float = 100.0,
        storage_capacity: float = 100.0,
        top_k_tasks: int = 5,
        downlink_windows: Optional[Sequence[Tuple[int, int]]] = None,
    ) -> None:
        self.original_tasks = [SatelliteTask(**asdict(t)) for t in tasks]
        self.horizon = horizon
        self.battery_capacity = battery_capacity
        self.storage_capacity = storage_capacity
        self.top_k_tasks = top_k_tasks
        self.downlink_windows = list(downlink_windows or [(18, 22), (46, 50), (74, 78)])

        self.tasks: List[SatelliteTask] = []
        self.completed_ids: set[int] = set()
        self.failed_ids: set[int] = set()
        self.time = 0
        self.battery = battery_capacity
        self.storage = 0.0
        self.total_reward = 0.0
        self.invalid_actions = 0
        self.schedule: List[ScheduleEntry] = []
        self.current_candidates: List[SatelliteTask] = []

    def clone(self) -> "SatelliteSchedulingEnv":
        return SatelliteSchedulingEnv(
            tasks=self.original_tasks,
            horizon=self.horizon,
            battery_capacity=self.battery_capacity,
            storage_capacity=self.storage_capacity,
            top_k_tasks=self.top_k_tasks,
            downlink_windows=self.downlink_windows,
        )

    def reset(self) -> np.ndarray:
        self.tasks = [SatelliteTask(**asdict(t)) for t in self.original_tasks]
        self.completed_ids = set()
        self.failed_ids = set()
        self.time = 0
        self.battery = self.battery_capacity
        self.storage = 0.0
        self.total_reward = 0.0
        self.invalid_actions = 0
        self.schedule = []
        self.current_candidates = []
        self._expire_old_tasks()
        return self._build_state()

    def is_done(self) -> bool:
        return self.time >= self.horizon or len(self.completed_ids) + len(self.failed_ids) >= len(self.tasks)

    def is_downlink_visible(self) -> bool:
        return any(start <= self.time <= end for start, end in self.downlink_windows)

    def _pending_tasks(self) -> List[SatelliteTask]:
        return [t for t in self.tasks if t.task_id not in self.completed_ids and t.task_id not in self.failed_ids]

    def _expire_old_tasks(self) -> None:
        for task in self._pending_tasks():
            if self.time > task.window_end:
                self.failed_ids.add(task.task_id)

    def _task_is_feasible_now(self, task: SatelliteTask) -> bool:
        if task.task_id in self.completed_ids or task.task_id in self.failed_ids:
            return False
        if not (task.window_start <= self.time <= task.window_end):
            return False
        if self.time + task.duration > self.horizon:
            return False
        if self.time + task.duration - 1 > task.window_end:
            return False
        if self.battery < task.energy_cost:
            return False
        if self.storage + task.storage_cost > self.storage_capacity:
            return False
        return True

    def _candidate_tasks(self) -> List[SatelliteTask]:
        pending = self._pending_tasks()
        scored: List[Tuple[float, SatelliteTask]] = []
        for task in pending:
            slack = task.window_end - self.time
            urgency = 1.0 / max(1, slack + 1)
            feasibility_bonus = 2.0 if self._task_is_feasible_now(task) else 0.0
            heuristic_score = (
                1.8 * task.priority
                + 0.12 * task.reward_value
                - 0.05 * task.energy_cost
                - 0.03 * task.storage_cost
                + 8.0 * urgency
                + feasibility_bonus
            )
            scored.append((heuristic_score, task))
        scored.sort(key=lambda x: x[0], reverse=True)
        return [task for _, task in scored[: self.top_k_tasks]]

    def _build_state(self) -> np.ndarray:
        self._expire_old_tasks()
        self.current_candidates = self._candidate_tasks()
        feasible_candidates = [t for t in self.current_candidates if self._task_is_feasible_now(t)]
        pending = self._pending_tasks()

        if self.current_candidates:
            urgencies = [1.0 / max(1, t.window_end - self.time + 1) for t in self.current_candidates]
            priorities = [t.priority / 5.0 for t in self.current_candidates]
            mean_urgency = float(np.mean(urgencies))
            mean_priority = float(np.mean(priorities))
        else:
            mean_urgency = 0.0
            mean_priority = 0.0

        state = np.array(
            [
                self.time / self.horizon,
                self.battery / self.battery_capacity,
                self.storage / self.storage_capacity,
                mean_urgency,
                mean_priority,
                len(pending) / max(1, len(self.tasks)),
                len(feasible_candidates) / max(1, self.top_k_tasks),
                1.0 if self.is_downlink_visible() else 0.0,
            ],
            dtype=np.float32,
        )
        return state

    def get_action_size(self) -> int:
        return self.top_k_tasks + 2

    def step(self, action: int, method_name: str = "DQN") -> Tuple[np.ndarray, float, bool, Dict[str, float]]:
        if self.is_done():
            return self._build_state(), 0.0, True, self.metrics()

        reward = 0.0

        # Select candidate task.
        if 0 <= action < self.top_k_tasks:
            if action >= len(self.current_candidates):
                reward -= 6.0
                self.invalid_actions += 1
                self.time += 1
            else:
                task = self.current_candidates[action]
                if self._task_is_feasible_now(task):
                    self.battery -= task.energy_cost
                    self.storage += task.storage_cost
                    start_time = self.time
                    self.time += task.duration
                    self.completed_ids.add(task.task_id)
                    reward += task.reward_value
                    reward += 1.5 * task.priority
                    reward -= 0.08 * task.energy_cost
                    self.schedule.append(
                        ScheduleEntry(
                            method=method_name,
                            task_id=task.task_id,
                            start_time=start_time,
                            end_time=self.time,
                            priority=task.priority,
                            reward_value=task.reward_value,
                            battery_after=round(self.battery, 2),
                            storage_after=round(self.storage, 2),
                        )
                    )
                else:
                    reward -= 7.0
                    self.invalid_actions += 1
                    self.time += 1

        # Downlink action.
        elif action == self.top_k_tasks:
            if self.is_downlink_visible() and self.storage > 0:
                cleared = self.storage
                self.storage = max(0.0, self.storage - min(50.0, self.storage))
                reward += 2.5 + 0.03 * cleared
            else:
                reward -= 2.5
                self.invalid_actions += 1
            self.time += 1

        # Idle action.
        else:
            reward -= 0.8
            self.time += 1

        # Battery recharge every time step.
        self.battery = min(self.battery_capacity, self.battery + 1.2)
        self._expire_old_tasks()

        # Small shaping signal to discourage large task backlogs.
        reward -= 0.02 * len(self._pending_tasks())
        self.total_reward += reward

        next_state = self._build_state()
        done = self.is_done()
        info = self.metrics()
        return next_state, float(reward), done, info

    def metrics(self) -> Dict[str, float]:
        total_tasks = max(1, len(self.tasks))
        completed_reward = sum(
            t.reward_value for t in self.tasks if t.task_id in self.completed_ids
        )
        return {
            "task_completion_ratio": len(self.completed_ids) / total_tasks,
            "total_priority_reward": completed_reward,
            "constraint_violation_count": float(self.invalid_actions),
            "battery_utilisation": 1.0 - (self.battery / self.battery_capacity),
            "storage_utilisation": self.storage / self.storage_capacity,
            "completed_tasks": float(len(self.completed_ids)),
            "pending_tasks": float(len(self._pending_tasks())),
            "episode_reward": float(self.total_reward),
        }

    def candidate_snapshot(self) -> List[Dict[str, float]]:
        rows: List[Dict[str, float]] = []
        for idx, task in enumerate(self.current_candidates):
            rows.append(
                {
                    "candidate_index": idx,
                    "task_id": task.task_id,
                    "window_start": task.window_start,
                    "window_end": task.window_end,
                    "duration": task.duration,
                    "priority": task.priority,
                    "energy_cost": task.energy_cost,
                    "storage_cost": task.storage_cost,
                    "reward_value": task.reward_value,
                    "feasible_now": self._task_is_feasible_now(task),
                }
            )
        return rows


# ============================================================
# Baseline schedulers
# ============================================================


class SchedulerBase:
    def __init__(self, name: str):
        self.name = name

    def choose_action(self, env: SatelliteSchedulingEnv, state: np.ndarray) -> int:
        raise NotImplementedError


class FCFSScheduler(SchedulerBase):
    def __init__(self) -> None:
        super().__init__("FCFS")

    def choose_action(self, env: SatelliteSchedulingEnv, state: np.ndarray) -> int:
        for idx, task in enumerate(env.current_candidates):
            if env._task_is_feasible_now(task):
                return idx
        if env.is_downlink_visible() and env.storage > 0:
            return env.top_k_tasks
        return env.top_k_tasks + 1


class HighestPriorityScheduler(SchedulerBase):
    def __init__(self) -> None:
        super().__init__("HighestPriority")

    def choose_action(self, env: SatelliteSchedulingEnv, state: np.ndarray) -> int:
        best_idx = None
        best_score = -1e9
        for idx, task in enumerate(env.current_candidates):
            if env._task_is_feasible_now(task):
                score = task.priority * 1000 - task.window_end + 0.1 * task.reward_value
                if score > best_score:
                    best_score = score
                    best_idx = idx
        if best_idx is not None:
            return best_idx
        if env.is_downlink_visible() and env.storage > 0:
            return env.top_k_tasks
        return env.top_k_tasks + 1


class EarliestDeadlineScheduler(SchedulerBase):
    def __init__(self) -> None:
        super().__init__("EarliestDeadline")

    def choose_action(self, env: SatelliteSchedulingEnv, state: np.ndarray) -> int:
        best_idx = None
        best_tuple = (10 ** 9, -10 ** 9)

        for idx, task in enumerate(env.current_candidates):
            if env._task_is_feasible_now(task):
                candidate_tuple = (task.window_end, -task.priority)
                if candidate_tuple < best_tuple:
                    best_tuple = candidate_tuple
                    best_idx = idx
        if best_idx is not None:
            return best_idx
        if env.is_downlink_visible() and env.storage > 0:
            return env.top_k_tasks
        return env.top_k_tasks + 1


class GreedyScoreScheduler(SchedulerBase):
    def __init__(self) -> None:
        super().__init__("GreedyScore")

    def choose_action(self, env: SatelliteSchedulingEnv, state: np.ndarray) -> int:
        best_idx = None
        best_score = -1e9
        for idx, task in enumerate(env.current_candidates):
            if env._task_is_feasible_now(task):
                slack = max(1, task.window_end - env.time + 1)
                score = (task.reward_value + 6 * task.priority) / (
                            0.8 * task.energy_cost + 1.2 * task.storage_cost + 0.5 * slack)
                if score > best_score:
                    best_score = score
                    best_idx = idx
        if best_idx is not None:
            return best_idx
        if env.is_downlink_visible() and env.storage > 0:
            return env.top_k_tasks
        return env.top_k_tasks + 1


# ============================================================
# DQN
# ============================================================


class ReplayBuffer:
    def __init__(self, capacity: int) -> None:
        self.capacity = capacity
        self.buffer: List[Tuple[np.ndarray, int, float, np.ndarray, float]] = []
        self.position = 0

    def push(self, state: np.ndarray, action: int, reward: float, next_state: np.ndarray, done: bool) -> None:
        item = (state, action, reward, next_state, float(done))
        if len(self.buffer) < self.capacity:
            self.buffer.append(item)
        else:
            self.buffer[self.position] = item
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = map(np.array, zip(*batch))
        return (
            torch.tensor(states, dtype=torch.float32),
            torch.tensor(actions, dtype=torch.long),
            torch.tensor(rewards, dtype=torch.float32),
            torch.tensor(next_states, dtype=torch.float32),
            torch.tensor(dones, dtype=torch.float32),
        )

    def __len__(self) -> int:
        return len(self.buffer)


class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int) -> None:
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class DQNScheduler(SchedulerBase):
    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        lr: float = 1e-3,
        gamma: float = 0.99,
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.05,
        epsilon_decay: float = 0.985,
        target_update: int = 10,
        buffer_capacity: int = 5000,
        batch_size: int = 64,
        device: Optional[str] = None,
    ) -> None:
        super().__init__("DQN")
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.target_update = target_update
        self.batch_size = batch_size
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))

        self.policy_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net = QNetwork(state_dim, action_dim).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=lr)
        self.loss_fn = nn.SmoothL1Loss()
        self.buffer = ReplayBuffer(buffer_capacity)
        self.train_steps = 0

    def choose_action(self, env: SatelliteSchedulingEnv, state: np.ndarray, greedy_only: bool = False) -> int:
        valid_actions = []

        for idx, task in enumerate(env.current_candidates):
            if env._task_is_feasible_now(task):
                valid_actions.append(idx)

        if env.is_downlink_visible() and env.storage > 0:
            valid_actions.append(env.top_k_tasks)

        # Idle is always valid
        valid_actions.append(env.top_k_tasks + 1)

        if (not greedy_only) and random.random() < self.epsilon:
            return random.choice(valid_actions)

        with torch.no_grad():
            state_tensor = torch.tensor(state, dtype=torch.float32, device=self.device).unsqueeze(0)
            q_values = self.policy_net(state_tensor).squeeze(0).cpu().numpy()

        masked_q = np.full_like(q_values, -1e9, dtype=np.float32)
        for a in valid_actions:
            masked_q[a] = q_values[a]

        return int(np.argmax(masked_q))

    def optimize(self) -> Optional[float]:
        if len(self.buffer) < self.batch_size:
            return None
        states, actions, rewards, next_states, dones = self.buffer.sample(self.batch_size)
        states = states.to(self.device)
        actions = actions.to(self.device)
        rewards = rewards.to(self.device)
        next_states = next_states.to(self.device)
        dones = dones.to(self.device)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_actions = self.policy_net(next_states).argmax(dim=1)
            next_q = self.target_net(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            target_q = rewards + self.gamma * next_q * (1 - dones)

        loss = self.loss_fn(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.train_steps += 1
        return float(loss.item())

    def update_target(self) -> None:
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)


# ============================================================
# Training and evaluation
# ============================================================


def run_scheduler_episode(env: SatelliteSchedulingEnv, scheduler: SchedulerBase, training: bool = False) -> Dict[str, object]:
    state = env.reset()
    rewards: List[float] = []
    losses: List[float] = []

    while not env.is_done():
        if isinstance(scheduler, DQNScheduler):
            action = scheduler.choose_action(env, state, greedy_only=not training)
            next_state, reward, done, info = env.step(action, method_name=scheduler.name)
            if training:
                scheduler.buffer.push(state, action, reward, next_state, done)
                loss = scheduler.optimize()
                if loss is not None:
                    losses.append(loss)
            state = next_state
        else:
            action = scheduler.choose_action(env, state)
            state, reward, done, info = env.step(action, method_name=scheduler.name)
        rewards.append(reward)

    result = env.metrics()
    result["schedule"] = env.schedule
    result["reward_trace"] = rewards
    result["loss_trace"] = losses
    return result


def train_dqn(
    tasks: Sequence[SatelliteTask],
    episodes: int,
    seed: int,
    output_dir: Path,
    horizon: int,
    battery_capacity: float,
    storage_capacity: float,
    top_k_tasks: int,
) -> Tuple[DQNScheduler, pd.DataFrame]:
    training_env = SatelliteSchedulingEnv(
        tasks=tasks,
        horizon=horizon,
        battery_capacity=battery_capacity,
        storage_capacity=storage_capacity,
        top_k_tasks=top_k_tasks,
    )
    state_dim = len(training_env.reset())
    action_dim = training_env.get_action_size()

    dqn = DQNScheduler(state_dim=state_dim, action_dim=action_dim)
    logs: List[Dict[str, float]] = []

    for episode in range(1, episodes + 1):
        env = training_env.clone()
        result = run_scheduler_episode(env, dqn, training=True)
        dqn.decay_epsilon()
        if episode % dqn.target_update == 0:
            dqn.update_target()

        avg_loss = float(np.mean(result["loss_trace"])) if result["loss_trace"] else math.nan
        logs.append(
            {
                "episode": episode,
                "episode_reward": result["episode_reward"],
                "task_completion_ratio": result["task_completion_ratio"],
                "constraint_violation_count": result["constraint_violation_count"],
                "epsilon": dqn.epsilon,
                "avg_loss": avg_loss,
            }
        )

        if episode % 20 == 0 or episode == 1 or episode == episodes:
            print(
                f"Episode {episode}/{episodes} | reward={result['episode_reward']:.2f} | "
                f"completion={result['task_completion_ratio']:.2f} | epsilon={dqn.epsilon:.3f}"
            )

    df = pd.DataFrame(logs)
    df.to_csv(output_dir / "training_log.csv", index=False)
    return dqn, df


def evaluate_methods(
    tasks: Sequence[SatelliteTask],
    dqn: DQNScheduler,
    n_eval_runs: int,
    horizon: int,
    battery_capacity: float,
    storage_capacity: float,
    top_k_tasks: int,
) -> Tuple[pd.DataFrame, Dict[str, List[ScheduleEntry]]]:
    methods: List[SchedulerBase] = [
        dqn,
        FCFSScheduler(),
        HighestPriorityScheduler(),
        EarliestDeadlineScheduler(),
        GreedyScoreScheduler(),
    ]

    rows: List[Dict[str, float]] = []
    schedule_examples: Dict[str, List[ScheduleEntry]] = {}

    for method in methods:
        for run_idx in range(n_eval_runs):
            env = SatelliteSchedulingEnv(
                tasks=tasks,
                horizon=horizon,
                battery_capacity=battery_capacity,
                storage_capacity=storage_capacity,
                top_k_tasks=top_k_tasks,
            )
            result = run_scheduler_episode(env, method, training=False)
            row = {k: v for k, v in result.items() if not isinstance(v, list)}
            row["method"] = method.name
            row["run"] = run_idx + 1
            rows.append(row)
            if run_idx == 0:
                schedule_examples[method.name] = result["schedule"]

    eval_df = pd.DataFrame(rows)
    return eval_df, schedule_examples


# ============================================================
# Visualisation
# ============================================================


def save_training_curve(training_df: pd.DataFrame, output_dir: Path) -> None:
    plt.figure(figsize=(10, 5))
    plt.plot(training_df["episode"], training_df["episode_reward"], label="Episode reward")
    if len(training_df) >= 5:
        rolling = training_df["episode_reward"].rolling(window=5, min_periods=1).mean()
        plt.plot(training_df["episode"], rolling, label="Rolling mean (5)")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("DQN training reward curve")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_dir / "reward_curve.png", dpi=200)
    plt.close()


def save_comparison_chart(eval_summary: pd.DataFrame, output_dir: Path, column: str, filename: str, title: str) -> None:
    plt.figure(figsize=(10, 5))
    x = np.arange(len(eval_summary))
    plt.bar(x, eval_summary[column])
    plt.xticks(x, eval_summary["method"], rotation=20)
    plt.ylabel(column)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


def save_schedule_plot(schedule: Sequence[ScheduleEntry], output_dir: Path, filename: str, title: str) -> None:
    plt.figure(figsize=(12, 4))
    if not schedule:
        plt.text(0.5, 0.5, "No scheduled tasks", ha="center", va="center")
        plt.axis("off")
    else:
        y = np.arange(len(schedule))
        starts = [s.start_time for s in schedule]
        durations = [max(1, s.end_time - s.start_time) for s in schedule]
        labels = [f"Task {s.task_id} (P{s.priority})" for s in schedule]
        plt.barh(y, durations, left=starts)
        plt.yticks(y, labels)
        plt.xlabel("Time step")
        plt.ylabel("Scheduled tasks")
        plt.title(title)
    plt.tight_layout()
    plt.savefig(output_dir / filename, dpi=200)
    plt.close()


# ============================================================
# Main pipeline
# ============================================================


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def export_schedule_csv(schedule_examples: Dict[str, List[ScheduleEntry]], output_dir: Path) -> None:
    rows: List[Dict[str, object]] = []
    for method, schedule in schedule_examples.items():
        for entry in schedule:
            row = asdict(entry)
            row["method"] = method
            rows.append(row)
    pd.DataFrame(rows).to_csv(output_dir / "sample_schedules.csv", index=False)


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Satellite task scheduling prototype with DQN and heuristic baselines."
    )
    # Final experiment configuration used for evaluation
    parser.add_argument("--tasks", type=int, default=60, help="Number of synthetic tasks to generate.")
    parser.add_argument("--episodes", type=int, default=300, help="Number of DQN training episodes.")
    parser.add_argument("--eval-runs", type=int, default=10, help="Number of evaluation episodes per method.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--horizon", type=int, default=96, help="Number of discrete time steps in one day.")
    parser.add_argument("--battery", type=float, default=85.0, help="Battery capacity.")
    parser.add_argument("--storage", type=float, default=45.0, help="Storage capacity.")
    parser.add_argument("--top-k", type=int, default=5, help="Number of candidate tasks exposed to the agent.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs_satellite_fyp",
        help="Directory where plots, CSV files and model weights are stored.",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    set_seed(args.seed)
    output_dir = Path(args.output_dir)
    ensure_dir(output_dir)

    # 1) Generate dataset.
    generator = TaskGenerator(horizon=args.horizon)
    tasks = generator.generate(n_tasks=args.tasks, seed=args.seed)
    task_df = generator.to_dataframe(tasks)
    task_df.to_csv(output_dir / "synthetic_tasks.csv", index=False)

    print(f"Generated {len(tasks)} tasks and saved dataset to {output_dir / 'synthetic_tasks.csv'}")

    # 2) Train DQN.
    dqn, training_df = train_dqn(
        tasks=tasks,
        episodes=args.episodes,
        seed=args.seed,
        output_dir=output_dir,
        horizon=args.horizon,
        battery_capacity=args.battery,
        storage_capacity=args.storage,
        top_k_tasks=args.top_k,
    )
    torch.save(dqn.policy_net.state_dict(), output_dir / "dqn_scheduler_weights.pt")

    # 3) Evaluate against baselines.
    eval_df, schedule_examples = evaluate_methods(
        tasks=tasks,
        dqn=dqn,
        n_eval_runs=args.eval_runs,
        horizon=args.horizon,
        battery_capacity=args.battery,
        storage_capacity=args.storage,
        top_k_tasks=args.top_k,
    )
    eval_df.to_csv(output_dir / "evaluation_runs.csv", index=False)

    summary = (
        eval_df.groupby("method", as_index=False)[
            [
                "task_completion_ratio",
                "total_priority_reward",
                "constraint_violation_count",
                "battery_utilisation",
                "storage_utilisation",
                "episode_reward",
            ]
        ]
        .mean()
        .sort_values(by="episode_reward", ascending=False)
    )
    summary.to_csv(output_dir / "evaluation_summary.csv", index=False)
    export_schedule_csv(schedule_examples, output_dir)

    # 4) Save plots.
    save_training_curve(training_df, output_dir)
    save_comparison_chart(
        summary,
        output_dir,
        column="episode_reward",
        filename="scheduler_comparison_reward.png",
        title="Mean episode reward by scheduling method",
    )
    save_comparison_chart(
        summary,
        output_dir,
        column="task_completion_ratio",
        filename="completion_rate_comparison.png",
        title="Task completion ratio by scheduling method",
    )
    best_method = summary.iloc[0]["method"]
    save_schedule_plot(
        schedule_examples.get(best_method, []),
        output_dir,
        filename="sample_schedule_timeline.png",
        title=f"Sample schedule timeline for {best_method}",
    )

    # 5) Brief console summary.
    print("\n=== Evaluation summary ===")
    print(summary.to_string(index=False))
    print(f"\nSaved all outputs to: {output_dir.resolve()}")
    print("Suggested report artefacts:")
    print("- reward_curve.png")
    print("- scheduler_comparison_reward.png")
    print("- completion_rate_comparison.png")
    print("- sample_schedule_timeline.png")


if __name__ == "__main__":
    main()
