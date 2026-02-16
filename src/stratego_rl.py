"""
Stratego RL training with action-masked PPO, CNN observations, reward shaping,
and self-play.

Implements:
- 25-channel (12 per player + 1 lake) CNN observation space on a 10x10 board
- Pre-computed geometric move table with flat Discrete action space + masking
- BattleResult-based reward shaping (rank * CAPTURE_REWARD_SCALE)
- Self-play via SelfPlayCallback (snapshots + opponent swaps)
- StrategoCNN feature extractor (3 conv layers → 256-dim)
- MaskablePPO from sb3-contrib
"""

import copy
import os
import random
from collections import namedtuple
from typing import Optional

import gymnasium as gym
import numpy as np
import torch as th
from gymnasium import spaces
from torch import nn

from sb3_contrib import MaskablePPO
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

# =============================================================================
# Constants
# =============================================================================

BOARD_SIZE = 10
EMPTY = 0
LAKE = 100

LAKE_POSITIONS = [
    (4, 2), (4, 3), (5, 2), (5, 3),
    (4, 6), (4, 7), (5, 6), (5, 7),
]
LAKE_POSITIONS_SET = frozenset(LAKE_POSITIONS)

FLAG_VALUE = 1000
BOMB_VALUE = 1001

PIECE_COUNTS = {
    "Flag": 1,
    "Bomb": 6,
    "Marshal": 1,
    "General": 1,
    "Colonel": 2,
    "Major": 3,
    "Captain": 4,
    "Lieutenant": 4,
    "Sergeant": 4,
    "Miner": 5,
    "Scout": 8,
    "Spy": 1,
}

PIECE_RANKS = {
    "Spy": 1,
    "Scout": 2,
    "Miner": 3,
    "Sergeant": 4,
    "Lieutenant": 5,
    "Captain": 6,
    "Major": 7,
    "Colonel": 8,
    "General": 9,
    "Marshal": 10,
}

# 12 piece types for channel encoding (order matters for indexing)
PIECE_TYPES = [
    "Flag", "Spy", "Scout", "Miner", "Sergeant", "Lieutenant",
    "Captain", "Major", "Colonel", "General", "Marshal", "Bomb",
]
PIECE_TYPE_TO_CHANNEL = {t: i for i, t in enumerate(PIECE_TYPES)}

# Observation: 12 channels for agent pieces, 12 for opponent, 1 for lakes
NUM_CHANNELS = 25

CAPTURE_REWARD_SCALE = 0.01

BattleResult = namedtuple(
    "BattleResult",
    ["attacker_type", "defender_type", "attacker_rank", "defender_rank", "outcome"],
)
# outcome: "attacker_wins", "defender_wins", "both_die"

DIRECTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

# =============================================================================
# Pre-compute ALL_MOVES: every geometrically valid (from, to) on a 10x10 board
# =============================================================================


def _precompute_moves():
    """Build a list of all geometrically possible moves (no piece/state checks).

    Includes:
    - Single-step orthogonal moves for all cells (non-lake → non-lake)
    - Multi-step scout moves along straight lines (non-lake → non-lake,
      no lake in between -- but we only check geometry here; legality is
      checked at runtime via action_masks).
    """
    moves = []
    move_index = {}

    for r in range(BOARD_SIZE):
        for c in range(BOARD_SIZE):
            if (r, c) in LAKE_POSITIONS_SET:
                continue
            for dr, dc in DIRECTIONS:
                # Walk along the direction (distance 1..8 covers the board)
                for dist in range(1, BOARD_SIZE):
                    nr, nc = r + dr * dist, c + dc * dist
                    if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                        break
                    if (nr, nc) in LAKE_POSITIONS_SET:
                        break  # can't pass through or land on lake
                    move = (r, c, nr, nc)
                    if move not in move_index:
                        move_index[move] = len(moves)
                        moves.append(move)

    return moves, move_index


ALL_MOVES, MOVE_TO_INDEX = _precompute_moves()
ACTION_SPACE_SIZE = len(ALL_MOVES)


# =============================================================================
# Helper: piece value ↔ info
# =============================================================================


def get_piece_info(piece_value):
    if piece_value == EMPTY or piece_value == LAKE:
        return None
    owner = 1 if piece_value > 0 else -1
    abs_val = abs(piece_value)
    if abs_val == FLAG_VALUE:
        return {"owner": owner, "type": "Flag", "rank": 0}
    elif abs_val == BOMB_VALUE:
        return {"owner": owner, "type": "Bomb", "rank": 0}
    else:
        for piece_type, rank in PIECE_RANKS.items():
            if rank == abs_val:
                return {"owner": owner, "type": piece_type, "rank": rank}
        return {"owner": owner, "type": "Unknown", "rank": abs_val}


# =============================================================================
# Environment
# =============================================================================


class StrategoEnv(gym.Env):
    """Stratego environment with CNN observations and action masking.

    The agent is Player 1 (positive piece values, rows 6-9).
    The opponent is Player -1 (negative piece values, rows 0-3).
    After the agent acts, the opponent auto-moves (random or learned policy).
    """

    metadata = {"render_modes": ["human"]}

    def __init__(self, opponent_policy=None, render_mode=None):
        super().__init__()

        self.observation_space = spaces.Box(
            low=0.0, high=1.0, shape=(NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32
        )
        self.action_space = spaces.Discrete(ACTION_SPACE_SIZE)

        self.opponent_policy = opponent_policy
        self.render_mode = render_mode

        self.board = None
        self.current_player = 1
        self.done = False

    # ------------------------------------------------------------------
    # Opponent policy setter (for self-play)
    # ------------------------------------------------------------------

    def set_opponent_policy(self, policy):
        self.opponent_policy = policy

    # ------------------------------------------------------------------
    # Observation helpers
    # ------------------------------------------------------------------

    def _get_obs(self):
        """Build a 25-channel binary tensor from self.board.

        Channels 0-11:  agent (player 1) piece types
        Channels 12-23: opponent (player -1) piece types
        Channel 24:     lake mask
        """
        obs = np.zeros((NUM_CHANNELS, BOARD_SIZE, BOARD_SIZE), dtype=np.float32)
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                val = self.board[r, c]
                if val == LAKE:
                    obs[24, r, c] = 1.0
                elif val != EMPTY:
                    info = get_piece_info(val)
                    if info is None:
                        continue
                    ch = PIECE_TYPE_TO_CHANNEL.get(info["type"])
                    if ch is None:
                        continue
                    if info["owner"] == 1:
                        obs[ch, r, c] = 1.0
                    else:
                        obs[12 + ch, r, c] = 1.0
        return obs

    def _get_obs_for_opponent(self):
        """Observation from the opponent's perspective (swap agent/opponent channels)."""
        obs = self._get_obs()
        agent_channels = obs[:12].copy()
        obs[:12] = obs[12:24]
        obs[12:24] = agent_channels
        return obs

    # ------------------------------------------------------------------
    # Action masks
    # ------------------------------------------------------------------

    def action_masks(self):
        """Boolean mask over ALL_MOVES: True where the move is legal for the agent."""
        return self._get_action_masks_for_player(1)

    def _get_action_masks_for_player(self, player):
        mask = np.zeros(ACTION_SPACE_SIZE, dtype=bool)
        legal = self._get_legal_moves(player)
        for move in legal:
            idx = MOVE_TO_INDEX.get(move)
            if idx is not None:
                mask[idx] = True
        return mask

    # ------------------------------------------------------------------
    # Legal move generation
    # ------------------------------------------------------------------

    def _get_legal_moves(self, player):
        """Return list of (from_r, from_c, to_r, to_c) tuples for `player`."""
        moves = []
        for r in range(BOARD_SIZE):
            for c in range(BOARD_SIZE):
                piece = self.board[r, c]
                if piece == EMPTY or piece == LAKE:
                    continue
                if (piece > 0 and player != 1) or (piece < 0 and player != -1):
                    continue
                info = get_piece_info(piece)
                if info["type"] in ("Flag", "Bomb"):
                    continue

                is_scout = info["type"] == "Scout"
                for dr, dc in DIRECTIONS:
                    max_dist = BOARD_SIZE if is_scout else 1
                    for dist in range(1, max_dist + 1):
                        nr, nc = r + dr * dist, c + dc * dist
                        if not (0 <= nr < BOARD_SIZE and 0 <= nc < BOARD_SIZE):
                            break
                        target = self.board[nr, nc]
                        if target == LAKE:
                            break
                        if target == EMPTY:
                            moves.append((r, c, nr, nc))
                        else:
                            target_info = get_piece_info(target)
                            if target_info and target_info["owner"] != player:
                                moves.append((r, c, nr, nc))
                            break  # blocked by any piece (friendly or enemy)
        return moves

    # Keep public alias for compatibility
    def get_legal_moves(self, player):
        return self._get_legal_moves(player)

    # ------------------------------------------------------------------
    # Piece placement
    # ------------------------------------------------------------------

    def _place_pieces_for_player(self, player):
        pieces = []
        for piece_type, count in PIECE_COUNTS.items():
            for _ in range(count):
                if piece_type == "Flag":
                    pieces.append(FLAG_VALUE if player == 1 else -FLAG_VALUE)
                elif piece_type == "Bomb":
                    pieces.append(BOMB_VALUE if player == 1 else -BOMB_VALUE)
                else:
                    rank = PIECE_RANKS[piece_type]
                    pieces.append(rank if player == 1 else -rank)

        random.shuffle(pieces)

        rows = list(range(6, 10)) if player == 1 else list(range(0, 4))
        positions = [
            (r, c)
            for r in rows
            for c in range(BOARD_SIZE)
            if self.board[r, c] == EMPTY
        ]

        for piece, pos in zip(pieces, positions):
            self.board[pos] = piece

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        for pos in LAKE_POSITIONS:
            self.board[pos] = LAKE
        self._place_pieces_for_player(1)
        self._place_pieces_for_player(-1)
        self.current_player = 1
        self.done = False
        return self._get_obs(), {}

    # ------------------------------------------------------------------
    # Battle resolution → BattleResult
    # ------------------------------------------------------------------

    def _resolve_battle(self, from_r, from_c, to_r, to_c):
        """Resolve combat and update the board. Returns a BattleResult."""
        attacker_val = self.board[from_r, from_c]
        defender_val = self.board[to_r, to_c]
        attacker = get_piece_info(attacker_val)
        defender = get_piece_info(defender_val)

        a_type, d_type = attacker["type"], defender["type"]
        a_rank, d_rank = attacker["rank"], defender["rank"]

        # Flag captured → attacker wins, game over
        if d_type == "Flag":
            self.board[to_r, to_c] = attacker_val
            self.board[from_r, from_c] = EMPTY
            self.done = True
            return BattleResult(a_type, d_type, a_rank, d_rank, "attacker_wins")

        # Bomb: only Miner defuses
        if d_type == "Bomb":
            if a_type == "Miner":
                self.board[to_r, to_c] = attacker_val
                self.board[from_r, from_c] = EMPTY
                return BattleResult(a_type, d_type, a_rank, d_rank, "attacker_wins")
            else:
                self.board[from_r, from_c] = EMPTY
                return BattleResult(a_type, d_type, a_rank, d_rank, "defender_wins")

        # Spy attacks Marshal
        if a_type == "Spy" and d_type == "Marshal":
            self.board[to_r, to_c] = attacker_val
            self.board[from_r, from_c] = EMPTY
            return BattleResult(a_type, d_type, a_rank, d_rank, "attacker_wins")

        # Standard rank comparison
        if a_rank > d_rank:
            self.board[to_r, to_c] = attacker_val
            self.board[from_r, from_c] = EMPTY
            return BattleResult(a_type, d_type, a_rank, d_rank, "attacker_wins")
        elif a_rank == d_rank:
            self.board[to_r, to_c] = EMPTY
            self.board[from_r, from_c] = EMPTY
            return BattleResult(a_type, d_type, a_rank, d_rank, "both_die")
        else:
            self.board[from_r, from_c] = EMPTY
            return BattleResult(a_type, d_type, a_rank, d_rank, "defender_wins")

    # ------------------------------------------------------------------
    # Move execution → Optional[BattleResult]
    # ------------------------------------------------------------------

    def _execute_move(self, from_r, from_c, to_r, to_c):
        """Execute a move. Returns BattleResult if combat occurred, else None."""
        target = self.board[to_r, to_c]
        if target == EMPTY:
            self.board[to_r, to_c] = self.board[from_r, from_c]
            self.board[from_r, from_c] = EMPTY
            self._check_win_conditions()
            if not self.done:
                self.current_player *= -1
            return None
        else:
            result = self._resolve_battle(from_r, from_c, to_r, to_c)
            self._check_win_conditions()
            if not self.done:
                self.current_player *= -1
            return result

    # ------------------------------------------------------------------
    # Win-condition check
    # ------------------------------------------------------------------

    def _check_win_conditions(self):
        if not np.any(self.board == FLAG_VALUE):
            self.done = True
        elif not np.any(self.board == -FLAG_VALUE):
            self.done = True

    def _terminal_reward_for_agent(self):
        """Return +1 if agent won, -1 if agent lost."""
        agent_flag = np.any(self.board == FLAG_VALUE)
        opp_flag = np.any(self.board == -FLAG_VALUE)
        if not opp_flag:
            return 1.0  # opponent flag gone → agent wins
        if not agent_flag:
            return -1.0  # agent flag gone → agent loses
        # Shouldn't reach here in a terminal state, but guard anyway
        return 0.0

    # ------------------------------------------------------------------
    # Reward shaping from BattleResult
    # ------------------------------------------------------------------

    @staticmethod
    def _compute_capture_reward(battle_result, agent_is_attacker):
        """Small shaped reward based on piece ranks involved in combat."""
        if battle_result is None:
            return 0.0

        outcome = battle_result.outcome
        d_rank = battle_result.defender_rank or 0
        a_rank = battle_result.attacker_rank or 0

        if agent_is_attacker:
            if outcome == "attacker_wins":
                return d_rank * CAPTURE_REWARD_SCALE
            elif outcome == "defender_wins":
                return -a_rank * CAPTURE_REWARD_SCALE
            else:  # both_die
                return (d_rank - a_rank) * CAPTURE_REWARD_SCALE
        else:
            # Agent is defender
            if outcome == "attacker_wins":
                return -d_rank * CAPTURE_REWARD_SCALE
            elif outcome == "defender_wins":
                return a_rank * CAPTURE_REWARD_SCALE
            else:
                return (a_rank - d_rank) * CAPTURE_REWARD_SCALE

    # ------------------------------------------------------------------
    # Step
    # ------------------------------------------------------------------

    def step(self, action):
        if self.done:
            return self._get_obs(), 0.0, True, False, {}

        reward = 0.0
        info = {}

        # Decode flat action
        from_r, from_c, to_r, to_c = ALL_MOVES[action]

        # Validate
        legal = self._get_legal_moves(1)
        if (from_r, from_c, to_r, to_c) not in legal:
            # With action masking this shouldn't happen, but handle gracefully
            return self._get_obs(), -0.1, False, False, {"illegal_move": True}

        # Execute agent move
        battle = self._execute_move(from_r, from_c, to_r, to_c)
        reward += self._compute_capture_reward(battle, agent_is_attacker=True)

        if self.done:
            reward += self._terminal_reward_for_agent()
            return self._get_obs(), reward, True, False, info

        # --- Opponent turn ---
        opp_moves = self._get_legal_moves(-1)
        if not opp_moves:
            self.done = True
            reward += 1.0  # opponent can't move → agent wins
            return self._get_obs(), reward, True, False, info

        # Choose opponent action
        if self.opponent_policy is not None:
            opp_obs = self._get_obs_for_opponent()
            opp_mask = self._get_action_masks_for_player(-1)
            if opp_mask.any():
                opp_action, _ = self.opponent_policy.predict(
                    opp_obs, action_masks=opp_mask, deterministic=False
                )
                opp_move_tuple = ALL_MOVES[int(opp_action)]
                # Validate the policy's move is actually legal
                if opp_move_tuple not in opp_moves:
                    opp_move_tuple = random.choice(opp_moves)
            else:
                opp_move_tuple = random.choice(opp_moves)
        else:
            opp_move_tuple = random.choice(opp_moves)

        opp_battle = self._execute_move(*opp_move_tuple)
        reward += self._compute_capture_reward(opp_battle, agent_is_attacker=False)

        if self.done:
            reward += self._terminal_reward_for_agent()
            return self._get_obs(), reward, True, False, info

        return self._get_obs(), reward, False, False, info

    # ------------------------------------------------------------------
    # Render
    # ------------------------------------------------------------------

    def render(self):
        if self.render_mode != "human":
            return

        def piece_str(val):
            if val == EMPTY:
                return "."
            if val == LAKE:
                return "~"
            info = get_piece_info(val)
            if info is None:
                return "?"
            prefix = "A" if info["owner"] == 1 else "O"
            if info["type"] == "Flag":
                return prefix + "F"
            if info["type"] == "Bomb":
                return prefix + "B"
            return prefix + str(info["rank"])

        print("Current board:")
        for i in range(BOARD_SIZE):
            row = " ".join(f"{piece_str(self.board[i, j]):>4}" for j in range(BOARD_SIZE))
            print(row)
        player_label = "Agent (1)" if self.current_player == 1 else "Opponent (-1)"
        print(f"Current player: {player_label}\n")


# =============================================================================
# Action masking wrapper (for sb3-contrib)
# =============================================================================


def mask_fn(env: StrategoEnv) -> np.ndarray:
    return env.action_masks()


# =============================================================================
# CNN Feature Extractor
# =============================================================================


class StrategoCNN(BaseFeaturesExtractor):
    """3-layer CNN for the 25×10×10 observation space → 256-dim features."""

    def __init__(self, observation_space: spaces.Box, features_dim: int = 256):
        super().__init__(observation_space, features_dim)
        n_channels = observation_space.shape[0]  # 25

        self.cnn = nn.Sequential(
            nn.Conv2d(n_channels, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Flatten(),
        )

        # Compute the flattened size
        with th.no_grad():
            sample = th.zeros(1, *observation_space.shape)
            n_flat = self.cnn(sample).shape[1]

        self.linear = nn.Sequential(
            nn.Linear(n_flat, features_dim),
            nn.ReLU(),
        )

    def forward(self, observations: th.Tensor) -> th.Tensor:
        return self.linear(self.cnn(observations))


# =============================================================================
# Self-play Callback
# =============================================================================


class SelfPlayCallback(BaseCallback):
    """Periodically snapshot the current model and swap the opponent policy.

    Every `swap_interval` steps, with 50% probability assign a past snapshot
    as the opponent (self-play) and 50% revert to random.
    """

    def __init__(
        self,
        save_interval: int = 50_000,
        swap_interval: int = 25_000,
        snapshot_dir: str = "selfplay_snapshots",
        verbose: int = 0,
    ):
        super().__init__(verbose)
        self.save_interval = save_interval
        self.swap_interval = swap_interval
        self.snapshot_dir = snapshot_dir
        self.snapshots: list[str] = []

    def _init_callback(self) -> None:
        os.makedirs(self.snapshot_dir, exist_ok=True)

    def _on_step(self) -> bool:
        # Save snapshot
        if self.num_timesteps % self.save_interval == 0 and self.num_timesteps > 0:
            path = os.path.join(
                self.snapshot_dir, f"snapshot_{self.num_timesteps}"
            )
            self.model.save(path)
            self.snapshots.append(path)
            if self.verbose:
                print(f"[SelfPlay] Saved snapshot at {self.num_timesteps} steps")

        # Swap opponent
        if self.num_timesteps % self.swap_interval == 0 and self.num_timesteps > 0:
            env = self.training_env.envs[0]
            # Unwrap to get the base StrategoEnv
            base_env = env.unwrapped if hasattr(env, "unwrapped") else env

            if self.snapshots and random.random() < 0.5:
                snap = random.choice(self.snapshots)
                opponent = MaskablePPO.load(snap)
                base_env.set_opponent_policy(opponent)
                if self.verbose:
                    print(f"[SelfPlay] Opponent ← snapshot {snap}")
            else:
                base_env.set_opponent_policy(None)
                if self.verbose:
                    print("[SelfPlay] Opponent ← random")

        return True


# =============================================================================
# Training
# =============================================================================


def train_agent(total_timesteps: int = 1_000_000, save_path: str = "stratego_ppo"):
    """Train with MaskablePPO + CNN + self-play."""
    env = StrategoEnv()
    env = ActionMasker(env, mask_fn)

    policy_kwargs = dict(
        features_extractor_class=StrategoCNN,
        features_extractor_kwargs=dict(features_dim=256),
        net_arch=dict(pi=[256], vf=[256]),
    )

    model = MaskablePPO(
        "CnnPolicy",
        env,
        policy_kwargs=policy_kwargs,
        verbose=1,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=64,
        n_epochs=10,
        gamma=0.99,
        tensorboard_log="./stratego_tb/",
    )

    callback = SelfPlayCallback(
        save_interval=50_000,
        swap_interval=25_000,
        verbose=1,
    )

    model.learn(total_timesteps=total_timesteps, callback=callback)
    model.save(save_path)
    print(f"Model saved as '{save_path}'")
    return model


# =============================================================================
# Simulation
# =============================================================================


def run_simulation(model, episodes: int = 3):
    """Run evaluation episodes with the trained model."""
    env = StrategoEnv(render_mode="human")

    for ep in range(episodes):
        obs, _ = env.reset()
        done = False
        total_reward = 0.0
        print(f"--- Episode {ep + 1} ---")

        while not done:
            env.render()
            masks = env.action_masks()
            if not masks.any():
                print("No legal moves for agent!")
                break
            action, _ = model.predict(obs, action_masks=masks, deterministic=True)
            obs, reward, done, truncated, info = env.step(int(action))
            total_reward += reward

        env.render()
        if total_reward > 0:
            print("Agent wins!")
        elif total_reward < 0:
            print("Agent loses!")
        else:
            print("Draw!")
        print(f"Total reward: {total_reward:.3f}\n")


# =============================================================================
# Main
# =============================================================================

if __name__ == "__main__":
    model = train_agent(total_timesteps=1_000_000)
    run_simulation(model, episodes=3)
