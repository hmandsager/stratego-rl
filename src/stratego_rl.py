"""
A complete Python implementation for a simplified Stratego game environment,
its simulation, and RL training using PPO from Stable Baselines 3.

This code includes:
1. A custom Gym environment for Stratego.
2. Methods to set up the board, generate legal moves, and resolve battles.
3. An RL training loop that trains an agent (Player 1) against a random opponent.
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random
import os

# =============================================================================
# Constants and Helper Functions
# =============================================================================

# Board settings
BOARD_SIZE = 10
EMPTY = 0
LAKE = 100  # A special value representing a lake cell

# Define the fixed lake positions (0-indexed). In standard Stratego there are two lakes.
LAKE_POSITIONS = [(4, 2), (4, 3), (5, 2), (5, 3),
                  (4, 6), (4, 7), (5, 6), (5, 7)]

# Special values used to encode the two non-numeric pieces:
FLAG_VALUE = 1000  # We use this to encode the Flag.
BOMB_VALUE = 1001  # We use this to encode Bombs.

# The standard counts for each piece type in a full Stratego set:
PIECE_COUNTS = {
    'Flag': 1,
    'Bomb': 6,
    'Marshal': 1,
    'General': 1,
    'Colonel': 2,
    'Major': 3,
    'Captain': 4,
    'Lieutenant': 4,
    'Sergeant': 4,
    'Miner': 5,
    'Scout': 8,
    'Spy': 1
}

# The “rank” values for all movable (non-special) pieces.
# (Higher numbers defeat lower numbers except for special cases.)
PIECE_RANKS = {
    'Spy': 1,
    'Scout': 2,
    'Miner': 3,
    'Sergeant': 4,
    'Lieutenant': 5,
    'Captain': 6,
    'Major': 7,
    'Colonel': 8,
    'General': 9,
    'Marshal': 10
}

def get_piece_info(piece_value):
    """
    Given an encoded piece value, return a dictionary with details:
      - owner: 1 (agent / Player 1) or -1 (opponent / Player 2)
      - type: a string (e.g. "Scout", "Bomb", "Flag", etc.)
      - rank: an integer rank (if applicable; Bombs have no rank)
    
    Returns None for EMPTY or LAKE cells.
    """
    if piece_value == EMPTY or piece_value == LAKE:
        return None
    owner = 1 if piece_value > 0 else -1
    abs_val = abs(piece_value)
    if abs_val == FLAG_VALUE:
        return {'owner': owner, 'type': 'Flag', 'rank': 0}
    elif abs_val == BOMB_VALUE:
        return {'owner': owner, 'type': 'Bomb', 'rank': None}
    else:
        # For normal pieces, use the rank (which was encoded as a positive integer)
        for piece_type, rank in PIECE_RANKS.items():
            if rank == abs_val:
                return {'owner': owner, 'type': piece_type, 'rank': rank}
        return {'owner': owner, 'type': 'Unknown', 'rank': abs_val}

# =============================================================================
# The Stratego Gym Environment
# =============================================================================

class StrategoEnv(gym.Env):
    """
    A custom OpenAI Gym environment for a simplified version of Stratego.
    
    Game assumptions:
      - Full observability: all pieces are visible.
      - The board is 10x10 with fixed lake positions.
      - Player 1 (RL agent) deploys pieces in rows 6-9.
      - Player 2 (opponent) deploys pieces in rows 0-3.
      - Movement is orthogonal (up, down, left, right).
      - Most pieces move one cell; Scouts can move multiple cells.
      - Battles are resolved by comparing ranks (with special rules for bombs, flags,
        and the spy attacking the marshal).
      
    The agent (Player 1) takes an action when it is its turn. When it completes a move,
    the environment automatically simulates the opponent (Player 2) by making a random legal move.
    """
    
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        super(StrategoEnv, self).__init__()
        
        # The observation is the board state: a 10x10 grid of integers.
        # Each integer encodes a piece (its type and owner) or EMPTY/LAKE.
        self.observation_shape = (BOARD_SIZE, BOARD_SIZE)
        self.observation_space = spaces.Box(low=-BOMB_VALUE, high=BOMB_VALUE,
                                            shape=self.observation_shape, dtype=np.int32)
        
        # The action is represented by four coordinates: (from_x, from_y, to_x, to_y).
        # We use a MultiDiscrete space with each coordinate in [0, BOARD_SIZE - 1].
        self.action_space = spaces.MultiDiscrete([BOARD_SIZE, BOARD_SIZE, BOARD_SIZE, BOARD_SIZE])
        
        # Initialize the game state.
        self.board = None
        self.current_player = 1  # 1 for the agent (Player 1); -1 for opponent.
        self.done = False
        
        # For logging purposes.
        self.last_action = None
        #self.seed = 69
        self.reset()
    
    # def seed(self, val=69):
    #     return val
        
    def reset(self, seed=None, options=None):
        """
        Reset the game state: create an empty board, place lakes, and deploy both players’ pieces.
        Returns the initial board observation and an empty info dictionary.
        """
        # Create an empty board.
        self.board = np.zeros((BOARD_SIZE, BOARD_SIZE), dtype=np.int32)
        
        # Set fixed lake positions.
        for pos in LAKE_POSITIONS:
            self.board[pos] = LAKE
        
        # Randomly place pieces for each player.
        self._place_pieces_for_player(1)   # Player 1 (agent) in rows 6-9.
        self._place_pieces_for_player(-1)  # Player 2 (opponent) in rows 0-3.
        
        self.current_player = 1  # Agent starts.
        self.done = False
        self.last_action = None
        
        return self.board.copy(), {}
    
    def _place_pieces_for_player(self, player):
        """
        Deploy pieces for the given player.
        For player 1, use rows 6-9; for player -1, use rows 0-3.
        Pieces are randomly shuffled into the deployment area.
        """
        pieces = []
        for piece_type, count in PIECE_COUNTS.items():
            for _ in range(count):
                if piece_type == 'Flag':
                    piece_value = FLAG_VALUE if player == 1 else -FLAG_VALUE
                elif piece_type == 'Bomb':
                    piece_value = BOMB_VALUE if player == 1 else -BOMB_VALUE
                else:
                    # For normal pieces, encode as the piece’s rank.
                    piece_value = PIECE_RANKS[piece_type] if player == 1 else -PIECE_RANKS[piece_type]
                pieces.append(piece_value)
        
        random.shuffle(pieces)
        
        # Determine which rows are available for deployment.
        if player == 1:
            rows = list(range(6, 10))
        else:
            rows = list(range(0, 4))
        
        available_positions = [(r, c) for r in rows for c in range(BOARD_SIZE) if self.board[r, c] == EMPTY]
        
        # Place each piece in a random available position.
        for piece in pieces:
            if not available_positions:
                break  # This should not happen if counts match available cells.
            pos = random.choice(available_positions)
            self.board[pos] = piece
            available_positions.remove(pos)
    
    def step(self, action):
        """
        Execute one game step.
        
        Parameters:
          action: a tuple (from_x, from_y, to_x, to_y) representing the agent’s move.
          
        Returns:
          observation: the updated board state (a copy of the board array).
          reward: the reward for this move (win: +1, loss: -1, illegal move: -0.1, else 0).
          done: whether the game is over.
          info: auxiliary debugging information.
        """
        if self.done:
            return self.board.copy(), 0, self.done, False, {}
        
        reward = 0
        info = {}
        self.last_action = action
        
        # The agent (Player 1) should only act on its turn.
        if self.current_player != 1:
            # (This should not happen because we simulate opponent moves automatically.)
            return self.board.copy(), reward, self.done, False, info
        
        # Parse the action tuple.
        from_x, from_y, to_x, to_y = action
        
        # Retrieve the legal moves for the agent.
        legal_moves = self.get_legal_moves(self.current_player)
        if (from_x, from_y, to_x, to_y) not in legal_moves:
            # Illegal move: assign a small penalty and do not change the state.
            reward = -0.1
            return self.board.copy(), reward, self.done, False, {"illegal_move": True}
        
        # Execute the agent’s move.
        self._execute_move(from_x, from_y, to_x, to_y)
        
        # Check for win/loss conditions immediately after the agent’s move.
        if self.done:
            # (Typically, the agent’s win occurs here; losses are more likely after opponent moves.)
            return self.board.copy(), reward, self.done, info
        
        # Now simulate the opponent’s move(s) until it is again the agent’s turn or the game ends.
        while not self.done and self.current_player == -1:
            opponent_moves = self.get_legal_moves(self.current_player)
            if not opponent_moves:
                # Opponent has no legal moves; the agent wins.
                self.done = True
                reward = 1
                break
            # Choose a random legal move for the opponent.
            opp_move = random.choice(opponent_moves)
            self._execute_move(*opp_move)
        
        # If the game ended during the opponent’s move, assign a loss reward for the agent.
        if self.done:
            reward = -1
        
        return self.board.copy(), reward, self.done, info
    
    def _execute_move(self, from_x, from_y, to_x, to_y):
        """
        Execute a move from (from_x, from_y) to (to_x, to_y) for the current player.
        This method moves the piece, handles battles, and then switches the turn.
        """
        moving_piece = self.board[from_x, from_y]
        target_piece = self.board[to_x, to_y]
        
        # If the destination is empty, simply move the piece.
        if target_piece == EMPTY:
            self.board[to_x, to_y] = moving_piece
            self.board[from_x, from_y] = EMPTY
        else:
            # A battle occurs because the destination is occupied by an enemy piece.
            self._resolve_battle(from_x, from_y, to_x, to_y)
        
        # Check if the game has been won/lost.
        self._check_win_conditions()
        
        # Switch the turn (if the game is not yet over).
        if not self.done:
            self.current_player *= -1
    
    def _resolve_battle(self, from_x, from_y, to_x, to_y):
        """
        Resolve a battle when a piece moves into an enemy-occupied square.
        
        Battle rules implemented:
          - If the defender is a Flag, the attacker wins immediately.
          - If the defender is a Bomb, only a Miner can win.
          - A Spy attacking a Marshal wins (special rule).
          - Otherwise, compare piece ranks.
          - If the ranks are equal, both pieces are removed.
        """
        attacker_val = self.board[from_x, from_y]
        defender_val = self.board[to_x, to_y]
        
        attacker = get_piece_info(attacker_val)
        defender = get_piece_info(defender_val)
        
        # If the defender is the Flag, capture it and win the game.
        if defender['type'] == 'Flag':
            self.board[to_x, to_y] = attacker_val
            self.board[from_x, from_y] = EMPTY
            self.done = True
            return
        
        # If the defender is a Bomb, only a Miner can defuse it.
        if defender['type'] == 'Bomb':
            if attacker['type'] == 'Miner':
                self.board[to_x, to_y] = attacker_val
                self.board[from_x, from_y] = EMPTY
            else:
                # Attacker loses.
                self.board[from_x, from_y] = EMPTY
            return
        
        # Special rule: A Spy defeating a Marshal when attacking.
        if attacker['type'] == 'Spy' and defender['type'] == 'Marshal':
            self.board[to_x, to_y] = attacker_val
            self.board[from_x, from_y] = EMPTY
            return
        
        # Standard battle: compare ranks.
        if attacker['rank'] is not None and defender['rank'] is not None:
            if attacker['rank'] > defender['rank']:
                # Attacker wins: move attacker into the defender’s square.
                self.board[to_x, to_y] = attacker_val
                self.board[from_x, from_y] = EMPTY
            elif attacker['rank'] == defender['rank']:
                # Both pieces are removed.
                self.board[to_x, to_y] = EMPTY
                self.board[from_x, from_y] = EMPTY
            else:
                # Attacker loses.
                self.board[from_x, from_y] = EMPTY
        else:
            # Fallback if ranks are somehow not comparable.
            self.board[from_x, from_y] = EMPTY
    
    def _check_win_conditions(self):
        """
        Check whether a win/loss condition has been met:
          - If a player’s flag is no longer on the board, that player loses.
          - Optionally, if the current player has no legal moves, the game ends.
        """
        # Check that Player 1’s flag is present.
        if not np.any(self.board == FLAG_VALUE):
            self.done = True
            return
        # Check that Player 2’s flag is present.
        if not np.any(self.board == -FLAG_VALUE):
            self.done = True
            return
        
        # Optionally, if the current player has no legal moves, end the game.
        if not self.get_legal_moves(self.current_player):
            self.done = True
    
    def get_legal_moves(self, player):
        """
        Generate and return a list of legal moves for the specified player.
        
        Each move is a tuple: (from_x, from_y, to_x, to_y)
        
        Movement rules:
          - Only movable pieces (not Flag or Bomb) may move.
          - Non-Scout pieces move one space orthogonally.
          - Scouts can move any number of spaces in a straight line until blocked.
          - A piece may move into an enemy-occupied square (to initiate a battle) but
            cannot move into a square occupied by a friendly piece or a lake.
        """
        moves = []
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # up, down, left, right
        
        for x in range(BOARD_SIZE):
            for y in range(BOARD_SIZE):
                piece = self.board[x, y]
                if piece == EMPTY or piece == LAKE:
                    continue
                # Check if this piece belongs to the current player.
                if (piece > 0 and player == 1) or (piece < 0 and player == -1):
                    info = get_piece_info(piece)
                    # Skip pieces that cannot move.
                    if info['type'] in ['Flag', 'Bomb']:
                        continue
                    
                    # Try each direction.
                    for dx, dy in directions:
                        nx = x + dx
                        ny = y + dy
                        # For Scouts, allow moving several squares.
                        if info['type'] == 'Scout':
                            while 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                                if self.board[nx, ny] == LAKE:
                                    break  # Cannot move into a lake.
                                if self.board[nx, ny] == EMPTY:
                                    moves.append((x, y, nx, ny))
                                    nx += dx
                                    ny += dy
                                else:
                                    # If occupied, you may be able to attack an enemy piece.
                                    target_info = get_piece_info(self.board[nx, ny])
                                    if target_info and target_info['owner'] != player:
                                        moves.append((x, y, nx, ny))
                                    break
                        else:
                            # Non-Scout pieces move only one square.
                            if 0 <= nx < BOARD_SIZE and 0 <= ny < BOARD_SIZE:
                                if self.board[nx, ny] == LAKE:
                                    continue
                                if self.board[nx, ny] == EMPTY:
                                    moves.append((x, y, nx, ny))
                                else:
                                    target_info = get_piece_info(self.board[nx, ny])
                                    if target_info and target_info['owner'] != player:
                                        moves.append((x, y, nx, ny))
        return moves
    
    def render(self, mode='human'):
        """
        Render the board to the console in a human-readable format.
        Displays the board grid, piece abbreviations, and indicates the current player.
        """
        def piece_str(val):
            return self._piece_to_str(val)
        
        print("Current board:")
        for i in range(BOARD_SIZE):
            row = ""
            for j in range(BOARD_SIZE):
                row += f"{piece_str(self.board[i, j]):>5} "
            print(row)
        print(f"Current player: {'Agent (1)' if self.current_player == 1 else 'Opponent (-1)'}")
        if self.last_action:
            print(f"Last action: {self.last_action}")
        print("\n")
    
    def _piece_to_str(self, val):
        """
        Convert an encoded piece value to a short string for printing.
        Uses the following format:
          - For Player 1, the piece string starts with 'A'
          - For Player 2, it starts with 'O'
          - Special pieces (Flag, Bomb) are abbreviated as F and B.
          - Other pieces are shown by their rank.
        """
        if val == EMPTY:
            return "."
        if val == LAKE:
            return "Lake"
        info = get_piece_info(val)
        if info is None:
            return "?"
        owner = "A" if info['owner'] == 1 else "O"
        if info['type'] == 'Flag':
            return owner + "F"
        elif info['type'] == 'Bomb':
            return owner + "B"
        else:
            return owner + str(info['rank'])

# =============================================================================
# RL Training and Simulation Code Using Stable Baselines 3
# =============================================================================

# Import PPO and the environment checker from Stable Baselines 3.
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

def train_agent(total_timesteps=50000):
    """
    Create an instance of the Stratego environment, check it,
    and train an RL agent (using PPO) for the given number of timesteps.
    
    The trained model is saved to disk.
    """
    env = StrategoEnv()
    
    # (Optional) Check that the environment follows the Gym API.
    check_env(env, warn=True)
    
    # Create a PPO model using a multilayer perceptron (MLP) policy.
    model = PPO("MlpPolicy", env, verbose=1)
    
    # Train the model.
    model.learn(total_timesteps=total_timesteps)
    
    # Save the model.
    model.save("stratego_ppo")
    print("Model saved as 'stratego_ppo'")
    
    return model

def run_simulation(model, episodes=3):
    """
    Run a few episodes of the game using the trained model.
    At each step, the board is rendered and the agent’s move is determined.
    
    If the agent’s chosen move is illegal (which can happen early in training),
    a random legal move is chosen instead.
    """
    env = StrategoEnv()
    for ep in range(episodes):
        obs = env.reset()
        done = False
        print(f"--- Episode {ep + 1} ---")
        while not done:
            env.render()
            # Retrieve legal moves for the agent.
            legal_moves = env.get_legal_moves(1)
            if not legal_moves:
                print("No legal moves available for the agent!")
                break
            # The model predicts an action based on the current observation.
            action, _ = model.predict(obs)
            # If the predicted action is not legal, pick one at random.
            if tuple(action) not in legal_moves:
                action = random.choice(legal_moves)
            obs, reward, done, info = env.step(action)
        env.render()
        if reward > 0:
            print("Agent wins!")
        elif reward < 0:
            print("Agent loses!")
        else:
            print("Draw!")
        print("\n")

# =============================================================================
# Main Entry Point
# =============================================================================

if __name__ == "__main__":
    # Train the RL agent. Adjust the total timesteps as needed.
    model = train_agent(total_timesteps=50000)
    
    # Run simulation episodes to see the trained agent in action.
    run_simulation(model, episodes=3)
