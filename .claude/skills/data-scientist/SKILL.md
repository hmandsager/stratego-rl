---
name: rl-modeling
description: Expert-level assistance for data scientists building reinforcement learning models. Use when users need help with RL architecture design, training strategies, debugging convergence issues, hyperparameter tuning, environment design, state/action space engineering, or implementing specific RL algorithms (DQN, PPO, A3C, SAC, etc.). Also use for game AI, robotics control, recommendation systems, or any sequential decision-making problems.
---

# Reinforcement Learning Model Development

This skill provides expert guidance for building, training, debugging, and deploying reinforcement learning models. It assumes the user has strong ML fundamentals and focuses on RL-specific challenges and best practices.

## When to Use This Skill

Trigger this skill when the user:
- Designs or implements RL algorithms (value-based, policy-based, actor-critic, model-based)
- Debugs training instabilities, convergence failures, or reward hacking
- Optimizes hyperparameters or training strategies
- Builds custom environments or wraps existing ones
- Implements state/action space representations
- Works on imperfect information games (poker, Stratego, partial observability)
- Develops multi-agent or hierarchical RL systems
- Handles sparse rewards, credit assignment, or exploration challenges
- Deploys RL models to production or runs large-scale experiments

## Core Principles

### 1. Diagnose Before Prescribing

RL failures have multiple root causes. Before suggesting solutions, identify the actual problem:

**Training instability symptoms:**
- Exploding/vanishing gradients → Check learning rates, network architecture, reward scaling
- Policy collapse → Examine entropy regularization, exploration schedule
- Catastrophic forgetting → Review replay buffer, target network updates, batch sampling

**Convergence issues:**
- No learning → Verify reward signal, check gradient flow, inspect action distributions
- Plateauing → Consider curriculum learning, reward shaping, architecture capacity
- Oscillating performance → Adjust target network frequency, learning rate decay

**Always ask for diagnostics first**: training curves, episode returns over time, action/value distributions, gradient norms.

### 2. Algorithm Selection Hierarchy

Match algorithm to problem structure:

```
Discrete action space + Full observability
  └─> Simple: DQN, Rainbow DQN
  └─> Complex: PPO, A2C/A3C

Continuous action space
  └─> Model-free: SAC, TD3, PPO
  └─> Sample efficient: DDPG, model-based (PETS, Dreamer)

Partial observability
  └─> LSTM/GRU policies + DRQN
  └─> Belief state representations
  └─> Memory-augmented networks

Multi-agent
  └─> Independent learners (QMIX, VDN)
  └─> Centralized training, decentralized execution (MADDPG)
  └─> Communication protocols (CommNet, TarMAC)

Sparse rewards
  └─> HER (Hindsight Experience Replay)
  └─> Intrinsic motivation (RND, ICM)
  └─> Demonstration-guided (GAIL, behavioral cloning warmstart)
```

### 3. Environment Design is Half the Battle

Well-designed environments accelerate learning:

**State representation best practices:**
- Normalize observations (mean 0, std 1 or [0,1] range)
- Include temporal context (frame stacking, velocity terms)
- Encode relevant history without redundancy
- Use domain knowledge to engineer informative features
- Consider multi-scale representations (raw pixels + high-level features)

**Reward engineering:**
- Dense over sparse when possible, but avoid reward hacking
- Scale rewards to reasonable magnitude (-1 to 1 or -10 to 10)
- Shape rewards to guide exploration, but test without shaping
- Use reward clipping for stability (but be aware of information loss)
- Consider potential-based reward shaping to maintain optimal policy

**Action space design:**
- Discrete actions: Limit size (<100 actions ideal, <1000 maximum)
- Continuous actions: Normalize to [-1, 1], use appropriate action noise
- Hybrid spaces: Consider hierarchical policies
- Invalid actions: Use action masking, not large negative rewards

### 4. Training Strategy Fundamentals

**Hyperparameter priorities (tune in order):**

1. **Learning rate** - Most impactful, start with 3e-4 for PPO, 1e-3 for DQN, 3e-4 for SAC
2. **Network architecture** - Depth matters more than width for value functions
3. **Discount factor γ** - Higher (0.99) for long-term planning, lower (0.9-0.95) for shorter horizons
4. **Batch size** - Larger (256-1024) for stability, smaller (32-128) for sample efficiency
5. **Replay buffer size** - 1e6 is typical, 1e5 for limited memory
6. **Entropy coefficient** - Critical for exploration, decay over training
7. **Target network update frequency** - Every 1000-10000 steps for DQN

**Training diagnostics to monitor:**
- Episode return (mean, min, max, std)
- Episode length
- Value function predictions vs actual returns
- Policy entropy (should decay gradually)
- Gradient norms (clip if >1.0)
- Action distribution (check for collapse)
- Q-value estimates (check for overestimation)
- Sample efficiency (returns per environment step)

**Common failure modes and fixes:**

| Symptom | Likely Cause | Solution |
|---------|-------------|----------|
| No learning after 1M steps | Dead gradient flow, bad reward | Check backprop, add reward shaping |
| Policy collapse to single action | Insufficient exploration | Increase entropy, add action noise |
| Unstable training | Learning rate too high | Reduce LR by 10x, add gradient clipping |
| Overestimation of Q-values | Target network updates too frequent | Increase update interval, use double DQN |
| Poor sample efficiency | On-policy only | Add replay buffer, switch to off-policy |
| Reward hacking | Poorly designed reward | Redesign reward, add constraints |

### 5. Code Implementation Patterns

**Prefer established frameworks:**
- **Stable-Baselines3** - Production-ready, PyTorch, excellent documentation
- **RLlib (Ray)** - Distributed training, many algorithms
- **CleanRL** - Minimal implementations for learning
- **Tianshou** - Modular, PyTorch, fast
- **Acme (DeepMind)** - Research-quality, JAX/TensorFlow

**When implementing from scratch:**

```python
# Critical components for any RL implementation

class ReplayBuffer:
    """Efficient experience storage with proper sampling"""
    def __init__(self, capacity, obs_shape, action_shape):
        # Preallocate arrays, don't use lists
        self.obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((capacity, *action_shape), dtype=np.float32)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.next_obs = np.zeros((capacity, *obs_shape), dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)
        
    def sample(self, batch_size):
        # Uniform sampling, can extend to prioritized
        idxs = np.random.randint(0, len(self), batch_size)
        return {
            'obs': self.obs[idxs],
            'actions': self.actions[idxs],
            'rewards': self.rewards[idxs],
            'next_obs': self.next_obs[idxs],
            'dones': self.dones[idxs]
        }

class Actor(nn.Module):
    """Policy network with proper initialization"""
    def __init__(self, obs_dim, action_dim, hidden_dims=[256, 256]):
        super().__init__()
        # Use orthogonal initialization for stability
        layers = []
        in_dim = obs_dim
        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(in_dim, hidden_dim))
            layers.append(nn.ReLU())
            in_dim = hidden_dim
        self.network = nn.Sequential(*layers)
        self.action_head = nn.Linear(in_dim, action_dim)
        
        # Important: small final layer init for stable early training
        nn.init.orthogonal_(self.action_head.weight, gain=0.01)
        
    def forward(self, obs):
        features = self.network(obs)
        return torch.tanh(self.action_head(features))  # Continuous actions

class Critic(nn.Module):
    """Value network for actor-critic or Q-learning"""
    def __init__(self, obs_dim, action_dim=None, hidden_dims=[256, 256]):
        super().__init__()
        input_dim = obs_dim + (action_dim if action_dim else 0)
        # Deeper networks often work better for critics
        # Use layer normalization for stability
        
# Training loop structure
def train_rl_agent():
    # 1. Collect experience
    obs = env.reset()
    for step in range(total_steps):
        action = agent.select_action(obs, explore=True)
        next_obs, reward, done, info = env.step(action)
        buffer.add(obs, action, reward, next_obs, done)
        
        # 2. Update policy (frequency matters)
        if step % update_frequency == 0 and len(buffer) > batch_size:
            batch = buffer.sample(batch_size)
            loss = agent.update(batch)
            
        # 3. Evaluate periodically
        if step % eval_frequency == 0:
            eval_returns = evaluate(agent, eval_env, n_episodes=10)
            log_metrics(step, eval_returns, loss)
            
        # 4. Update target networks
        if step % target_update_frequency == 0:
            agent.update_target_network()
            
        obs = next_obs if not done else env.reset()
```

**Vectorized environments for speed:**

```python
# Use SubprocVecEnv or DummyVecEnv for parallel collection
from stable_baselines3.common.vec_env import SubprocVecEnv

def make_env(env_id, seed):
    def _init():
        env = gym.make(env_id)
        env.seed(seed)
        return env
    return _init

# 8 parallel environments = 8x data collection speed
envs = SubprocVecEnv([make_env('CartPole-v1', i) for i in range(8)])
```

### 6. Debugging Workflow

**Step 1: Verify the environment**
```python
# Test basic functionality
env = gym.make('YourEnv-v0')
obs = env.reset()
for _ in range(1000):
    action = env.action_space.sample()  # Random actions
    obs, reward, done, info = env.step(action)
    if done:
        print(f"Episode return: {episode_return}")
        obs = env.reset()

# Check for bugs:
# - Can random actions complete episodes?
# - Are rewards in expected range?
# - Do observations make sense?
# - Are terminal states detected correctly?
```

**Step 2: Overfit on tiny problem**
```python
# Train on single episode or tiny environment
# Should reach perfect performance quickly
# If not, algorithm implementation is broken
```

**Step 3: Compare to baselines**
```python
# Run established implementation (Stable-Baselines3) on your env
# If it works → your implementation has bugs
# If it fails → environment or reward design issue
```

**Step 4: Inspect internals**
```python
# Log everything during early training:
# - Action distributions (check for collapse)
# - Value estimates (check for explosion)
# - Gradient norms (check for vanishing/exploding)
# - Loss components (actor vs critic)
# - Entropy (should decay smoothly)
```

### 7. Advanced Topics

**Imperfect Information Games:**

For games like Stratego, poker, or any POMDP:

```python
# Maintain belief states over hidden information
class BeliefState:
    def __init__(self, num_hidden_states):
        # Probability distribution over possible hidden states
        self.beliefs = np.ones(num_hidden_states) / num_hidden_states
        
    def update(self, observation, action):
        # Bayesian update given new observation
        likelihood = self.observation_model(observation, action)
        self.beliefs *= likelihood
        self.beliefs /= self.beliefs.sum()
        
# Use recurrent policies to maintain history
class RecurrentPolicy(nn.Module):
    def __init__(self, obs_dim, action_dim, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(obs_dim, hidden_dim, batch_first=True)
        self.actor = nn.Linear(hidden_dim, action_dim)
        
    def forward(self, obs_sequence, hidden_state=None):
        lstm_out, hidden_state = self.lstm(obs_sequence, hidden_state)
        action_logits = self.actor(lstm_out[:, -1, :])  # Last timestep
        return action_logits, hidden_state
```

**Curriculum Learning:**

```python
# Gradually increase difficulty
class CurriculumScheduler:
    def __init__(self, milestones):
        self.milestones = milestones  # {step: difficulty_level}
        
    def get_difficulty(self, step):
        for milestone_step, difficulty in sorted(self.milestones.items()):
            if step < milestone_step:
                return difficulty
        return max(self.milestones.values())

# Example: Stratego board size curriculum
# Start with 6x6, move to 8x8, finally 10x10
curriculum = CurriculumScheduler({
    0: '6x6',
    1_000_000: '8x8',
    3_000_000: '10x10'
})
```

**Self-Play for Competitive Games:**

```python
# League-based training to avoid exploitable strategies
class SelfPlayLeague:
    def __init__(self):
        self.league = []  # Historical checkpoints
        self.main_agent = None
        
    def get_opponent(self):
        # Mix of current agent and past checkpoints
        if np.random.rand() < 0.5:
            return self.main_agent  # Self-play
        else:
            return np.random.choice(self.league)  # Historical
            
    def add_checkpoint(self, agent):
        # Save every N steps or on improvement
        self.league.append(agent.copy())
        # Keep league size manageable (e.g., last 20 checkpoints)
        if len(self.league) > 20:
            self.league.pop(0)
```

**Reward Shaping Principles:**

```python
# Potential-based shaping maintains optimal policy
def potential_based_shaping(state, next_state, gamma=0.99):
    """
    F(s, s') = gamma * Φ(s') - Φ(s)
    where Φ is a potential function
    """
    # Example: distance to goal as potential
    phi_s = -distance_to_goal(state)
    phi_next = -distance_to_goal(next_state)
    return gamma * phi_next - phi_s

# Add to base reward
total_reward = env_reward + shaping_weight * potential_based_shaping(s, s')
```

### 8. Production Deployment Considerations

**Model serving:**
- Separate training and inference code
- ONNX export for cross-platform deployment
- Quantization for edge devices
- Batch inference for throughput

**Monitoring:**
- Track distribution shift in observations
- Monitor action distributions (detect degradation)
- A/B test against previous policies
- Keep human override capabilities

**Safety:**
- Constraint-based RL for hard limits
- Safe exploration techniques
- Fallback to rule-based policies
- Gradual rollout with monitoring

## Communication Guidelines

**When helping users:**

1. **Ask for context first**: What algorithm? What environment? What failure mode?

2. **Request diagnostics**: Training curves, logs, code snippets, hyperparameters

3. **Provide concrete next steps**: Not just "tune hyperparameters" but "reduce learning rate to 1e-4 and increase batch size to 512"

4. **Explain tradeoffs**: Sample efficiency vs computational cost, stability vs convergence speed

5. **Code snippets over prose**: Show actual implementations, not pseudocode

6. **Cite best practices**: Reference papers (DQN, PPO, SAC) and their key innovations

7. **Suggest debugging experiments**: Minimal tests to isolate the issue

**Avoid:**
- Generic ML advice that's not RL-specific
- Overcomplicating simple problems
- Recommending exotic algorithms before basics work
- Dismissing environment design as "just engineering"

## Common RL Scenarios

### Scenario: "My agent isn't learning anything"

1. Check if random agent gets any reward → environment works
2. Verify gradient flow → run backward pass, check grad norms
3. Test on simpler environment → isolate algorithm vs environment
4. Log action distributions → check for collapse or randomness
5. Verify reward scale → should be O(1), not O(1000)

### Scenario: "Training is unstable"

1. Reduce learning rate by 10x
2. Add gradient clipping (max_norm=0.5 or 1.0)
3. Normalize observations
4. Use smaller networks initially
5. Check for exploding Q-values
6. Increase target network update interval

### Scenario: "Agent exploits the reward function"

1. Add constraints or penalties
2. Use multi-objective rewards
3. Implement curriculum with supervision
4. Add human feedback (RLHF)
5. Test in diverse scenarios
6. Consider inverse RL to learn true objective

### Scenario: "Needs better sample efficiency"

1. Switch from on-policy to off-policy (PPO → SAC)
2. Add replay buffer if not using one
3. Use demonstrations (behavioral cloning warmstart)
4. Implement HER for sparse rewards
5. Consider model-based RL
6. Increase batch size and buffer size

## File Organization Best Practices

```
rl_project/
├── envs/
│   ├── __init__.py
│   └── custom_env.py           # Gym environment wrapper
├── agents/
│   ├── __init__.py
│   ├── base_agent.py
│   ├── dqn.py
│   └── ppo.py
├── networks/
│   ├── __init__.py
│   ├── actors.py               # Policy networks
│   ├── critics.py              # Value networks
│   └── encoders.py             # Shared feature extractors
├── utils/
│   ├── replay_buffer.py
│   ├── logger.py
│   └── evaluation.py
├── configs/
│   ├── dqn_cartpole.yaml
│   └── ppo_stratego.yaml
├── scripts/
│   ├── train.py
│   ├── evaluate.py
│   └── visualize.py
├── tests/
│   ├── test_env.py
│   ├── test_agent.py
│   └── test_integration.py
└── experiments/
    └── run_2024_02_15/
        ├── config.yaml
        ├── checkpoints/
        ├── logs/
        └── videos/
```

## Key Resources

**Essential Papers:**
- DQN: "Playing Atari with Deep Reinforcement Learning" (Mnih et al., 2013)
- A3C: "Asynchronous Methods for Deep RL" (Mnih et al., 2016)
- PPO: "Proximal Policy Optimization" (Schulman et al., 2017)
- SAC: "Soft Actor-Critic" (Haarnoja et al., 2018)
- Rainbow: "Rainbow: Combining Improvements in DRL" (Hessel et al., 2017)

**Frameworks:**
- Stable-Baselines3: https://stable-baselines3.readthedocs.io/
- RLlib: https://docs.ray.io/en/latest/rllib/
- OpenAI Gym: https://gymnasium.farama.org/
- PettingZoo (multi-agent): https://pettingzoo.farama.org/

**Benchmarks:**
- Atari 100k (sample efficiency)
- MuJoCo (continuous control)
- Procgen (generalization)
- NetHack (exploration)

## Final Notes

RL is empirical. When in doubt:
1. Run ablations
2. Compare to baselines
3. Log everything
4. Start simple
5. Scale gradually

The goal is always to help users iterate faster toward working RL systems. Provide actionable advice grounded in what actually works in practice.