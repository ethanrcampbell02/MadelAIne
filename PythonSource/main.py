import os
import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv, VecFrameStack
from stable_baselines3.common.callbacks import CheckpointCallback, EvalCallback, BaseCallback
from stable_baselines3.common.monitor import Monitor
import logging

from CelesteEnv import CelesteEnv
from wrappers import SimplifiedActionSpace, SkipFrame, ClipReward
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation
from utils import get_current_date_time_string

# Configuration
TOTAL_TIMESTEPS = 1_000_000  # Total training timesteps
EVAL_FREQ = 10_000  # Evaluate every N steps
SAVE_FREQ = 50_000  # Save checkpoint every N steps
N_EVAL_EPISODES = 5  # Number of episodes for evaluation
LEARNING_RATE = 3e-4
N_STEPS = 2048  # Number of steps to collect before update
BATCH_SIZE = 64  # Minibatch size for PPO updates
N_EPOCHS = 10  # Number of epochs when optimizing the surrogate loss
GAMMA = 0.99  # Discount factor
GAE_LAMBDA = 0.95  # GAE lambda parameter
CLIP_RANGE = 0.2  # PPO clipping parameter
ENT_COEF = 0.01  # Entropy coefficient for exploration
VF_COEF = 0.5  # Value function coefficient
MAX_GRAD_NORM = 0.5  # Max gradient norm for clipping

LOAD_CHECKPOINT = False  # Set to True to continue training from checkpoint
CHECKPOINT_PATH = None  # Path to checkpoint if loading

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

# Create model directory
model_path = os.path.join("models", "ppo_" + get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)
checkpoint_dir = os.path.join(model_path, "checkpoints")
log_dir = os.path.join(model_path, "logs")
os.makedirs(checkpoint_dir, exist_ok=True)
os.makedirs(log_dir, exist_ok=True)

print(f"Model will be saved to: {model_path}")
if torch.cuda.is_available():
    print(f"Using CUDA device: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available, using CPU")


class ProgressCallback(BaseCallback):
    """Custom callback for logging training progress"""
    def __init__(self, verbose=0):
        super().__init__(verbose)
        self.episode_rewards = []
        self.episode_lengths = []
        self.episode_distances_travelled = []  # Track total distance travelled each episode
        self.episode_distance_per_step = []  # Track distance travelled per step
        self.num_episodes = 0
        
    def _on_step(self) -> bool:
        # Check if episode completed (only count when actually finished)
        for idx, done in enumerate(self.locals.get("dones", [])):
            if done:
                info = self.locals["infos"][idx]
                if "episode" in info:
                    self.episode_rewards.append(info["episode"]["r"])
                    self.episode_lengths.append(info["episode"]["l"])
                    self.num_episodes += 1
                    
                    # Track distance metrics from the episode info
                    if "distance_travelled" in info:
                        self.episode_distances_travelled.append(info["distance_travelled"])
                        
                        # Calculate distance per step (higher is better)
                        episode_time = info["episode"]["l"]  # Episode length in steps
                        if episode_time > 0:
                            distance_per_step = info["distance_travelled"] / episode_time
                            self.episode_distance_per_step.append(distance_per_step)
                    
                    # Log to TensorBoard
                    if len(self.episode_distances_travelled) > 0:
                        self.logger.record("rollout/ep_distance_travelled_mean", np.mean(self.episode_distances_travelled[-10:]))
                    if len(self.episode_distance_per_step) > 0:
                        self.logger.record("rollout/ep_distance_per_step_mean", np.mean(self.episode_distance_per_step[-10:]))
                
        # Log every 1000 steps
        if self.n_calls % 1000 == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-10:])  # Last 10 episodes
                mean_length = np.mean(self.episode_lengths[-10:])
                mean_distance = np.mean(self.episode_distances_travelled[-10:]) if len(self.episode_distances_travelled) > 0 else 0
                logging.info(
                    f"Step: {self.n_calls} | "
                    f"Episodes: {self.num_episodes} | "
                    f"Mean Reward (last 10): {mean_reward:.2f} | "
                    f"Mean Length (last 10): {mean_length:.1f} | "
                    f"Mean Distance Travelled (last 10): {mean_distance:.2f}"
                )
        return True


def make_env():
    """Create and wrap the Celeste environment"""
    env = CelesteEnv(reward_mode="best", render_mode="human")
    env = ClipReward(env, min_reward=-1, max_reward=1)
    env = SimplifiedActionSpace(env)
    env = SkipFrame(env, skip=4)
    env = ResizeObservation(env, shape=(84, 84))  # Standard Atari size
    env = GrayscaleObservation(env, keep_dim=True)  # Keep channel dimension
    env = Monitor(env, log_dir)  # Monitor for logging
    return env


def main():
    # Create environment (single instance only - CelesteEnv doesn't support parallel envs)
    print("Creating environment...")
    env = DummyVecEnv([make_env])
    env = VecFrameStack(env, n_stack=4)  # Stack 4 frames
    
    # Create or load model
    if LOAD_CHECKPOINT and CHECKPOINT_PATH and os.path.exists(CHECKPOINT_PATH):
        print(f"Loading model from {CHECKPOINT_PATH}")
        model = PPO.load(
            CHECKPOINT_PATH,
            env=env,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    else:
        print("Creating new PPO model...")
        model = PPO(
            policy="CnnPolicy",  # CNN policy for image observations
            env=env,
            learning_rate=LEARNING_RATE,
            n_steps=N_STEPS,
            batch_size=BATCH_SIZE,
            n_epochs=N_EPOCHS,
            gamma=GAMMA,
            gae_lambda=GAE_LAMBDA,
            clip_range=CLIP_RANGE,
            ent_coef=ENT_COEF,
            vf_coef=VF_COEF,
            max_grad_norm=MAX_GRAD_NORM,
            verbose=1,
            tensorboard_log=log_dir,
            device='cuda' if torch.cuda.is_available() else 'cpu'
        )
    
    # Set up callbacks
    checkpoint_callback = CheckpointCallback(
        save_freq=SAVE_FREQ,
        save_path=checkpoint_dir,
        name_prefix="ppo_celeste",
        save_replay_buffer=False,
        save_vecnormalize=False
    )
    
    # Note: EvalCallback disabled because CelesteEnv doesn't support multiple instances
    # The model will be saved at checkpoints instead
    
    progress_callback = ProgressCallback()
    
    # Train the model
    print(f"\nStarting training for {TOTAL_TIMESTEPS:,} timesteps...")
    print(f"Checkpoints will be saved every {SAVE_FREQ:,} steps to {checkpoint_dir}")
    print(f"TensorBoard logs: tensorboard --logdir {log_dir}\n")
    
    try:
        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=[checkpoint_callback, progress_callback],
            progress_bar=True
        )
        
        # Save final model
        final_model_path = os.path.join(model_path, "final_model")
        model.save(final_model_path)
        print(f"\nTraining complete! Final model saved to {final_model_path}")
        
    except KeyboardInterrupt:
        print("\nTraining interrupted by user")
        interrupt_model_path = os.path.join(model_path, "interrupted_model")
        model.save(interrupt_model_path)
        print(f"Model saved to {interrupt_model_path}")
    
    finally:
        # Clean up
        env.close()


if __name__ == "__main__":
    main()
