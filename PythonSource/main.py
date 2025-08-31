import os
import torch
import logging
from tqdm import trange
import matplotlib.pyplot as plt
import numpy as np

from MadelAIneAgent import MadelAIneAgent

from CelesteEnv import CelesteEnv
from wrappers import apply_wrappers
from utils import *

SAVE_METRICS_INTERVAL = 10
SAVE_MODEL_AFTER_EPOCH = False
SHOULD_TRAIN = True
NUM_OF_EPOCHS = 1000
TRAINING_EPISODES_PER_EPOCH = 100
VALIDATION_EPISODES_PER_EPOCH = 10

TRAIN_FROM_CKPT = False

# Set up logger to use tqdm
logger = logging.getLogger()
logger.setLevel(logging.INFO)
handler = TqdmLoggingHandler()
formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
handler.setFormatter(formatter)
logger.handlers = [handler]  # Replace existing handlers

model_path = os.path.join("models", get_current_date_time_string())
os.makedirs(model_path, exist_ok=True)
# Create subdirectories for organization
model_subdir = os.path.join(model_path, "checkpoints")
plot_subdir = os.path.join(model_path, "plots")
stats_subdir = os.path.join(model_path, "stats")
images_subdir = os.path.join(model_path, "images")
os.makedirs(model_subdir, exist_ok=True)
os.makedirs(plot_subdir, exist_ok=True)
os.makedirs(stats_subdir, exist_ok=True)
os.makedirs(images_subdir, exist_ok=True)

if torch.cuda.is_available():
    print("Using CUDA device:", torch.cuda.get_device_name(0))
else:
    print("CUDA is not available")

# Set up plots with separate subplots for reward and loss
plt.ion()
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8, 8))

lines = {
    "train_reward": ax1.plot([], [], label="Avg Training Reward")[0],
    "val_reward": ax1.plot([], [], label="Avg Validation Reward")[0],
    "train_loss": ax2.plot([], [], label="Avg Training Loss")[0]
}

ax1.set_xlabel("Epoch")
ax1.set_ylabel("Reward")
ax1.legend()
ax1.set_title("Training and Validation Rewards")

ax2.set_xlabel("Epoch")
ax2.set_ylabel("Loss")
ax2.legend()
ax2.set_title("Training Loss")

fig.tight_layout()
fig.canvas.draw()
fig.show()

# Function to save plot and metric values
def save_plot_and_values(epoch=None):
    # Save plot
    plot_name = f'training_metrics{"_epoch_" + str(epoch) if epoch is not None else ""}.png'
    fig.savefig(os.path.join(plot_subdir, plot_name))
    # Save values/statistics
    np.save(os.path.join(stats_subdir, f'avg_training_rewards{"_epoch_" + str(epoch) if epoch is not None else ""}.npy'), np.array(avg_training_rewards))
    np.save(os.path.join(stats_subdir, f'avg_validation_rewards{"_epoch_" + str(epoch) if epoch is not None else ""}.npy'), np.array(avg_validation_rewards))
    np.save(os.path.join(stats_subdir, f'avg_training_losses{"_epoch_" + str(epoch) if epoch is not None else ""}.npy'), np.array(avg_training_losses))

env = CelesteEnv()
env = apply_wrappers(env)

input_dims = env.observation_space.shape
num_actions = 2 ** env.action_space.n
agent = MadelAIneAgent(input_dims=input_dims, num_actions=num_actions, epsilon=1.0, eps_min=0.1, eps_decay=0.9999995)

folder_name = "2025-08-30-10_02_46"
ckpt_name = "model_5000_iter.pt"
if not SHOULD_TRAIN:
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.05
    agent.eps_min = 0.0
    agent.eps_decay = 0.0
elif TRAIN_FROM_CKPT:
    agent.load_model(os.path.join("models", folder_name, ckpt_name))
    agent.epsilon = 0.7
    agent.eps_min = 0.01
    agent.eps_decay = 0.9999995

# Initialize lists to store metrics for plotting
avg_training_rewards = []
avg_validation_rewards = []
avg_training_losses = []

for epoch in trange(NUM_OF_EPOCHS, desc="Epochs"):

    episode_training_rewards = []
    episode_training_losses = []
    episode_validation_rewards = []

    # Training
    if SHOULD_TRAIN:
        for i in trange(TRAINING_EPISODES_PER_EPOCH, desc="Training Episodes"):
            done = False
            state, _ = env.reset()
            total_reward = 0
            batch_losses = []

            while not done:
                a = agent.choose_action(state)
                action_multi_binary = [int(x) for x in format(a, f'0{env.action_space.n}b')]
                action = np.array(action_multi_binary, dtype=np.float32)

                new_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                total_reward += reward

                agent.store_in_memory(state, a, reward, new_state, done)
                loss = agent.learn()
                if loss is not None:
                    batch_losses.append(loss)

                state = new_state

            avg_loss = np.mean(batch_losses) if batch_losses else None
            episode_training_rewards.append(total_reward)
            episode_training_losses.append(avg_loss if avg_loss is not None else 0)
            logging.info(f"Epoch {epoch + 1} | Training Episode {i + 1} | Reward: {total_reward} | Loss: {avg_loss} | Epsilon: {agent.epsilon} | Learn steps: {agent.learn_step_counter}")

        if SAVE_MODEL_AFTER_EPOCH:
            agent.save_model(os.path.join(model_subdir, "model_" + str((epoch + 1) * TRAINING_EPISODES_PER_EPOCH) + "_epoch.pt"))

    # Store current training epsilon parameters
    current_epsilon = agent.epsilon
    current_eps_min = agent.eps_min
    current_eps_decay = agent.eps_decay

    # Validation
    for i in trange(VALIDATION_EPISODES_PER_EPOCH, desc="Validation Episodes"):
        done = False
        state, _ = env.reset()
        total_reward = 0

        agent.epsilon = 0.05 # Small epsilon for validation
        agent.eps_decay = 0.0
        agent.eps_min = 0.0

        while not done:
            a = agent.choose_action(state)
            action_multi_binary = [int(x) for x in format(a, f'0{env.action_space.n}b')]
            action = np.array(action_multi_binary, dtype=np.float32)

            new_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            total_reward += reward

        episode_validation_rewards.append(total_reward)
        logging.info(f"Epoch {epoch + 1} | Validation Episode {i + 1} | Reward: {total_reward}")

    # Restore training epsilon parameters
    agent.epsilon = current_epsilon
    agent.eps_min = current_eps_min
    agent.eps_decay = current_eps_decay

    # Compute averages for this epoch
    avg_train_reward = np.mean(episode_training_rewards) if episode_training_rewards else 0
    avg_val_reward = np.mean(episode_validation_rewards) if episode_validation_rewards else 0
    avg_train_loss = np.mean(episode_training_losses) if episode_training_losses else 0

    avg_training_rewards.append(avg_train_reward)
    avg_validation_rewards.append(avg_val_reward)
    avg_training_losses.append(avg_train_loss)

    # Update plot
    epochs = np.arange(1, len(avg_training_rewards) + 1)
    lines["train_reward"].set_data(epochs, avg_training_rewards)
    lines["val_reward"].set_data(epochs, avg_validation_rewards)
    lines["train_loss"].set_data(epochs, avg_training_losses)
    ax1.relim()
    ax1.autoscale_view()
    ax2.relim()
    ax2.autoscale_view()
    fig.canvas.draw()
    fig.canvas.flush_events()

    # Save plot and values after every SAVE_METRICS_INTERVAL epochs
    if (epoch + 1) % SAVE_METRICS_INTERVAL == 0:
        save_plot_and_values(epoch + 1)


env.close()
# Save plot and values one final time after training
save_plot_and_values()
# Example: Save images to images_subdir if you generate any
# image.save(os.path.join(images_subdir, f"image_{epoch}.png"))