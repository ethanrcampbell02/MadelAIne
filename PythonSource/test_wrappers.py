"""
Test script to visualize environment observations at each wrapper stage

Controls:
- Arrow Keys / IJKL: Movement
- D: Jump
- S: Dash
- A: Grab
- SPACE/ENTER: Take step with current action
- R: Reset environment
- N: Next stage
- Q/ESC: Quit
"""
import cv2
import numpy as np
from CelesteEnv import CelesteEnv
from wrappers import SimplifiedActionSpace, SkipFrame
from gymnasium.wrappers import ResizeObservation, GrayscaleObservation, FrameStackObservation
import gymnasium as gym

def visualize_observation(obs, title="Observation", wait_time=0):
    """Display an observation in a window"""
    if obs is None:
        print(f"{title}: None")
        return
    
    print(f"{title} shape: {obs.shape}, dtype: {obs.dtype}, min: {obs.min()}, max: {obs.max()}")
    
    # Handle different observation shapes
    if len(obs.shape) == 2:  # Grayscale 2D
        display = obs
    elif len(obs.shape) == 3:
        if obs.shape[0] == 4:  # Frame stack (4, H, W)
            # Show the most recent frame
            display = obs[0]
        elif obs.shape[2] == 1:  # Grayscale with channel (H, W, 1)
            display = obs[:, :, 0]
        elif obs.shape[2] == 3:  # RGB (H, W, 3)
            display = cv2.cvtColor(obs, cv2.COLOR_RGB2BGR)
        elif obs.shape[2] == 4:  # RGBA (H, W, 4)
            display = cv2.cvtColor(obs[:, :, :3], cv2.COLOR_RGB2BGR)
        else:
            display = obs[:, :, 0]
    else:
        print(f"Unexpected shape: {obs.shape}")
        return
    
    # Scale up for better visibility
    if display.shape[0] < 200:
        scale = 4
    else:
        scale = 2
    display = cv2.resize(display, (display.shape[1] * scale, display.shape[0] * scale), 
                        interpolation=cv2.INTER_NEAREST)
    
    cv2.imshow(title, display)
    if wait_time > 0:
        cv2.waitKey(wait_time)

def get_action_from_keys(keys, action_space):
    """Convert keyboard state to action based on action space type"""
    if isinstance(action_space, gym.spaces.Discrete):  # Discrete
        # For SimplifiedActionSpace: 0=idle(grab), 1=jump, 2=up, 3=right, 4=right+jump
        right_pressed = keys.get(ord('l'), False) or keys.get(3, False)
        jump_pressed = keys.get(ord('d'), False) or keys.get(ord('D'), False)
        up_pressed = keys.get(ord('i'), False) or keys.get(0, False)
        
        if right_pressed and jump_pressed:
            return 4  # Right + Jump
        if right_pressed:
            return 3  # Right
        if up_pressed:
            return 2  # Up
        if jump_pressed:
            return 1  # Jump
        return 0  # Idle (grab only)
    else:  # MultiBinary
        action = np.zeros(7, dtype=np.int8)
        if keys.get(ord('i'), False) or keys.get(0, False):
            action[0] = 1
        if keys.get(ord('k'), False) or keys.get(1, False):
            action[1] = 1
        if keys.get(ord('j'), False) or keys.get(2, False):
            action[2] = 1
        if keys.get(ord('l'), False) or keys.get(3, False):
            action[3] = 1
        if keys.get(ord('d'), False) or keys.get(ord('D'), False):
            action[4] = 1
        if keys.get(ord('s'), False) or keys.get(ord('S'), False):
            action[5] = 1
        if keys.get(ord('a'), False) or keys.get(ord('A'), False):
            action[6] = 1
        return action

def print_action(action, action_space):
    """Print human-readable action"""
    if isinstance(action_space, gym.spaces.Discrete):  # Discrete
        actions = ["IDLE(GRAB)", "JUMP", "UP", "RIGHT", "RIGHT+JUMP"]
        return actions[action] if action < len(actions) else f"Action {action}"
    else:  # MultiBinary
        parts = []
        if action[0]: parts.append("UP")
        if action[1]: parts.append("DOWN")
        if action[2]: parts.append("LEFT")
        if action[3]: parts.append("RIGHT")
        if action[4]: parts.append("JUMP")
        if action[5]: parts.append("DASH")
        if action[6]: parts.append("GRAB")
        return " + ".join(parts) if parts else "NEUTRAL"

def interactive_stage(env, stage_name):
    """Interactive testing for a single wrapper stage"""
    print(f"\n{'=' * 60}")
    print(f"Stage: {stage_name}")
    print(f"Action Space: {env.action_space}")
    print(f"Observation Space: {env.observation_space}")
    print("Controls: Arrow/IJKL=move, D=jump, S=dash, A=grab, SPACE=step, R=reset, N=next, Q=quit")
    print('=' * 60)
    
    current_keys = {}
    obs, info = env.reset()
    visualize_observation(obs, stage_name)
    step_count = 0
    
    while True:
        action = get_action_from_keys(current_keys, env.action_space)
        print(f"\nStep {step_count} | Action: {print_action(action, env.action_space)}")
        if info:
            print(f"Distance: {info.get('distance', 'N/A'):.2f} | Died: {info.get('playerDied', False)}")
        
        key = cv2.waitKey(0) & 0xFF
        
        # Quit
        if key == ord('q') or key == ord('Q') or key == 27:
            return 'quit'
        
        # Next stage
        if key == ord('n') or key == ord('N'):
            return 'next'
        
        # Reset
        if key == ord('r') or key == ord('R'):
            print("[RESET]")
            current_keys = {}
            obs, info = env.reset()
            visualize_observation(obs, stage_name)
            step_count = 0
            continue
        
        # Toggle keys
        if key in [ord('a'), ord('A'), ord('s'), ord('S'), ord('d'), ord('D'),
                  ord('i'), ord('I'), ord('k'), ord('K'), ord('j'), ord('J'), ord('l'), ord('L'),
                  0, 1, 2, 3]:
            if key in current_keys and current_keys[key]:
                current_keys[key] = False
                print(f"Released: {chr(key) if key >= 32 else f'Arrow({key})'}")
            else:
                current_keys[key] = True
                print(f"Pressed: {chr(key) if key >= 32 else f'Arrow({key})'}")
            continue
        
        # Step
        if key == ord(' ') or key == 13:
            print(f"[STEP] Action: {print_action(action, env.action_space)}")
            obs, reward, terminated, truncated, info = env.step(action)
            visualize_observation(obs, stage_name)
            print(f"Reward: {reward:.3f} | Term: {terminated} | Trunc: {truncated}")
            step_count += 1
            
            if terminated or truncated:
                print(f"\n[EPISODE END] {'Terminated' if terminated else 'Truncated'}")
                print("Press R to reset, N for next stage, or Q to quit...")

def test_wrapper_stages():
    """Test and visualize each wrapper stage"""
    print("Starting wrapper visualization test...")
    print("Interactive controls enabled!\n")
    
    stages = [
        ("1. Raw CelesteEnv", lambda: CelesteEnv(reward_mode="best", render_mode=None)),
        ("2. + SimplifiedActionSpace", lambda: SimplifiedActionSpace(CelesteEnv(reward_mode="best", render_mode=None))),
        ("3. + SkipFrame", lambda: SkipFrame(SimplifiedActionSpace(CelesteEnv(reward_mode="best", render_mode=None)), skip=4)),
        ("4. + ResizeObservation", lambda: ResizeObservation(SkipFrame(SimplifiedActionSpace(CelesteEnv(reward_mode="best", render_mode=None)), skip=4), shape=(84, 84))),
        ("5. + GrayscaleObservation", lambda: GrayscaleObservation(ResizeObservation(SkipFrame(SimplifiedActionSpace(CelesteEnv(reward_mode="best", render_mode=None)), skip=4), shape=(84, 84)), keep_dim=True)),
        ("6. + FrameStackObservation", lambda: FrameStackObservation(GrayscaleObservation(ResizeObservation(SkipFrame(SimplifiedActionSpace(CelesteEnv(reward_mode="best", render_mode=None)), skip=4), shape=(84, 84)), keep_dim=True), stack_size=4)),
    ]
    
    for stage_name, env_factory in stages:
        env = env_factory()
        result = interactive_stage(env, stage_name)
        env.close()
        cv2.destroyAllWindows()
        
        if result == 'quit':
            print("\n[QUIT] Exiting test...")
            return
        elif result == 'next':
            print(f"\n[NEXT] Moving to next stage...")
            continue
    
    print("\n" + "=" * 60)
    print("All stages complete!")
    print("=" * 60)
    cv2.destroyAllWindows()

if __name__ == "__main__":
    test_wrapper_stages()

