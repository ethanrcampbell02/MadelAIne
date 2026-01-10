"""
Debug script for manually stepping through CelesteEnv to debug handshaking.

Controls:
- Arrow keys: Movement (up, down, left, right)
- D: Jump
- S: Dash
- A: Grab
- R: Reset environment
- Q/ESC: Quit

Press any key to take a step with the current action.
"""

import numpy as np
import cv2
import logging
from CelesteEnv import CelesteEnv

logging.basicConfig(level=logging.DEBUG)

def get_action_from_keys(keys):
    """
    Convert keyboard state to action.
    Action format: [up, down, left, right, jump, dash, grab]
    """
    action = np.zeros(7, dtype=np.int8)
    
    # IJKL for movement
    if keys.get(ord('i'), False) or keys.get(ord('I'), False):  # Up
        action[0] = 1
    if keys.get(ord('k'), False) or keys.get(ord('K'), False):  # Down
        action[1] = 1
    if keys.get(ord('j'), False) or keys.get(ord('J'), False):  # Left
        action[2] = 1
    if keys.get(ord('l'), False) or keys.get(ord('L'), False):  # Right
        action[3] = 1
    
    # Action buttons
    if keys.get(ord('d'), False) or keys.get(ord('D'), False):  # Jump
        action[4] = 1
    if keys.get(ord('s'), False) or keys.get(ord('S'), False):  # Dash
        action[5] = 1
    if keys.get(ord('a'), False) or keys.get(ord('A'), False):  # Grab
        action[6] = 1
    
    return action

def print_action(action):
    """Print human-readable action"""
    parts = []
    if action[0]: parts.append("UP")
    if action[1]: parts.append("DOWN")
    if action[2]: parts.append("LEFT")
    if action[3]: parts.append("RIGHT")
    if action[4]: parts.append("JUMP")
    if action[5]: parts.append("DASH")
    if action[6]: parts.append("GRAB")
    
    if parts:
        return " + ".join(parts)
    else:
        return "NEUTRAL"

def main():
    print("=" * 60)
    print("CelesteEnv Debug Tool")
    print("=" * 60)
    print("\nWaiting for connection from Celeste...")
    
    # Create environment
    env = CelesteEnv(reward_mode="best", render_mode="human")
    
    print("Connected!")
    print("\nControls:")
    print("  IJKL: Movement (I=up, K=down, J=left, L=right)")
    print("  D: Jump")
    print("  S: Dash")
    print("  A: Grab")
    print("  R: Reset environment")
    print("  Q/ESC: Quit")
    print("\nPress any key to step...")
    print("=" * 60)
    
    # Track current key states
    current_keys = {}
    
    # Reset environment
    print("\n[RESET] Calling env.reset()...")
    obs, info = env.reset()
    print(f"[RESET] Got observation: {obs.shape if obs is not None else None}")
    print(f"[RESET] Info: {info}")
    
    step_count = 0
    episode_count = 1
    
    try:
        while True:
            # Display current state
            print(f"\n--- Step {step_count} (Episode {episode_count}) ---")
            if info:
                print(f"Distance: {info.get('distance', 'N/A'):.2f}")
                print(f"Player Died: {info.get('playerDied', False)}")
                print(f"Room: {info.get('levelName', 'unknown')}")
            
            # Get current action from key states
            action = get_action_from_keys(current_keys)
            print(f"Current Action: {print_action(action)}")
            print("\nPress keys to modify action, SPACE to step, R to reset...")
            
            # Wait for key press
            key = cv2.waitKey(0)
            
            # Handle special keys
            if key == ord('q') or key == ord('Q') or key == 27:  # Q or ESC
                print("\n[QUIT] Exiting...")
                break
            
            if key == ord('r') or key == ord('R'):  # Reset
                print("\n[RESET] Calling env.reset()...")
                current_keys = {}  # Clear all keys
                obs, info = env.reset()
                print(f"[RESET] Got observation: {obs.shape if obs is not None else None}")
                print(f"[RESET] Info: {info}")
                step_count = 0
                episode_count += 1
                continue
            
            # Update key states (toggle on/off)
            if key in [ord('a'), ord('A'), ord('s'), ord('S'), ord('d'), ord('D'),
                      ord('i'), ord('I'), ord('k'), ord('K'), ord('j'), ord('J'), ord('l'), ord('L')]:
                if key in current_keys and current_keys[key]:
                    current_keys[key] = False
                    print(f"Released key: {chr(key).upper()}")
                else:
                    current_keys[key] = True
                    print(f"Pressed key: {chr(key).upper()}")
                continue  # Don't step, just update keys
            
            # Space bar or any other key = step
            if key == ord(' ') or key == 13:  # Space or Enter
                print(f"\n[STEP] Taking action: {print_action(action)}")
                obs, reward, terminated, truncated, info = env.step(action)
                
                print(f"[STEP] Got observation: {obs.shape if obs is not None else None}")
                print(f"[STEP] Reward: {reward:.3f}")
                print(f"[STEP] Terminated: {terminated}, Truncated: {truncated}")
                print(f"[STEP] Info: {info}")
                
                step_count += 1
                
                # Auto-reset if episode ended
                if terminated or truncated:
                    reason = "terminated" if terminated else "truncated"
                    print(f"\n[AUTO-RESET] Episode ended ({reason})")
                    print("Press any key to reset and continue, or Q to quit...")
                    key = cv2.waitKey(0)
                    if key == ord('q') or key == ord('Q') or key == 27:
                        break
                    
                    print("\n[RESET] Calling env.reset()...")
                    current_keys = {}  # Clear all keys
                    obs, info = env.reset()
                    print(f"[RESET] Got observation: {obs.shape if obs is not None else None}")
                    print(f"[RESET] Info: {info}")
                    step_count = 0
                    episode_count += 1
    
    except KeyboardInterrupt:
        print("\n\n[INTERRUPT] Received Ctrl+C")
    
    except Exception as e:
        print(f"\n\n[ERROR] Exception occurred: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("\n[CLEANUP] Closing environment...")
        env.close()
        cv2.destroyAllWindows()
        print("[CLEANUP] Done!")

if __name__ == "__main__":
    main()
