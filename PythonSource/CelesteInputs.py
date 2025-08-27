import pynput
import time

class CelesteInputs:
    def __init__(self, up=False, down=False, left=False, right=False, jump=False, dash=False, grab=False):
        self.up = up
        self.down = down
        self.left = left
        self.right = right
        self.jump = jump
        self.dash = dash
        self.grab = grab

    def update_keyboard(self):
        # Update the keyboard based on the current attributes
        keyboard = pynput.keyboard.Controller()
        keyboard.press(pynput.keyboard.Key.up) if self.up else keyboard.release(pynput.keyboard.Key.up)
        keyboard.press(pynput.keyboard.Key.down) if self.down else keyboard.release(pynput.keyboard.Key.down)
        keyboard.press(pynput.keyboard.Key.left) if self.left else keyboard.release(pynput.keyboard.Key.left)
        keyboard.press(pynput.keyboard.Key.right) if self.right else keyboard.release(pynput.keyboard.Key.right)
        keyboard.press(pynput.keyboard.KeyCode.from_char('d')) if self.jump else keyboard.release(pynput.keyboard.KeyCode.from_char('d'))
        keyboard.press(pynput.keyboard.KeyCode.from_char('s')) if self.grab else keyboard.release(pynput.keyboard.KeyCode.from_char('s'))
        keyboard.press(pynput.keyboard.KeyCode.from_char('a')) if self.grab else keyboard.release(pynput.keyboard.KeyCode.from_char('a'))

    def reset_keyboard(self):
        self.up = False
        self.down = False
        self.left = False
        self.right = False
        self.jump = False
        self.dash = False
        self.grab = False
        self.update_keyboard()

    def restart_chapter_celeste(self):
        keyboard = pynput.keyboard.Controller()
        keyboard.press(pynput.keyboard.KeyCode.from_char('r'))
        time.sleep(0.1)
        keyboard.release(pynput.keyboard.KeyCode.from_char('r'))
        keyboard.press(pynput.keyboard.KeyCode.from_char('d'))
        time.sleep(0.1)
        keyboard.release(pynput.keyboard.KeyCode.from_char('d'))

    def to_action(self):
        return [
            int(self.up),
            int(self.down),
            int(self.left),
            int(self.right),
            int(self.jump),
            int(self.dash),
            int(self.grab)
        ]
    
    @staticmethod
    def from_action(action):
        inputs = CelesteInputs()
        inputs.up = bool(action[0])
        inputs.down = bool(action[1])
        inputs.left = bool(action[2])
        inputs.right = bool(action[3])
        inputs.jump = bool(action[4])
        inputs.dash = bool(action[5])
        inputs.grab = bool(action[6])
        return inputs
