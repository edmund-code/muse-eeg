One of the easiest game one can play with a consumer-grade EEG device like Muse2 is the Dino Run - a game can be played with just single click (default is the space key) 

<img width="560" alt="image" src="https://github.com/edmund-code/muse-eeg/assets/52833150/6af61233-4d71-48ce-806c-fe1939246336">

For this to work we only need 4 comonents

1. Code to capture EEG signal from the device
2. Model to classify the EEG signal captured in the previous step
3. Code that allows Python to control keyboard/mouse
4. The game itself



## 1. Code to capture EEG signal from the device
### Installation
First, you need to install [`pylsl`](https://github.com/labstreaminglayer/pylsl):
```bash
pip install pylsl
```

## 2. Model to classify the EEG signal captured in the previous step


---
## 3. Simulating Key Press in Python
https://pyautogui.readthedocs.io/en/latest/
```pip install pyautogui```


To simulate a key press event in Python that interacts with other programs, we can use the `pyautogui` library. This library allows for the automation of the mouse and keyboard and can send keystrokes to external applications.

### Installation
First, you need to install `pyautogui`:
```bash
pip install pyautogui
```

### Python Script

Here's a simple script to simulate a key press:

```python
import pyautogui
import time

def simulate_key_press(key):
    # Simulate a key press
    pyautogui.press(key)

# Give you time to switch to the program where you want the key press to be sent
print("Switch to the program where you want the key press to be sent. You have 5 seconds.")
time.sleep(5)

# Simulating a key press, for example, the 'a' key
simulate_key_press('a')
```

### Caution

- `pyautogui` sends keystrokes to the active window, so ensure that the correct window is in focus.
- Use such scripts carefully, as automated key presses can interfere with system operations. Always have an emergency stop plan, like stopping the script with `KeyboardInterrupt` (Ctrl+C).

---


## 4. The game itself
For Chrome users, simply type [chrome://dino/](chrome://dino/) in the URL.

For non-Chrome users, there are several alternatives, for example, https://offline-dino-game.firebaseapp.com/
