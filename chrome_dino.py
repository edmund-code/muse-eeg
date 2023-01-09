import pyautogui # pip install pyautogui
from PIL import Image, ImageGrab # pip install pillow
# from numpy import asarray
import time
import threading

def click(key):
    pyautogui.keyDown(key)
    return

def run_game():
    while True:
        time.sleep(0.5)
        click('up') 

thread = threading.Thread(target=run_game)
thread.daemon = True
thread.start()

while True:
    time.sleep(4)
    print("ee")