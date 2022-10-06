################################################################# 
#                      M I N D   R E A D E R                    #
#################################################################

# UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!!!!!
# Mind, this is not a real mind reader :-)


# *******************  IMPORTING MODULES ********************

from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
from tkinter import *
import tkinter as tk
import time
import random
import threading
import numpy as np
import tensorflow as tf
from nltk import flatten

import pygame_menu
from pygame_menu.examples import create_example_window
import string

from typing import Tuple, Any



# *********************  G L O B A L S *********************

alpha = beta = delta = theta = gamma = [-1,-1,-1,-1]
all_waves = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
all_samples = []

sample_nr = 0
expected_samples = 20                                           # there are 5 frequencies (alpa...gamma) and 4 sensors, if all 4 sensors are used
                                                                # this should be 5 x 4 = 20, the frequency is 10 Hz. 2 seconds of data with all
                                                                # 4 sensors = 2 * 5 * 4 * 10 = 400. 

confidence_threshold = 0.6                                      # default in Edge Impulse is 0.6
global isFailed
blinks = 0                                                      # amount of blinks
blinked = False                                                 # did you blink?

IP = "0.0.0.0"                                                  # listening on all IP-addresses
PORT = 5000                                                     # on this port


# ==========================================================
# *******************  F U N C T I O N S *******************
# ==========================================================


# *********** Initiates TensorFlow Lite ***********
def initiate_tf():
    global interpreter, input_details, output_details

    ####################### TF Lite path and file ######################
    path = "models/"
    lite_file = "ei-muse-blinks-separately-recorded-nn-classifier-tensorflow-lite-float32-model (6).lite"

    ####################### INITIALIZE TF Lite #########################
    # Load TFLite model and allocate tensors.
    interpreter = tf.lite.Interpreter(model_path = path + lite_file)

    # Get input and output tensors.
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()

    # Allocate tensors
    interpreter.allocate_tensors()

    # Printing input and output details for debug purposes in case anything is not working
    print (input_details)
    print(output_details)



# ****************** EEG handlers START ******************


# ********** Handling blinks **********
def blink_handler(address, *args):
    global blinks, blinked

    blinks += 1
    blinked = True
    print("Blink detected ")

# ******* Handling jaw clenches *******
# (no functionality tied to them)
def jaw_handler(address, *args):
    global jaw_clenches, jaw_clenched

    jaw_clenches += 1
    jaw_clenched = True
    print("Jaw Clench detected")


# ***** ALPHA waves *****
def alpha_handler(address: str,*args):
    global alpha, beta, delta, theta, gamma

    if (len(args)==5):                              #If OSC Stream Brainwaves = All Values
        for i in range(1,5):
            all_waves[i-1] = args[i]

# ***** BETA waves *****
def beta_handler(address: str,*args):
    global alpha, beta, delta, theta, gamma

    if (len(args)==5):                              #If OSC Stream Brainwaves = All Values
        for i in range(1,5):
            all_waves[i-1 + 4] = args[i]

# ***** DELTA waves *****
def delta_handler(address: str,*args):
    global alpha, beta, delta, theta, gamma

    if (len(args)==5):                              #If OSC Stream Brainwaves = All Values
        for i in range(1,5):
            all_waves[i-1 + 8] = args[i]

# ***** THETA waves *****
def theta_handler(address: str,*args):
    global alpha, beta, delta, theta, gamma

    if (len(args)==5):                              #If OSC Stream Brainwaves = All Values
        for i in range(1,5):
            all_waves[i-1 + 12] = args[i]

# ***** GAMMA waves *****
def gamma_handler(address: str,*args):
    global alpha, beta, delta, theta, gamma
    global sample_nr, expected_samples, all_samples, sample

    if (len(args)==5):                                  # If OSC Stream Brainwaves = All Values
        for i in range(1,5):
            all_waves[i-1 + 16] = args[i]

        all_samples.append(all_waves)                   # Appending all data...
        sample_nr += 1 
        
        if sample_nr == expected_samples:               # Collected all samples...
            all_samples = flatten(all_samples)          # ...and flattening them
            inference()                                 # Inference function call 
            sample_nr = 0
            all_samples.clear()
            all_samples = []

# ****************** EEG handlers END ******************


# ******** INFERENCE ******** 
def inference():
    global score, expected, choice, blinks, blinked

    input_samples = np.array(all_samples, dtype=np.float32)
    input_samples = np.expand_dims(input_samples, axis=0)

    # input_details[0]['index'] = the index which accepts the input
    interpreter.set_tensor(input_details[0]['index'], input_samples)

    # run the inference
    interpreter.invoke()

    # output_details[0]['index'] = the index which provides the input
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # finding output data
    blink       = output_data[0][0]
    background  = output_data[0][1]

    # checking if over confidence threshold
    if blink >= confidence_threshold:
        choice = "Blink"
        blinks += 1
        blinked = True
    elif background >= confidence_threshold:
        choice = "Background"
    else:
        choice = "----"

    print(f"Blink:{blink:.4f} - Background:{background:.4f}     {choice}          ")


# ====================== MUSE COMMUNICATION ==========================

# ******** Muse communication 1 ********
def get_dispatcher():
    dispatcher = Dispatcher()
    dispatcher.map("/muse/elements/blink", blink_handler)
    dispatcher.map("/muse/elements/jaw_clench", jaw_handler)
    dispatcher.map("/muse/elements/delta_absolute", delta_handler,0)
    dispatcher.map("/muse/elements/theta_absolute", theta_handler,1)
    dispatcher.map("/muse/elements/alpha_absolute", alpha_handler,2)
    dispatcher.map("/muse/elements/beta_absolute" , beta_handler,3)
    dispatcher.map("/muse/elements/gamma_absolute", gamma_handler,4)
    
    return dispatcher

# ******** Muse communication 2 ********
def start_blocking_server(ip, port):
    server = BlockingOSCUDPServer((ip, port), dispatcher)
    server.serve_forever()  # Blocks forever

# ******** Muse communication 3 ********
def dispatch():
    global dispatcher

    dispatcher = get_dispatcher()
    start_blocking_server(IP, PORT)


# ========================== G A M E  ==============================


# *********** Moving the ball ***********
def move_ball(ball, sp, score):
    global wait, blink_window_wait, blinked

        
    if blinked == True:                                         # Did you blink? Yes...
        c.itemconfigure(blink_window, state='normal')           # ...showing a message that you did...
        blinked = False

    if blink_window_wait == 50:                                 # ...for a short while...
        blink_window_wait = 0
        c.itemconfigure(blink_window, state='hidden')           # ...until we hide it
    else:
        blink_window_wait += 1

        
    what = blinks % 4                                           # % = modulo, we have 4 states, 2 of them pausing:
    if what == 1:                                               # -> STOP -> LEFT -> STOP -> RIGHT
        movepaddleLR(paddle, 'l', 0-paddle_speed)               # moving left
    elif what == 0 or what == 2:
        movepaddleLR(paddle, 'stop', 0)                         # stopping
    elif what == 3:
        movepaddleLR(paddle, 'r', paddle_speed)                 # moving right




# *************** INITIALISING ***************
def start_threads():

    # starting the Muse communication in separate thread
    thread = threading.Thread(target=dispatch)
    thread.daemon = True
    thread.start()



if __name__ == "__main__":
    initiate_tf()
    start_threads()
    pong()                                                                          # Start Ponging!
