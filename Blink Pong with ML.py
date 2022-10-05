################################################################# 
#                       B L I N K   P O N G                     #
#################################################################

# Very simple Pong game, assembled from bits and pieces by Thomas Vikström
# The objective is to try to increase the score by hitting the ball
# Needs Muse EEG-device and MindMonitor app
#
# Usage: Blink to move the paddle in one direction, 
#        following blink will stop the paddle,
#        and yet following move it in the opposite direction


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
    path = "Models/"
    lite_file = "ei-muse-blinks-separately-recorded-nn-classifier-tensorflow-lite-float32-model.lite"

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

# *********** Moving the paddle ***********
def movepaddleLR(paddle, dir, x, y = 0):
    x1, y1, x2, y2 = c.coords(paddle)                               # fetching coordinates
    if ((x1 > 0 and dir == 'l') 
            or (x2 < 400 and dir == 'r')):                          # if within bounds, moving left or right
        c.move(paddle, x, y)
        c.update()
    elif dir == 'stop':                                             # if asked to stop moving
        c.move(paddle, 0, 0)
        c.update()


# *********** Moving the ball ***********
def move_ball(ball, sp, score):
    global wait, blink_window_wait, blinked

    s = random.randint(-sp, sp)                                     # When starting...
    x, y = s, 0-sp                                                  # ...ball moving in random direction. 0-sp is used to get negative value
    c.move(ball, x, y)

    for p in range(1, 500000):                                      # "Infinite" loop
        l, t, r, b = c.coords(ball)                                 # fetching coordinates
        txtS.delete(0, END)                                         # emptying the score window...
        txtS.insert(0, "Score: " + str(score))                      # ...and refilling it again with current score

        # Need to change direction when hitting the wall. There are eight options
        if(r >= 400 and x >= 0 and y < 0):                          # Ball moving ↗ and hit right wall
            x, y = 0-sp, 0-sp
        elif(r >= 400 and x >= 0 and y >= 0):                       # Ball moving ↘ and hit right wall
            x, y = 0-sp, sp
        elif(l <= 0 and x < 0 and y < 0):                           # Ball moving ↖ and hit left wall
            x, y = sp, 0-sp
        elif(l <= 0 and x < 0 and y >= 0):                          # Ball moving ↙ and hit left wall
            x, y = sp, sp
        elif(t <= 0 and x >= 0 and y < 0):                          # Ball moving ↗ and hit top wall
            x, y = sp, sp
        elif(t <= 0 and x < 0 and y < 0):                           # Ball moving ↖ and hit top wall
            x, y = 0-sp, sp
        elif(b >= 385):                                             # Ball reached paddle level. Check if paddle touches ball
            tchPt = l + 10                                          # Size is 20. Half of it.
            bsl, bst, bsr, bsb = c.coords(paddle)
            if(tchPt >= bsl and tchPt <= bsr):                      # Ball touches paddle
                n = random.randint(-sp, sp)
                x, y = n, 0-sp
                score += 1
            else:                                                   # Oh no, we hit the bottom!
                wait += 1                                           # Waiting to let the ball hit the bottom, more or less
                if wait == 5:
                    wait = 0
                    global isFailed
                    isFailed = True
                    break                                           # Breaking out of the function
        
        time.sleep(.050)                                            # Dare to remove this? Speed of the ball

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

        c.move(ball, x, y)
        c.update()


# ***** RESTARTING AFTER HITTING THE BOTTOM *****
def restart():
    global isFailed
    if(isFailed == True):
        isFailed = False
        c.moveto(paddle, 150, 385)
        c.moveto(ball, 190, 365)
        move_ball(ball, ball_speed, score)


# *************** INITIALISING ***************
def pong():
    global c, ball, txtS, paddle, blink_window, ball_speed, paddle_speed
    global score, wait, blink_window_wait

    # misc. initialisation stuff
    root = Tk()
    root.minsize(400,400)
    root.title("Blink Pong")
    paddle_speed = 5
    ball_speed = 5
    score = 10
    wait = 0
    blink_window_wait = 0
    global isFailed
    isFailed = False

    # starting the Muse communication in separate thread
    thread = threading.Thread(target=dispatch)
    thread.daemon = True
    thread.start()

    # canvas related stuff
    c = Canvas(width=400, height=400, background='#a0aa00')
    c.pack()
    paddle = c.create_rectangle(150, 385, 250, 400, fill='blue', outline='blue')    # paddle
    ball = c.create_oval(190, 365, 210, 385, fill='red', outline='red')             # ball
    txtS = tk.Entry(c, text='0')                                                    # score
    txtScore = c.create_window(300, 0, anchor='nw', window=txtS)

    # blink detection window related stuff
    blink_label = tk.Label(c, text='Blink detected')
    blink_window = c.create_window(10, 10, anchor='nw', window=blink_label)
    c.itemconfigure(blink_window, state='hidden')

    # left and right keys can be used when the paddle is STOPPED (blink once if the keys are not working)
    root.bind("<KeyPress-Left>", lambda event: movepaddleLR(paddle, 'l', 0-paddle_speed))
    root.bind("<KeyPress-Right>", lambda event: movepaddleLR(paddle, 'r', paddle_speed))
    
    # main "infinite" loop
    while 1:
        move_ball(ball, ball_speed, score)                                          # if the ball hit the bottom, we escaped out...
        score -= 1                                                                  # ...of the function and decrease the score
    root.mainloop()


if __name__ == "__main__":
    initiate_tf()
    pong()                                                                          # Start Ponging!
