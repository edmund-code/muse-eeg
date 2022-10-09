################################################################# 
#                      M I N D   R E A D E R                    #
#################################################################

# UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!!!!!
# Mind, this is not a real mind reader :-)


# *******************  IMPORTING MODULES ********************

from turtle import back
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer

import time
import random
import threading
import numpy as np
import tensorflow as tf
from nltk import flatten

import pygame
import pygame_menu
from pygame_menu.examples import create_example_window
import string
import os

from typing import Tuple, Any
from pynput.keyboard import Key, Controller
from timeit import default_timer as timer



# *********************  G L O B A L S *********************

alpha = beta = delta = theta = gamma = [-1,-1,-1,-1]
all_waves = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]
all_samples = []

sample_nr = 0
expected_samples = 30                                           # there are 5 frequencies (alpa...gamma) and 4 sensors, if all 4 sensors are used
                                                                # this should be 5 x 4 = 20, the frequency is 10 Hz. 2 seconds of data with all
                                                                # 4 sensors = 2 * 5 * 4 * 10 = 400. 

confidence_threshold = 0.5                                      # default in Edge Impulse is 0.6
global isFailed
left = right = background = 0

blinks = 0                                                      # amount of blinks
blinked = False                                                 # did you blink?
char = 0                                                        # character chosen
state = 0
alphabet = []

start = timer()
secs = 3

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
    lite_file = "ei-muse_separately_recorded_events-nn-classifier-tensorflow-lite-float32-model (7).lite"

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
    global blinks, blinked, state

    blinks += 1
    blinked = True

    if blinked == True:
        print(chr(67), end = "")
#        blinked = False
        state = 0

#    print("Blink detected ")


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
    global start, end

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

    end = timer()
    if (end - start) >= secs:                           # if we've waited enough for the current event
        start = timer()                                 # getting current time
        if state == 1:
            keyboard.release(Key.right)
            keyboard.press(Key.right)
        elif state == -1:
            keyboard.release(Key.left)
            keyboard.press(Key.left)


# ****************** EEG handlers END ******************


# ******** INFERENCE ******** 
def inference():
    global score, expected, choice, blinks, blinked, state
    global background, left, right

    input_samples = np.array(all_samples, dtype=np.float32)
    input_samples = np.expand_dims(input_samples, axis=0)

    # input_details[0]['index'] = the index which accepts the input
    interpreter.set_tensor(input_details[0]['index'], input_samples)

    # run the inference
    interpreter.invoke()

    # output_details[0]['index'] = the index which provides the input
    output_data = interpreter.get_tensor(output_details[0]['index'])

    # finding output data
    background  = output_data[0][0]
    left        = output_data[0][1]
    right       = output_data[0][2]

    # checking if over confidence threshold
    if left >= confidence_threshold:
        choice = "Left"
        # keyboard.release(Key.left)
        # keyboard.press(Key.left)
        state = -1
    elif right >= confidence_threshold:
        choice = "Right"
        # keyboard.release(Key.right)
        # keyboard.press(Key.right)
        state = 1
    else:
        choice = "----"

#    print(f"Left:{left:.4f} - Background:{background:.4f}   Right:{right:.4f}    {choice}          ")


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


# *************** INITIALISING ***************
def start_threads():

    # starting the Muse communication in separate thread
    thread = threading.Thread(target=dispatch)
    thread.daemon = True
    thread.start()

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Clears the screen ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def clear_screen():
#	screen = pygame.display.set_mode(size, pygame.FULLSCREEN)			# clearing screen
	screen = pygame.display.set_mode(size)			                    # clearing screen
	pygame.display.update()												# update screen
   	

# ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~ Shows a random image ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
def show_image():
    global screen, blinked, state, alphabet

    scr_width  = size[0]                                # surface width
    scr_height = size[1]                                # surface height
    MAXHEALTH = 9
    GREEN = (48, 141, 70)
    WHITE = (200,200,200)
    back_color = (55,55,55)
    HB_X = (scr_width / 2) - 185
    HB_HEIGHT = 11

    font = pygame.font.SysFont(None, 60)


    def drawHealthMeterLeft(currentHealth):
        pygame.draw.rect(screen, back_color, (  HB_X, scr_height / 2 + 50, 100 * MAXHEALTH, HB_HEIGHT))
        for i in range(currentHealth): # draw red health bars
            pygame.draw.rect(screen, GREEN,   (HB_X + (10 * MAXHEALTH) - (i * 10), scr_height / 2 + 50, 20, HB_HEIGHT))
        for i in range(MAXHEALTH): # draw the white outlines
            pygame.draw.rect(screen, WHITE, (HB_X + (10 * MAXHEALTH) - (i * 10), scr_height / 2 + 50, 20, HB_HEIGHT), 1)

    def drawHealthMeterBackground(currentHealth):
        for i in range(currentHealth): # draw red health bars
            pygame.draw.rect(screen, GREEN,   (HB_X+120 + (10 * MAXHEALTH) - i * 10, scr_height / 2 + 50, 20, HB_HEIGHT))
        for i in range(MAXHEALTH): # draw the white outlines
            pygame.draw.rect(screen, WHITE, (HB_X+120 + (10 * MAXHEALTH) - i * 10, scr_height / 2 + 50, 20, HB_HEIGHT), 1)

    def drawHealthMeterRight(currentHealth):
        for i in range(currentHealth): # draw red health bars
            pygame.draw.rect(screen, GREEN,   (HB_X+240 + (10 * MAXHEALTH) - i * 10, scr_height / 2 + 50, 20, HB_HEIGHT))
        for i in range(MAXHEALTH): # draw the white outlines
            pygame.draw.rect(screen, WHITE, (HB_X+240 + (10 * MAXHEALTH) - i * 10, scr_height / 2 + 50, 20, HB_HEIGHT), 1)

    def write(txt, x,y, color):
        img = font.render(txt, True, color)
        screen.blit(img, (x, y))

    def write_alphabet(list):
        pygame.draw.rect(screen, back_color, (0, scr_height / 2 + 100, scr_width, 100))
        i = 0
        for c in list:
            #print(c,c[0])
            write(c[0], 50 + (i*40), scr_height/2 + 100, WHITE)
            i+=1


    screen = pygame.display.set_mode(size)			    # clearing screen
    pygame.display.update()								# update screen

    images=[]                                           # list used for images found...
    path = 'Images/'                                    # ...in this subfolder of the current folder
    
    for image in os.listdir(path):                      # populating the list...
        if len(image) == 7 and image.endswith('.png'):  # ...only files with name [nnn].png (n=number)
            images.append(image)

    print(images)

    img_w_def = 150                                     # default image width, changing this might lead to a cascade effect...

    screen.fill(back_color)                           # surface background color
    color = (48, 141, 70)                               # image selector color

    # Drawing selector Rectangle (x, y, width, height, border thickness, corner radius)
    pygame.draw.rect(screen, color, pygame.Rect((scr_width/2)-(img_w_def/2)-15, (scr_height/2)-(img_w_def/2)-15, 
        img_w_def + 20, img_w_def/1.3),  5, 7)

    clock = pygame.time.Clock()                         # clock

    running = True                                      # Prepare loop condition
    while running:                                      # Event loop
    
        # Close window event
        for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

    
        nr_images = len(images)                                             # how many images found?
        images = np.roll(images, state*-1)                                  # yippii! rotating the image carousel
        
        for i in range(nr_images):
            image = pygame.image.load(path + images[i])	                    # concatenating the folder with image name

            img_width   = image.get_width()                                 # finding image width...
            img_height  = image.get_height()                                # ...and height for scaling purposes

            IMAGE_SIZE = (img_w_def, img_w_def * img_height / img_width)	# setting the size for the image
            image = pygame.transform.smoothscale(image, IMAGE_SIZE)			# scaling the image
            IMAGE_POSITION = ((i * (img_w_def + 20)) + 10, 300)				# placing the image

            screen.blit(image, IMAGE_POSITION)                              # show the image

            large_image = pygame.image.load(path + images[3])               # enlarging the center image which is #3

            img_width   = large_image.get_width()                           # finding image width...
            img_height  = large_image.get_height()                          # ...and height for scaling purposes

            IMAGE_SIZE = (img_w_def*2.5, img_w_def*img_height/img_width*2.5)	# setting the size for the image
            large_image = pygame.transform.smoothscale(large_image, IMAGE_SIZE)	# scaling the image
            IMAGE_POSITION = ((scr_width/2) - IMAGE_SIZE[0] / 2, 20)        # placing the image

            screen.blit(large_image, IMAGE_POSITION)                        # show the image

        
        print(left, background, right)

        drawHealthMeterLeft       (int(left * MAXHEALTH))
        drawHealthMeterBackground (int(background * MAXHEALTH))
        drawHealthMeterRight      (int(right * MAXHEALTH))

        print(blinked, images[3])
        if images[3] == '020.png' and blinked == True:
            write_alphabet(alphabet)
            alphabet = np.roll(alphabet, 1,0)

            write("Chosen", 20, 20, WHITE)
            print("Chosen")
            state = 0
        else:
            write("Chosen", 20, 20, back_color)

        blinked = False


        # Part of event loop
        pygame.display.flip()
        time.sleep(.1)
        clock.tick(60)



def init_menu():
    global keyboard, alphabet

    keyboard = Controller()
    
    surface = create_example_window('Mind Reader', size)

    menu = pygame_menu.Menu(
        height=size[1],
        theme=pygame_menu.themes.THEME_BLUE,
        title='Mind Reader',
        width=size[0]
    )

    
    chars = list(string.ascii_uppercase)
    chars.append(' ')
    print(chars)

    def createList(r1, r2):
        return list(range(r1, r2+1))

    numbers = createList(65,90)
    numbers.append(32)
    print(numbers)

    alphabet = list(zip(chars,numbers))
    print(alphabet)

#    user_name = menu.add.text_input('Name: ', default='John Doe', maxchar=10)
    menu.add.selector('Difficulty: ', alphabet, onchange=set_difficulty)
    menu.add.button('Play', start_the_game)
    menu.add.button('Init', initiate_tf)
    menu.add.button('Quit', pygame_menu.events.EXIT)

    menu.mainloop(surface)


def set_difficulty(selected: Tuple, value: Any) -> None:
    global blinked, char

    """
    Set the difficulty of the game.
    """
#    print(f'Set difficulty to {selected[0]} ({value})')
    char = value
    state = 0


def start_the_game() -> None:
    pygame.init()
    pygame.font.init()
    clear_screen()
    show_image()



if __name__ == "__main__":
    size = (1200, 768)	
    initiate_tf()
    start_threads()
    init_menu()
