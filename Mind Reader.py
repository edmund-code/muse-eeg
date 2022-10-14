################################################################# 
#                      M I N D   R E A D E R                    #
#################################################################

# UNDER CONSTRUCTION !!!!!!!!!!!!!!!!!!!!!
# Mind, this is not a real mind reader :-)


# *******************  IMPORTING MODULES ********************

from tkinter.tix import IMAGE
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
from pygame.locals import *
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
bl2 = bl3 = False
jaw_clenches = 0
jaw_clenched = False
char = 0                                                        # character chosen
state = 0
alphabet = []
blink_time = []

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
    global blinks, blinked, state, bl2

    bl2_treshold = 0.7

    blink_time.append(timer())

    bl2 = False

    if blinks >= 2:
        now  = blink_time[blinks]
        then = blink_time[blinks-1]
#        print(then, now, now - then)
        if (now - then) < bl2_treshold:
            bl2 = True
#            print("Double blink")
        else:
            bl2 = False
            blinked = True
            state = 0
#            print("Single blink")

    blinks += 1
    
#        print("Blinks detected: ", blinks)
#        print()


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

    def clear_area():
        pygame.draw.rect(screen, back_color, (  HB_X, scr_height / 2 + 50, 100 * MAXHEALTH, HB_HEIGHT))

    def drawHealthMeterLeft(currentHealth):
        clear_area()
        for i in range(currentHealth): # draw red health bars
            pygame.draw.rect(screen, GREEN,   (HB_X + (10 * MAXHEALTH) - (i * 10), scr_height / 2 + 50, 20, HB_HEIGHT))
        for i in range(MAXHEALTH): # draw the white outlines
            pygame.draw.rect(screen, WHITE, (HB_X + (10 * MAXHEALTH) - (i * 10), scr_height / 2 + 50, 20, HB_HEIGHT), 1)

    def drawHealthMeterBackground(currentHealth):
        cH = currentHealth
        for i in range(cH): # draw red health bars
            pygame.draw.rect(screen, GREEN, ((scr_width/2) - (5*cH) + (i*10)-10, scr_height / 2 + 50, 20, HB_HEIGHT))
        for i in range(MAXHEALTH): # draw the white outlines
            pygame.draw.rect(screen, WHITE, (HB_X+120 + (10 * MAXHEALTH) - i * 10, scr_height / 2 + 50, 20, HB_HEIGHT), 1)

    def drawHealthMeterRight(currentHealth):
        for i in range(currentHealth): # draw red health bars
            pygame.draw.rect(screen, GREEN, (HB_X+160 + (10 * MAXHEALTH) + i * 10, scr_height / 2 + 50, 20, HB_HEIGHT))
        for i in range(MAXHEALTH): # draw the white outlines
            pygame.draw.rect(screen, WHITE, (HB_X+160 + (10 * MAXHEALTH) + i * 10, scr_height / 2 + 50, 20, HB_HEIGHT), 1)


    def write(txt, x,y, color, size):
        font = pygame.font.SysFont(None, size)
        img = font.render(txt, True, color)
        screen.blit(img, (x, y))

    def write_alphabet(list):
        pygame.draw.rect(screen, back_color, (0, scr_height / 2 + 100, scr_width, 35))      # emptying the background
        i = 0
        for c in list:
            write(c[0], 50 + (i*40), scr_height/2 + 100, WHITE, 60)
            i+=1


    def text_editor():
        global alphabet, text, blinked, state, bl2

        # Drawing selector Rectangle (x, y, width, height, border thickness, corner radius)
        pygame.draw.rect(screen, GREEN, pygame.Rect((scr_width/2)-35, (scr_height/2) + 88, 
            40, 60),  4, 6)
        write_alphabet(alphabet)

        # Text "editor" frame
        pygame.draw.rect(screen, WHITE, pygame.Rect(20, (scr_height/2) + 180, scr_width-40, 170),  1, 6)

        text = ""
        img = font.render(text, True, WHITE)
        rect = img.get_rect()

        start = timer()
        end = start
        blinked = False
        editing = True

        while editing == True:
            # Close window event
            for event in pygame.event.get():
                    if event.type == pygame.KEYDOWN:
                        if event.key == pygame.K_ESCAPE:
                            editing = False

            drawHealthMeterLeft       (int(left * MAXHEALTH))
            drawHealthMeterBackground (int(background * MAXHEALTH))
            drawHealthMeterRight      (int(right * MAXHEALTH))

            wait = 1
            end = timer()

            if (end - start) > wait:
                start = timer()
                if state == -1:
                    alphabet = np.roll(alphabet, 1,0)
                    write_alphabet(alphabet)
                elif state == 1:
                    alphabet = np.roll(alphabet, -1,0)
                    write_alphabet(alphabet)

            if blinked == True:
                print("BLINKED!!!")
                state = 0
                blinked = False
                text += alphabet[13][0]
                img = font.render(text, True, WHITE)
                rect.size=img.get_size()


            pygame.draw.rect(screen, back_color, (0, scr_height / 2 + 180, scr_width-40, 170))      # emptying the background
            pygame.draw.rect(screen, WHITE, pygame.Rect(20, (scr_height/2) + 180, scr_width-40, 170),  1, 6)
            rect = img.get_rect()
            rect.topleft = (40, 580)

            screen.blit(img, rect)
            pygame.display.update()

            if bl2 == True:
                # Text "editor" frame
                pygame.draw.rect(screen, back_color, (0, (scr_height/2) + 88, scr_width, scr_height))
                pygame.display.update()
                editing = False


    screen = pygame.display.set_mode(size)			    # clearing screen
    pygame.display.update()								# update screen

    images=[]                                           # list used for images found...
    path = 'Images/'                                    # ...in this subfolder of the current folder
    
    for image in os.listdir(path):                      # populating the list...
        if image.startswith('0') and image.endswith('.png'):  # ...only files with name [nnn].png (n=number)
            images.append(image)

    print(images)

    img_w_def = 150                                     # default image width, changing this might lead to a cascade effect...

    screen.fill(back_color)                             # surface background color


    def writeLabels():
        write("Left",       HB_X +  40, scr_height/2+65, WHITE, 24)
        write("Background", HB_X + 130, scr_height/2+65, WHITE, 24)
        write("Right",      HB_X + 280, scr_height/2+65, WHITE, 24)

    writeLabels() 

    # Drawing selector Rectangle (x, y, width, height, border thickness, corner radius)
    pygame.draw.rect(screen, GREEN, pygame.Rect((scr_width/2)-(img_w_def/2)-15, (scr_height/2)-(img_w_def/2)-25, 
        img_w_def + 20, img_w_def/1.3),  5, 7)

    clock = pygame.time.Clock()                         # clock
    start = timer()
    end = start
    nr_images = len(images)                             # how many images found?
    bl2 = False

    running = True                                      # Prepare loop condition
    while running:                                      # Event loop
        
        if bl2 == True:
            #bl2 = False
            running = False

        # Close window event
        for event in pygame.event.get():
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False

        #print(blinked, images[3])
        if blinked == True:
            write("Chosen", 20, 20, WHITE, 60)
            print("Chosen: " + images[3])
            state = 0
            if images[3][0:3] == '020':
                print("EDITING")
                text_editor()
            blinked = False
        else:
            write("Chosen", 20, 20, back_color, 60)


        end = timer()
        if (end - start) > 0.1:
            start = timer()

            images = np.roll(images, state*-1)                                  # yippii! rotating the image carousel
            
            for i in range(nr_images):
                image = pygame.image.load(path + images[i])	                    # concatenating the folder with image name

                img_width   = image.get_width()                                 # finding image width...
                img_height  = image.get_height()                                # ...and height for scaling purposes

                IMAGE_SIZE = (img_w_def, img_w_def * img_height / img_width)	# setting the size for the image
                image = pygame.transform.scale(image, IMAGE_SIZE)			    # scaling the image
                IMAGE_POSITION = ((i * (img_w_def + 20)) + 10, 290)				# placing the image

                pygame.draw.rect(screen, back_color, (IMAGE_POSITION[0] + 20,
                    IMAGE_POSITION[1]+112, scr_width, 24))

                write(images[i][4:-4],IMAGE_POSITION[0] + 20, 
                    IMAGE_POSITION[1]+112,WHITE,24)

                screen.blit(image, IMAGE_POSITION)                              # show the image

                large_image = pygame.image.load(path + images[3])               # enlarging the center image which is #3

                img_width   = large_image.get_width()                           # finding image width...
                img_height  = large_image.get_height()                          # ...and height for scaling purposes

                IMAGE_SIZE = (img_w_def*2.5, img_w_def*img_height/img_width*2.5)# setting the size for the image
                large_image = pygame.transform.scale(large_image, IMAGE_SIZE)	# scaling the image
                IMAGE_POSITION = ((scr_width/2) - IMAGE_SIZE[0] / 2, 20)        # placing the image

                screen.blit(large_image, IMAGE_POSITION)                        # show the image
        
        #print(left, background, right)

        drawHealthMeterLeft       (int(left * MAXHEALTH))
        drawHealthMeterBackground (int(background * MAXHEALTH))
        drawHealthMeterRight      (int(right * MAXHEALTH))

        # Part of event loop
        pygame.display.flip()

        #time.sleep(.8)
        clock.tick(120)



def mainmenu_background():
    global surface
    """
    Background color of the main menu, on this function user can plot
    images, play sounds, etc.
    """
    background_image = pygame.image.load("Images/eye.png")
    # Load image
   # background_image = pygame_menu.BaseImage(
   #     image_path=pygame_menu.baseimage.IMAGE_EXAMPLE_WALLPAPER
   # )

    surface.blit(background_image, (0, 0)) #example, just place the image anywhere you want
    pygame.display.flip()
    #background_image.draw(surface)


def init_menu():
    global keyboard, alphabet, surface

    keyboard = Controller()
    
    surface = create_example_window('Mind Reader', size)
 

    menu = pygame_menu.Menu(
        height=size[1],
        onclose=pygame_menu.events.EXIT,  # User press ESC button
        theme=pygame_menu.themes.THEME_BLUE,
        title='Mind Reader',
        width=size[0]
    )

    
    chars = list(string.ascii_uppercase)
    chars.append(' ')
    #print(chars)

    def createList(r1, r2):
        return list(range(r1, r2+1))

    numbers = createList(65,90)
    numbers.append(32)
    #print(numbers)

    alphabet = list(zip(chars,numbers))
    #print(alphabet)

#    user_name = menu.add.text_input('Name: ', default='John Doe', maxchar=10)
#    menu.add.selector('Difficulty: ', alphabet, onchange=set_difficulty)
    menu.add.button('Play', start_the_game)
#    menu.add.button('Init', initiate_tf)
    menu.add.button('Quit', pygame_menu.events.EXIT)

    while True:
        #if menu.is_enabled():
        menu.mainloop(surface, mainmenu_background)

        pygame.display.flip()

    #menu.mainloop(surface)


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
