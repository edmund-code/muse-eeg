import threading
from pythonosc.dispatcher import Dispatcher
from pythonosc.osc_server import BlockingOSCUDPServer
import pygame
from pygame import *
IP = "0.0.0.0"  # listening on all IP-addresses
PORT = 5000     # on this port


# ********** Handling blinks **********
def blink_handler(address, *args):
    blink_event = pygame.event.Event(pygame.KEYDOWN, unicode=" ", key=pygame.K_SPACE, mod=pygame.KMOD_NONE) #create the event
    pygame.event.post(blink_event) #add the event to the queue
    print("Blink detected (%d)" % (1))

# ******* Handling jaw clenches *******
def jaw_handler(address, *args):
    jaw_event = pygame.event.Event(pygame.KEYDOWN, unicode="", key=pygame.K_DOWN, mod=pygame.KMOD_NONE) #create the event
    pygame.event.post(jaw_event) #add the event to the queue
    print("Jaw Clench detected (%d)" % (1))


# ====================== MUSE COMMUNICATION ==========================

# ******** Muse communication 1 ********
def get_dispatcher():
    dispatcher = Dispatcher()
    dispatcher.map("/muse/elements/blink", blink_handler)
    dispatcher.map("/muse/elements/jaw_clench", jaw_handler)
    
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
def start_listen():
    # starting the Muse communication in separate thread
    thread = threading.Thread(target=dispatch)
    thread.daemon = True
    thread.start()