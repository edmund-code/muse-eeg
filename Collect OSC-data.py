#####################################################################
#       Record OSC-data from Muse EEG through Mind Monitor app      #
#####################################################################
#           Coded     : Thomas Vikström (2022)                      #
#           Credits   : James Clutterbuck, Mind Monitor developer   #
#           Requires  : pip install python-osc                      #
#####################################################################


# *******************  IMPORTING MODULES ********************
from datetime import datetime
from pythonosc import dispatcher
from pythonosc import osc_server
from timeit import default_timer as timer

# *********************  G L O B A L S *********************
alpha = beta = delta = theta = gamma = [-1,-1,-1,-1]
all_waves = [-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1]

ip = "0.0.0.0"
port = 5000

filePath = 'Blinks/Noise.csv'
filePath2 = 'Blinks/'

f = open (filePath,'a+')
header = 'timestamp,A9,A7,A8,A10,B9,B7,B8,B10,D9,D7,D8,D10,T9,T7,T8,T10,G9,G7,G8,G10\n'

current_file = ''
current_event = 0
row = 0

secs = 2
start = timer()
recording = False

# You have 2 choices when recording EEG-data: 
# 1) Show an event to record for a predefined time, this is preferable as it creates files that directly can be uploaded to Edge Impulse
# 2) Record a long stream of EEG-data into one file that needs to be split into files recognizable by Edge Impulse
# if 1) then record_many should be True, if 2) then it should be False

record_many = True
                                       
# Put the events to record in this dictionary within "" and after : the seconds
# This is used for the Blink Pong game
rec_dict = {
    "1"     : 2,
    "Noise" : 2
}  

# This is used for the Mind Reader app, uncomment below rows and comment all other rec_dict rows
# rec_dict = {
#     "Background" : 3,
#     "Left"       : 3,
#     "Right"      : 3
# }

# Yet another example
# rec_dict = {
#     "Left"    : 2,
#     "Right"   : 2,
#     "Noise"   : 2
# }  




# ==========================================================
# *******************  F U N C T I O N S *******************
# ==========================================================


# ****************** EEG-handlers ******************

def alpha_handler(address: str,*args):
    global alpha, beta, delta, theta, gamma

    if (len(args)==5):                                              # If OSC Stream Brainwaves = All Values
        for i in range(1,5):
            all_waves[i-1] = args[i]


def beta_handler(address: str,*args):
    global alpha, beta, delta, theta, gamma

    if (len(args)==5):                                              # If OSC Stream Brainwaves = All Values
        for i in range(1,5):
            all_waves[i-1 + 4] = args[i]


def delta_handler(address: str,*args):
    global alpha, beta, delta, theta, gamma

    if (len(args)==5):                                              # If OSC Stream Brainwaves = All Values
        for i in range(1,5):
            all_waves[i-1 + 8] = args[i]


def theta_handler(address: str,*args):
    global alpha, beta, delta, theta, gamma

    if (len(args)==5):                                              # If OSC Stream Brainwaves = All Values
        for i in range(1,5):
            all_waves[i-1 + 12] = args[i]


def gamma_handler(address: str,*args):
    global alpha, beta, delta, theta, gamma
    global record_many

    if (len(args)==5):                                              # If OSC Stream Brainwaves = All Values
        for i in range(1,5):
            all_waves[i-1 + 16] = args[i]

    if record_many == False:
        for i in range(0,19):
            f.write(str(all_waves[i]) + ",")
        f.write(str(all_waves[19]))
        f.write("\n")
    else:
        show_event()

# ********* Showing one event at a time *********
def show_event():
    global current_event, current_file
    global start, end, secs, row

    end = timer()
    if (end - start) >= secs:                                       # if we've waited enough for the current event
        start = timer()                                             # getting current time
        ev = list(rec_dict.items())[current_event][0]               # fetching current event
        secs = list(rec_dict.items())[current_event][1]             # fetching seconds for current event
        row = 0
        
        dateTimeObj = datetime.now()
        timestampStr = dateTimeObj.strftime("%Y-%m-%d %H_%M_%S.%f")

        ev = list(rec_dict.items())[current_event][0]
        current_file = filePath2 + ev + '.' + timestampStr + '.csv'
        evf = open (current_file,'a+')
        evf.write(header)


        print(f"Think:\t {ev}   \t\t{secs}  seconds")

        dict_length = len(rec_dict)                                 # how many events in the dictionary
        if current_event < dict_length-1:                           # if end not reached...
            current_event += 1                                      # ...increasing counter
        else:
            current_event = 0                                       # if end reached, starting over
    else:
        if current_file != '':
            evf = open (current_file,'a+')

            evf.write(str(row) + ',')
            row += 1
            for i in range(0,19):
                evf.write(str(all_waves[i]) + ",")
            evf.write(str(all_waves[19]))
            evf.write("\n")


def marker_handler(address: str,i):
    global recording, record_many, start, end

    dateTimeObj = datetime.now()
    timestampStr = dateTimeObj.strftime("%Y-%m-%d %H:%M:%S.%f")
    markerNum = address[-1]
    f.write(timestampStr+",,,,/Marker/"+markerNum+"\n")
    start = timer()
    if (markerNum=="1"):        
        recording = True
        print("Recording Started.")
    if (markerNum=="2"):
        f.close()
        server.shutdown()
        print("Recording Stopped.") 

    if (markerNum=="3"):
        start = timer()

        for i in range(len(rec_dict)):
            ev = list(rec_dict.items())[i][0]
            evf = open (filePath2 + ev + '.csv','a+')
            evf.write(header)

        if record_many == False:
            record_many = True
            show_event()
        else:
            record_many = False


if __name__ == "__main__":
    dispatcher = dispatcher.Dispatcher()

    dispatcher.map("/Marker/*", marker_handler)
    dispatcher.map("/muse/elements/delta_absolute", delta_handler,0)
    dispatcher.map("/muse/elements/theta_absolute", theta_handler,1)
    dispatcher.map("/muse/elements/alpha_absolute", alpha_handler,2)
    dispatcher.map("/muse/elements/beta_absolute",  beta_handler,3)
    dispatcher.map("/muse/elements/gamma_absolute", gamma_handler,4)


    server = osc_server.ThreadingOSCUDPServer((ip, port), dispatcher)
    print("Listening on UDP port "+str(port)+"\nSend Marker 1 to Start recording, Marker 2 to Stop Recording, Marker 3 to show events. ")
    print()
    server.serve_forever()