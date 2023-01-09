"""

"""

import numpy as np  # Module that simplifies computations on matrices
from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data
import utils  # MUSE-LSL utility functions

import collections
from statistics import mean, stdev

import pyautogui # pip install pyautogui

# ********** Handling blinks **********
def blink_handler(count, avg, std):
    pyautogui.keyDown('up')
    print("pyGame Blink detected %d, avg=%.3f, std=%.3f" % (count, avg, std))

# Index of the channel(s) (electrodes) to be used
# 0 = left ear, 1 = left forehead, 2 = right forehead, 3 = right ear
INDEX_CHANNEL = [0]

def start_listen():
    """ 1. CONNECT TO EEG STREAM """
    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    # Set active EEG stream to inlet and apply time correction
    print("Start acquiring data")
    inlet = StreamInlet(streams[0], max_chunklen=12)
    eeg_time_correction = inlet.time_correction()

    # Get the stream info and description
    info = inlet.info()
    description = info.desc()

    buffer_size = 64
    """ 2. INITIALIZE BUFFERS """
    # Initialize raw EEG data buffer
    eeg_buffer = np.zeros((int(buffer_size), 1))
    filter_state = None  # for use with the notch filter

    """ 3. GET DATA """
    # The try/except structure allows to quit the while loop by aborting the
    # script with <Ctrl-C>
    print('Press Ctrl-C in the console to break the while loop.')

    count = 0
    past_avgs = collections.deque(10*[0.0], 10)

    try:
        # The following loop acquires data, computes band powers, and calculates neurofeedback metrics based on those band powers
        while True:
            """ 3.1 ACQUIRE DATA """
            # Obtain EEG data from the LSL stream
            eeg_data, timestamp = inlet.pull_chunk(timeout=1, max_samples=buffer_size)

            # Only keep the channel we're interested in
            ch_data = np.array(eeg_data)[:, INDEX_CHANNEL]

            # Update EEG buffer with the new data
            eeg_buffer, filter_state = utils.update_buffer(eeg_buffer, ch_data, notch=True, filter_state=filter_state)

            """ 3.2 COMPUTE BAND POWERS """
            # Get newest samples from the buffer
            data_epoch = utils.get_last_data(eeg_buffer, buffer_size)

            samples = np.transpose(data_epoch)[0]
            avg = np.average(samples)
            std = np.std(samples)
            
            if std > 40:
                count += 1
                blink_handler(count, avg, std)

            prev_avg = avg

    except KeyboardInterrupt:
        print('Closing!')


if __name__ == "__main__":
    start_listen()