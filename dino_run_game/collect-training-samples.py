from pylsl import StreamInlet, resolve_byprop  # Module to receive EEG data

from statistics import mean, stdev

from timeit import default_timer as timer
from my_muselsl import record_multi


def start_record():
    """ CONNECT TO EEG STREAM """
    # Search for active LSL streams
    print('Looking for an EEG stream...')
    streams = resolve_byprop('type', 'EEG', timeout=2)
    if len(streams) == 0:
        raise RuntimeError('Can\'t find EEG stream.')

    try:
        record_multi(8, 5)
            
    except KeyboardInterrupt:
        print('Closing!')


if __name__ == "__main__":
    start_record()