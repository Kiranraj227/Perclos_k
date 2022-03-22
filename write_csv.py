import pandas as pd
import numpy
import random

#pd.read_csv("logdata.csv")


frame_counter = 0
frame_counter_list = []  # frame number
state = 0  # state of eyes (open - 0, close - 1)
blink_update = 0  # flag to track whether need to update blink
statelist = []  # eyes state list
state_string = []
state_string_list = []  #simply added this

ear = 0
blink_start = 0
blink_end = 1
blink_dur = 0
blink_counter = 0
eye_close_sum = 0
blink_rate = 0
blink_interval = 0
perclos_epoch = 0
blink_rate_epoch = 0

loops = 1800*5

blink_signal = [] #tracks blink events
perclos_list = [] #keeps all the perclos values
perclos_window = [] #size of 1800
blink_rate_window = []

video_data = pd.DataFrame()

print('program start')

for i in range(loops):
    frame_counter += 1
    frame_counter_list.append(frame_counter)

    ear = random.randint(10, 45)

    if ear < 20:
        if state == 0:
            blink_update = 1
            blink_start = frame_counter
        state = 1  # close
        #blink_dur += 1
    else:
        state = 0  # open
        if blink_update == 1:  # blink ends
            blink_counter += 1
            blink_interval = blink_start - blink_end
            blink_end = frame_counter
            blink_dur = blink_end - blink_start
            # keeps track of blink events
            blink_signal.append((blink_counter, blink_start, blink_end, blink_dur, blink_interval))
            blink_update = 0
        #blink_dur = 0
    statelist.append(state)

    # Blink rate (every frame)
    blink_rate_window.append(blink_counter)
    if len(blink_rate_window) > 1800:
        blink_rate_window.pop(0)

    blink_rate = blink_rate_window[-1] - blink_rate_window[0]




    # PERCLOS PART HERE
    perclos_window.append(state)
    if len(perclos_window) > 1800:
        popvalue = perclos_window.pop(0)
    else:
        popvalue = 0

    eye_close_sum = eye_close_sum + state - popvalue
    perclos = (eye_close_sum / 1800) * 100
    perclos_list.append(perclos)

    if frame_counter % 1800 == 0:
        perclos_epoch = perclos
        blink_rate_epoch = blink_rate

    frame_info_dict = {
        'Frames': [frame_counter],
        'EAR': [ear],
        'Eye_state': [state],
        'PERCLOS': [perclos],
        'PERCLOS_epoch': [perclos_epoch],
        'Blink_count': [blink_counter],
        'Blink_start': [blink_start],
        'Blink_end': [blink_end],
        'Blink_duration': [blink_dur],
        'Blink_interval': [blink_interval],
        'Blink_rate': [blink_rate],
        'Blink_rate_epoch': [blink_rate_epoch]
    }

    frame_info = pd.DataFrame(frame_info_dict)

    video_data = pd.concat([video_data, frame_info])

print('done {} loops'.format(loops))

# writing data frame to a CSV file
video_data.to_csv('PERCLOS_DATASET.csv', index=False)
