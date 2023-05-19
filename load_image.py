import os
import shutil
from PIL import Image
import time
import cv2

ORGINAL_PATH = '/data-fast/127-data2/omartinez/FULL_EPIC_KITCHENS/'
# NEW_PATH = '/data-fast/107-data4/omartinez/FULL_EPIC_KITCHENS_RESIZED_112/'

for dir in ['train']:
    print(f'Copying frames from {dir}...')
    dir_path = os.path.join(ORGINAL_PATH, dir)
    for participant in os.listdir(dir_path):
        participant_path = os.path.join(dir_path, participant)
        for folder in os.listdir(participant_path):
            folder_path = os.path.join(participant_path, folder)
            for video in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video)
                try:
                    for i, frame in enumerate(os.listdir(video_path)):
                        
                        frame_path = os.path.join(video_path, frame)
                        start_time = time.time()
                        img = Image.open(frame_path)
                        # img = cv2.imread(frame_path, cv2.IMREAD_UNCHANGED)
                        open_time = time.time()
                        print(f'Elapsed time to open img {i}: {open_time - start_time}')
                        img = img.resize(size=(112, 112))
                        resize_time = time.time()
                        print(f'Elapsed time to resize img {i}: {resize_time - open_time}')
                        # img.save(new_frame_path)
                        if i == 10:
                            break
                except(Exception):
                    pass