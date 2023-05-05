import os
import shutil
from PIL import Image

ORGINAL_PATH = '/data-fast/127-data2/omartinez/FULL_EPIC_KITCHENS/'
NEW_PATH = '/data-fast/127-data2/omartinez/FULL_EPIC_KITCHENS_RESIZED_256/'

if not os.path.exists(NEW_PATH):
    os.mkdir(NEW_PATH)
    print(f'Created folder {NEW_PATH}')

original_labels_path = '/data-fast/127-data2/omartinez/FULL_EPIC_KITCHENS/labels/'
new_labels_path = '/data-fast/127-data2/omartinez/FULL_EPIC_KITCHENS_RESIZED_256/labels/'
if not os.path.exists(new_labels_path):
    os.mkdir(new_labels_path)
    print(f'Created folder {new_labels_path}')

for file in os.listdir(original_labels_path):
    source = os.path.join(original_labels_path, file)
    destination = os.path.join(new_labels_path, file)
    shutil.copy(source, destination)

for dir in ['test', 'train', 'val']:
    print(f'Copying frames from {dir}...')
    dir_path = os.path.join(ORGINAL_PATH, dir)
    new_dir_path = os.path.join(NEW_PATH, dir)
    if not os.path.exists(new_dir_path):
        os.mkdir(new_dir_path)
        # print(f'Created folder {new_dir_path}')
    for participant in os.listdir(dir_path):
        participant_path = os.path.join(dir_path, participant)
        new_participant_path = os.path.join(new_dir_path, participant)
        if not os.path.exists(new_participant_path):
            os.mkdir(new_participant_path)
            # print(f'Created folder {new_participant_path}')
        for folder in os.listdir(participant_path):
            folder_path = os.path.join(participant_path, folder)
            new_folder_path = os.path.join(new_participant_path, folder)
            if not os.path.exists(new_folder_path):
                os.mkdir(new_folder_path)
                # print(f'Created folder {new_folder_path}')
            for video in os.listdir(folder_path):
                video_path = os.path.join(folder_path, video)
                new_video_path = os.path.join(new_folder_path, video)
                if not os.path.exists(new_video_path):
                    os.mkdir(new_video_path)
                    # print(f'Created folder {new_video_path}')
                for frame in os.listdir(video_path):
                    frame_path = os.path.join(video_path, frame)
                    new_frame_path = os.path.join(new_video_path, frame)
                    img = Image.open(frame_path)
                    img = img.resize(size=(256, 256))
                    img.save(new_frame_path)
        


