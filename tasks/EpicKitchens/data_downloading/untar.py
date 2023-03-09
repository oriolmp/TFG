import glob
import os
import tarfile

def untar_dirs(path):
    splits = glob.glob(path + '/*')

    for split_path in splits:
        kitchens = glob.glob(split_path + '/*')

        for kitchen_path in kitchens:
            kitchen_path = os.path.join(kitchen_path, 'rgb_frames')
            tar_files = glob.glob(kitchen_path + '/*.tar')

            for file in tar_files:
                # Extract the kitchen name
                name = file.split('/')[-1]
                name = name.split('.')[0]

                # Create a directory
                dir_path = '/'.join(file.split('/')[:-1])
                final_dir_path = os.path.join(dir_path, name)
                if not os.path.exists(final_dir_path):
                    os.makedirs(final_dir_path)

                # Untar the file in that directory
                tar = tarfile.open(file)
                tar.extractall(path=final_dir_path)  # untar file into same directory
                tar.close()

                # Remove the tar file
                os.remove(file)

if __name__ == '__main__':
    untar_dirs(path='/data-net/datasets/EpicKitchens/UDA_EPIC_KITCHENS100')