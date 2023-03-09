# This script is used to download the corresponding parts of the Epic_kitchens_55vs100 datasets
# Concretely, for now we download the RGB frames of P08, P01 and P22, which results in 6 cross domain tasks

# For simplicity we use only the 10 largest actions (since the domain of possible actions is then super large!!

import os

import git


def download_epic_kitchen_data(install_downloader=False):
    if install_downloader:
        git.Repo.clone_from('https://github.com/epic-kitchens/epic-kitchens-download-scripts.git', 'EP_downloaders')
    else:
        # Check if the repo is dirty
        my_repo = git.Repo('EP_downloaders')
        if my_repo.is_dirty(untracked_files=True):
            print('Changes detected.')

    # This downloads "only" the videeos and frames, but no the annotations
    command = 'cd EP_downloaders; python epic_downloader.py --output-path ./ --rgb-frames --val '
    # command = 'cd EP_downloaders; python epic_downloader.py --output-path ./ --videos --specific-videos P01_01,P08_01'

    os.system(command)


download_epic_kitchen_data(install_downloader=False)
