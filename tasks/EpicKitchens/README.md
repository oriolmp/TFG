# Action Recognition on EpicKitchens100

## Data downloading
First of all, download the scripts from [here](https://github.com/epic-kitchens/epic-kitchens-100-annotations#unsupervised-domain-adaptation-challenge), and run
```
python epic_downloader.py --rgb-frames
```
Alternatively, I provide a [script](data_downloading/download_data.py) that automatizes the repo cloning and runs a specified command. Just make sure to change it for the appropriate one.

All in all, this yields the .tar compressed files. Then run the [untaring file](data_downloading/untar.py) to decompress all the dataset. Make sure to change the dataset path.

