# TFG

Welcome to the repository of the TFG "Efficient transformers for video classification"!

In this repo you will find all code related to the custom transformer described at our work.
The model is a customizable transformer encoder, which can use 4 different attentions: vanilla, linformer, nyströmformer and cosformer.


## Installation
All code is implemented with PyTorch.
Please, first install all requierements.

``
pip install -r requirements.txt
``

## Code organitzation
The code is organized as follows:
- **attention_zoo:** Folder containig all code related to the different attention mechanisms
- **configs:** contains all Hydra configuration files
- **custom_sets:** csv containing the information related to the used dataset, which is dervied from EpicKitchens original dataset
- **dataset:** contains custom dataset class to load EpicKitchens 
- **logs:** folder to save logs from training and test results
- **models:** contain the model.py where our model is defined
- **notebooks:** some useful notebooks for debugging. It also includes a specific notebook for inference
- **samples:** some video obtained from the dataset that serve as example
- **tasks:** some useful scripts to manage the dataset and debugging

**WARNING**: Attention_zoo is not publicly available. The code is associated with an unpublished paper and must remain confidential until publication.

There are also 3 main files:
- **main.py**: Runs an experiment to train a specific model
- **train.py**: Script with the training function
- **test.py**: Inference a specified model.


## Configuration
All code options are selected using Hydra config files.
Specifically, we have the following cfg categories:
- **dataset:** determines the input video properties and the dataset used. The default cfg file is dataset1.yaml with
  - NAME: 'epic_kitchens'
  - FRAME_SIZE: 112 -> we consider square imgs
  - NUM_FRAMES: 100
  - IN_CHANNELS: 3   
- **inference:** select the model for inference. The default cfg file is inference1.yaml with:
  - WEIGHTS_PATH: 'your/weights/path'
  - MODEL: 'model_name'   
- **model:** select model parameters and which attention to use. The default cfg file is model_v1.yaml with:
  - ATTENTION: 'vanilla_attention'
  - NUM_CLASSES: 4
  - PATCH_SIZE: 16
  - DEPTH: 2 -> attention blocks
  - HEADS: 4 -> attention heads
- **training:** select training parameters. The default cfg file is train1.yaml with:
  - EPOCHS: 1
  - SEED: 0
  - BATCH_SIZE: 4
  - DATA_THREADS: 5 -> dataloader number of workers
  - PRINT_BATCH: 50
  - LEARNING_RATE: 1e-5 
  - PRETRAINED_STATE_PATH: None 
  - GPU: 0
  - SCHEDULER: False -> whether use lr scheduler or not

For example, if we would like to run a trainig with 'linformer' (which is described at model_v3.yaml) with patch size 32 we should run the follwoing command:
``
python main.py model=model_v3 model.PATCH_SIZE=32
``


