import torch
import torch.nn as nn
import numpy as np
import hydra
from omegaconf import OmegaConf
from datetime import datetime
import wandb

from models.model_v1 import Model
from dataset.dataset import Dataset
from sklearn.metrics import f1_score, recall_score, precision_score

DATA_PATH = '/data-slow/datasets/EpicKitchens/FULL_EPIC_KITCHENS/test/'
LABEL_PATH = '/data-slow/datasets/EpicKitchens/FULL_EPIC_KITCHENS/labels/EPIC_100_test_timestamps.csv'
DEVICE = torch.device('cpu')

def Test(model, dataloader, criterion):
    model.eval()

    # initialize metrics
    test_loss = 0
    fscore = []
    precision= []
    recall = []
    all_labels = []
    all_pred = []
    corrects = 0
    total_clips = 0
    
    for clips, labels in dataloader:

        all_labels = np.append(all_labels, labels.numpy())
        clips = clips.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():

            output = model(clips)
            
            test_loss += criterion(output, labels).item()
            
            label_pred = torch.max(output, 1)[1]
            all_pred = np.append(all_pred, label_pred.cpu().numpy())
            corrects += torch.sum(label_pred == labels)

            labels_cpu = labels.cpu().numpy()
            label_pred_cpu = label_pred.cpu().numpy()

            Fscore = f1_score(labels_cpu, label_pred_cpu, 
                              zero_division=0)
            fscore.append(Fscore)
            Recall = recall_score(labels_cpu, label_pred_cpu, 
                                  zero_division=0)
            recall.append(Recall)
            Precision = precision_score(labels_cpu, label_pred_cpu, 
                                        zero_division=0)
            precision.append(Precision)
            
            total_clips += len(output)
        pass

    accuracy = int(corrects)/int(total_clips)

    test_fscore = np.average(np.array(fscore))
    test_precision = np.average(np.array(precision))
    test_recall = np.average(np.array(recall))

    print(f'Corrects/Total: {corrects}/{total_clips}')
    print(f'Test loss: {test_loss}')
    print(f'Accuracy: {accuracy}')
    print(f'F1 score: {test_fscore}')
    print(f'Precision: {test_precision}')
    print(f'Recall: {test_recall}')
    pass

    return all_pred, all_labels

@hydra.main(version_base=None, config_path='configs', config_name='config')
def run_inference(cfg: OmegaConf):

    wandb.init(project=cfg.NAME)
    working_directory = wandb.run.dir
    model_path = cfg.inference.MODEL_PATH

    print(f'Testing model saved at path {model_path}')

    model = Model(cfg)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location ='cpu'))
    criterion = nn.CrossEntropyLoss()

    print("Loading the data...")
    batch_size = cfg.training.BATCH_SIZE
    data_threads = cfg.training.DATA_THREADS
    
    test_set = Dataset(cfg, frames_dir=DATA_PATH, annotations_file=LABEL_PATH)
    test_sampler = torch.utils.data.sampler.RandomSampler(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, sampler=test_sampler, batch_size=batch_size,
                                                num_workers=data_threads, drop_last=True, pin_memory=True)

    print('Start inference...')
    _ = Test(model, test_loader)


if __name__ == '__main__':
    run_inference()
