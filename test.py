import torch
import torch.nn as nn
import numpy as np
import hydra
import os
from omegaconf import OmegaConf
from datetime import datetime
import csv
# import wandb

from models.model_v1 import Model
from dataset.dataset import Dataset
from sklearn.metrics import balanced_accuracy_score, top_k_accuracy_score, accuracy_score

DATA_PATH = '/data-fast/127-data2/omartinez/FULL_EPIC_KITCHENS_RESIZED_256/val/'
LABEL_PATH = '/data-fast/127-data2/omartinez/FULL_EPIC_KITCHENS_RESIZED_256/labels/EPIC_100_validation.csv'
RESULTS_PATH = '/home-net/omartinez/TFG/logs/test_results/'

os.environ["HYDRA_FULL_ERROR"] = "1"

def Test(model, dataloader, criterion, file, device):
    model.eval()

    # initialize metrics
    test_loss = 0
    acc_lst = []
    top_k_acc_lst = []
    all_labels = []
    all_pred = []
    corrects = 0
    total_clips = 0
    
    for clips, labels in dataloader:

        # all_labels.append(labels.numpy())
        all_labels += labels.tolist()
        clips = clips.to(device)
        labels = labels.to(device)

        with torch.no_grad():

            output = model(clips)
            
            test_loss += criterion(output, labels).item()
       
            _, preds = torch.max(output, dim=1)
            corrects += torch.sum(preds == labels)

            all_pred += preds.cpu()
            all_labels += labels.data.cpu()

            acc = accuracy_score(labels.cpu().numpy(), preds.cpu().numpy())
            acc_lst.append(acc)

            top_k_acc = top_k_accuracy_score(labels.data.cpu(), output.cpu(), k=5, labels=[x for x in range(96)])
            top_k_acc_lst.append(top_k_acc)
            
            total_clips += len(output)
        pass

    test_acc = np.average(acc_lst)
    test_top_k_acc = np.average(top_k_acc_lst)
    balanced_acc = balanced_accuracy_score(all_labels, all_pred)

    print(f'Corrects/Total: {corrects}/{total_clips}')
    print(f'Test loss: {test_loss}')
    print(f'Accuracy: {test_acc}')
    print(f'Top 5 Acc: {test_top_k_acc}')
    print(f'Balanced Acc: {balanced_acc}')

    # write to results file
    file.write(f'Corrects/Total: {corrects}/{total_clips}\n')
    file.write(f'Test loss: {test_loss}\n')
    file.write(f'Accuracy: {test_acc}\n')
    file.write(f'Top 5 Acc: {test_top_k_acc}\n')
    file.write(f'Balanced Accuracy: {balanced_acc}\n')
    pass

    return all_pred, all_labels

@hydra.main(version_base=None, config_path='configs', config_name='config')
def run_inference(cfg: OmegaConf):

    # Define the device
    DEVICE = torch.device('cpu')
    if torch.cuda.is_available:
        DEVICE = torch.device('cuda')
        torch.cuda.set_device(cfg.training.GPU) 

    model_path = cfg.inference.WEIGHTS_PATH + cfg.inference.MODEL

    # Create file to save results
    f_path = RESULTS_PATH + 'results_' + cfg.inference.MODEL + '.txt'
    f = open(f_path, 'w')

    print(f'Testing model saved at path {model_path}')
    f.write(f'Testing model saved at path {model_path}\n')

    model = Model(cfg)
    model = model.to(DEVICE)
    model.load_state_dict(torch.load(model_path, map_location=DEVICE))
    criterion = nn.CrossEntropyLoss()

    print("Loading the data...")
    batch_size = cfg.training.BATCH_SIZE
    data_threads = cfg.training.DATA_THREADS
    
    test_set = Dataset(cfg, frames_dir=DATA_PATH, annotations_file=LABEL_PATH)
    test_sampler = torch.utils.data.sampler.RandomSampler(test_set)
    test_loader = torch.utils.data.DataLoader(test_set, sampler=test_sampler, batch_size=batch_size,
                                                num_workers=data_threads, drop_last=True, pin_memory=True)

    print('Start inference...')
    print(f'Datetime: {datetime.now()}')
    predicted, labels = Test(model, test_loader, criterion, f, DEVICE)

     # Create file to save labels and predicted labels
    i = 1
    f_labels = RESULTS_PATH + 'labels_' + cfg.inference.MODEL + '.csv'
    with open(f_labels, 'w') as csvfile: 
        csvwriter = csv.writer(csvfile) 
        rows = [predicted, labels] 
        csvwriter.writerows(rows)
    csvfile.close()

    print(f'Inference completed at {datetime.now()}')
    f.close()


if __name__ == '__main__':
    run_inference()
