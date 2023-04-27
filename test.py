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
from sklearn.metrics import f1_score, recall_score, precision_score, top_k_accuracy_score, accuracy_score

DATA_PATH = '/data-slow/datasets/EpicKitchens/FULL_EPIC_KITCHENS/val/'
LABEL_PATH = '/data-slow/datasets/EpicKitchens/FULL_EPIC_KITCHENS/labels/EPIC_100_validation.csv'
DEVICE = torch.device('cpu')
RESULTS_PATH = '/home-net/omartinez/TFG/logs/test_results/'

os.environ["HYDRA_FULL_ERROR"] = "1"

def Test(model, dataloader, criterion, file):
    model.eval()

    # initialize metrics
    test_loss = 0
    acc_lst = []
    top_k_acc_lst = []
    f1_lst = []
    prec_lst = []
    recall_lst = []
    all_labels = []
    all_pred = []
    corrects = 0
    total_clips = 0
    
    for clips, labels in dataloader:

        # all_labels.append(labels.numpy())
        all_labels += labels.tolist()
        clips = clips.to(DEVICE)
        labels = labels.to(DEVICE)

        with torch.no_grad():

            output = model(clips)
            
            test_loss += criterion(output, labels).item()
       
            label_pred = torch.max(output, dim=1)[1]
            all_pred += label_pred.tolist()
            corrects += torch.sum(label_pred == labels)

            labels_cpu = labels.cpu().numpy()
            label_pred_cpu = label_pred.cpu().numpy()

            acc = accuracy_score(labels_cpu, label_pred_cpu)
            acc_lst.append(acc)

            top_k_acc = top_k_accuracy_score(labels_cpu, output, k=5, labels=[x for x in range(97)])
            top_k_acc_lst.append(top_k_acc)

            f1 = f1_score(labels_cpu, label_pred_cpu, 
                              zero_division=0, average='micro')
            f1_lst.append(f1)

            recall = recall_score(labels_cpu, label_pred_cpu, 
                                  zero_division=0, average='micro')
            recall_lst.append(recall)

            precision = precision_score(labels_cpu, label_pred_cpu, 
                                        zero_division=0, average='micro')
            prec_lst.append(precision)
            
            total_clips += len(output)
        # pass
        break

    test_acc = np.average(acc_lst)
    test_top_k_acc = np.average(top_k_acc_lst)
    test_f1 = np.average(np.array(f1_lst))
    test_precision = np.average(np.array(prec_lst))
    test_recall = np.average(np.array(recall_lst))

    print(f'Corrects/Total: {corrects}/{total_clips}')
    print(f'Test loss: {test_loss}')
    print(f'Accuracy: {test_acc}')
    print(f'Top 5 Acc: {test_top_k_acc}')
    print(f'F1 score: {test_f1}')
    print(f'Precision: {test_precision}')
    print(f'Recall: {test_recall}')

    # write to results file
    file.write(f'Corrects/Total: {corrects}/{total_clips}\n')
    file.write(f'Test loss: {test_loss}\n')
    file.write(f'Accuracy: {test_acc}\n')
    file.write(f'Top 5 Acc: {test_top_k_acc}\n')
    file.write(f'F1 score: {test_f1}\n')
    file.write(f'Precision: {test_precision}\n')
    file.write(f'Recall: {test_recall}\n')
    pass

    return all_pred, all_labels

@hydra.main(version_base=None, config_path='configs', config_name='config')
def run_inference(cfg: OmegaConf):

    model_path = cfg.inference.WEIGHTS_PATH + cfg.inference.MODEL

    # Create file to save results
    i = 1
    f_path = RESULTS_PATH + 'results_' + cfg.inference.MODEL + '.txt'
    while os.path.isdir(f_path):
        i += 1
        f_path = RESULTS_PATH + 'results_' + cfg.inference.MODEL + 'txt'
    f = open(f_path, 'w')

    print(f'Testing model saved at path {model_path}')
    f.write(f'Testing model saved at path {model_path}')

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
    print(f'Datetime: {datetime.now()}')
    predicted, labels = Test(model, test_loader, criterion, f)

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
