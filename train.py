import time
import copy
import torch
import wandb

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs=25, print_batch=50):
    since = time.time()

    softmax = torch.nn.Softmax(dim=0)
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
                dataloader = dataloaders[0]  
            else:
                model.eval()  
                dataloader = dataloaders[1]   

            running_loss = 0.0
            running_corrects = 0
            total_videos = 0

            # Iterate over data.
            for i, (inputs, labels) in enumerate(dataloader):
                inputs = inputs.to(device)
                labels = labels.to(device)

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    # Get model outputs and calculate loss
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)

                    _, preds = torch.max(softmax(outputs), 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)
                total_videos += len(outputs)
                
                if i % print_batch == 0 and phase == 'train':
                    l = running_loss/total_videos
                    acc = running_corrects.cpu().numpy()/total_videos
                    print(' - Batch Number {} -> Loss: {:.3f} Accuracy: {:.3f}'.format(i, l, acc))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            if phase == 'train':
                wandb.log({'train/train_loss': epoch_loss,
                           'train/train_acc': epoch_acc,
                           'train/epoch': epoch / (num_epochs-1)})
            elif phase == 'val':
                wandb.log({'val/val_loss': epoch_loss,
                           'val/val_acc': epoch_acc,
                           'val/epoch': epoch / (num_epochs-1)})

            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model 