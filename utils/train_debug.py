import time
import copy
import torch
import wandb
from sklearn.metrics import balanced_accuracy_score

def train_model(model, dataloaders, criterion, optimizer, device, num_epochs, print_batch, checkpoint_path, scheduler=None):
    since = time.time()

    start_time = time.time()
    softmax = torch.nn.Softmax(dim=0)
    scaler = torch.cuda.amp.GradScaler()
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    print(f'Initialize train functions: {time.time() - start_time}')

    for epoch in range(num_epochs):
        print(f'Epoch {epoch}/{num_epochs-1}')
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            start_time = time.time()
            if phase == 'train':
                model.train()
                dataloader = dataloaders[0]  
            else:
                model.eval()  
                dataloader = dataloaders[1]  
            print(f'Select dataloader: {time.time() - start_time}') 

            num_batches = len(dataloader)

            running_loss = 0.0
            total_clips = 0

            # running_corrects = 0
            # running_batch_loss = 0
            # running_batch_corrects = 0
            # total_batch_clips = 0
            
            step_pred = []
            step_labels = []
            epoch_pred = []
            epoch_labels = []
            

            # Iterate over data.
            start_time = time.time()
            for i, (inputs, labels) in enumerate(dataloader):
                print(f'Load batch clips: {time.time() - start_time}')
                print(f'len inputs: {len(inputs)}')
                
                
                start_time = time.time()
                inputs = inputs.to(device)
                labels = labels.to(device)
                print(f'Send clip to GPU: {time.time() - start_time}')

                # zero the parameter gradients
                optimizer.zero_grad()

                # forward
                # track history if only in train
                start_time = time.time()
                with torch.set_grad_enabled(phase == 'train'):
                    # Runs the forward pass with autocasting
                    with torch.autocast(device_type='cuda', dtype=torch.float16):
                        print(f'inputs is nan: {torch.isnan(inputs).any()}')
                        # Get model outputs and calculate loss
                        outputs = model(inputs)
                        print(f'outputs is nan: {torch.isnan(outputs).any()}')
                        print(f'Run: {time.time() - start_time}')
                        loss = criterion(softmax(outputs), labels) # By CorrsEntropy docs, it shoul be used with normalization. However, Pythorch tutorial doesn't
                        start_time = time.time()
                        print(f'Compute loss: {time.time() - start_time}')
    
                    _, preds = torch.max(softmax(outputs), 1)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        # Use gradient scaler
                        scaler.scale(loss).backward()
                        scaler.step(optimizer)
                        scaler.update()

                start_time = time.time()
                # statistics
                running_loss += loss.item() * inputs.size(0) # loss.item = loss divided by number of batch elements
                total_clips += len(outputs)

                # running_corrects += torch.sum(preds == labels.data)
                # running_batch_loss += loss.item()
                # running_batch_corrects += torch.sum(preds == labels.data)
                # total_batch_clips += len(outputs)

                step_pred += preds.cpu()
                step_labels += labels.data.cpu()
                epoch_pred += preds.cpu()
                epoch_labels += labels.data.cpu()

                print(f'Compute batch metrics: {time.time() - start_time}')
                
                if (i + 1) % print_batch == 0 or i == num_batches:
                    # batch_loss = running_batch_loss/total_batch_clips
                    # batch_acc = running_corrects.cpu().numpy()/total_batch_clips
                    # batch_acc = torch.sum(preds == labels.data) / inputs.size(0)
                    step_acc = balanced_accuracy_score(step_labels, step_pred)

                    if phase == 'train':
                        wandb.log({
                            'train_batches/train_loss': loss.item(),
                            'train_batches/train_acc': step_acc,
                            'train_batches/batch': i + 1
                        })
                        # out some control prints
                        print(' - Batch Number {} -> Loss: {:.3f} Accuracy: {:.3f}'.format(i+1, loss.item(), step_acc))
                    elif phase == 'val':
                        wandb.log({
                            'val_batches/val_loss': loss.item(),
                            'val_batches/val_acc': step_acc,
                            'val_batches/batch': i + 1
                        })

                    # running_batch_loss = 0
                    # running_batch_corrects = 0
                    step_pred = []
                    step_labels = []

                if i == 200:
                    break

            if phase == 'train' and scheduler is not None:
                scheduler.step()

            epoch_loss = running_loss / total_clips
            # epoch_acc = running_corrects.double() / total_clips
            epoch_acc = balanced_accuracy_score(epoch_labels, epoch_pred)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))

            # if phase == 'train':
            #     wandb.log({'train_epochs/train_loss': epoch_loss,
            #                'train_epochs/train_acc': epoch_acc,
            #                'train_epochs/epoch': epoch})
            # elif phase == 'val':
            #     wandb.log({'val_epochs/val_loss': epoch_loss,
            #                'val_epochs/val_acc': epoch_acc,
            #                'val_epochs/epoch': epoch})

            # deep copy the model
            # if phase == 'val' and epoch_acc > best_acc:
            #     best_acc = epoch_acc
            #     best_model_wts = copy.deepcopy(model.state_dict())
            #     checkpoint_path = checkpoint_path + f'_{epoch}'
            #     torch.save(model.state_dict(), checkpoint_path)
            break
            

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    # load best model weights
    model.load_state_dict(best_model_wts)
    return model 