import torch
from tqdm import tqdm

def train_model(net, dataloaders_dict, optimizer, criterion, num_epochs, early_stopping=None, save_path=None):
    """
    function for training model

    Parameters
    ----------
    net: object
        model to train
    dataloaders_dict: dict
        dictionary contained dataloaders of train and validation
    criterion: object
        loss function used for training
    optimizer: object
        optimizer used for training
    num_epochs: int
        number of epochs for iteration

    Returns
    -------
    path_list: list
        list stored data path
    """
    # initialize GPU
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    print('using device:', device)
    net.to(device)
    torch.backends.cudnn.benchmark = True

    # lists for storing logging
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    best_val_loss = float('inf')
    no_improve_cnt = 0
    
    for epoch in range(num_epochs):
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('---------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
    
            epoch_loss = 0.0
            epoch_corrects = 0
    
            for inputs, labels in tqdm(dataloaders_dict[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
                
                if phase == 'train':
                    # forward
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)

                    # backward
                    loss.backward()
                    optimizer.step()
                    
                elif phase == 'val':
                    with torch.no_grad():
                        outputs = net(inputs)
                        loss = criterion(outputs, labels)
                        _, preds = torch.max(outputs, 1)

                else:
                    raise 
    
                epoch_loss += loss.item()
                epoch_corrects += torch.sum(preds == labels.data)
    
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.float() / len(dataloaders_dict[phase].dataset)
    
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

            if phase == 'train':
                train_losses.append(epoch_loss)
                train_accuracies.append(epoch_acc)
            else:
                val_losses.append(epoch_loss)
                val_accuracies.append(epoch_acc)

                if val_losses[-1] < best_val_loss:
                    vest_val_loss = val_losses[-1]
                    no_improve_cnt = 0
                    if save_path is not None:
                        state = {
                        'model_state_dict': net.state_dict(),
                        'optimizer_state_dict': optimizer.state_dict(),
                        'val_loss': val_losses[-1]
                        }
                        torch.save(state, save_path)
                else:
                    no_improve_cnt += 1
    
                if early_stopping and no_improve_cnt >= early_stopping:
                    print('Stopping early')
                    break

    plt.plot(train_losses, label='training loss')
    plt.plot(val_losses, label='validation loss')
    plt.legend()
    plt.show()

    plt.plot(train_accuracies, label='training accuracy')
    plt.plot(val_accuracies, label='validation accuracy')
    plt.legend()
    plt.show()

    return train_loss_hist, train_corrects_hist, val_loss_hist, val_corrects_hist