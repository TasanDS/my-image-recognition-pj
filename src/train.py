import torch
from tqdm import tqdm

def train_model(net, dataloaders_dict, criterion, optimizer, num_epochs):
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
    for epoch in range(num_epochs):

        device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        print('using device:', device)

        net.to(device)

        torch.backends.cudnn.benchmark = True
        
        print(f'Epoch {epoch+1}/{num_epochs}')
        print('---------------')

        for phase in ['train', 'val']:
            if phase == 'train':
                net.train()
            else:
                net.eval()
    
            epoch_loss = 0.0
            epoch_corrects = 0
    
            if (epoch == 0) and (phase == 'train'):
                continue
    
            for inputs, labels in tqdm(dataloaders_dict[phase]):

                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()
    
                with torch.set_grad_enabled(phase=='train'):
                    outputs = net(inputs)
                    loss = criterion(outputs, labels)
                    _, preds = torch.max(outputs, 1)
    
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()
    
                epoch_loss += loss.item() * inputs.size(0)
    
                epoch_corrects += torch.sum(preds == labels.data)
    
            epoch_loss = epoch_loss / len(dataloaders_dict[phase].dataset)
            epoch_acc = epoch_corrects.double(
            ) / len(dataloaders_dict[phase].dataset)
    
            print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')