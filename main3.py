import torch
import torch.optim as optim
import torch.nn as nn
import torchvision.models as models
from torchvision import datasets, transforms
import copy
import os

# Load Pretrained ResNet-50
resnet50 = models.resnet50(pretrained=True)

# Modify the final layer
num_ftrs = resnet50.fc.in_features
resnet50.fc = nn.Linear(num_ftrs, 2)  # 2 classes: folded and not folded

# Define the loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(resnet50.parameters(), lr=0.001, momentum=0.9)

# Data preparation (assuming dataloaders are already created)
# Example transforms, adjust based on your dataset
data_transforms = {
    'train': transforms.Compose([
        transforms.RandomResizedCrop(224),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
    'val': transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ]),
}

# Assuming you have datasets in 'data_dir'
data_dir = 'path_to_your_data'
image_datasets = {x: datasets.ImageFolder(os.path.join(data_dir, x),
                                          data_transforms[x])
                  for x in ['train', 'val']}
dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=4,
                                              shuffle=True, num_workers=4)
               for x in ['train', 'val']}

# Training function


def train_model(model, dataloaders, criterion, optimizer, num_epochs=25):
    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                optimizer.zero_grad()
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double(
            ) / len(dataloaders[phase].dataset)

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        print(f'Epoch {epoch}/{num_epochs - 1}')
        print(f'{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}')

    model.load_state_dict(best_model_wts)
    return model


# Example usage:
model_ft = train_model(resnet50, dataloaders, criterion,
                       optimizer, num_epochs=25)

# Reward function example


def reward_function(preds, labels):
    rewards = (preds == labels).float()
    return rewards.mean().item()


for inputs, labels in dataloaders['train']:
    optimizer.zero_grad()
    with torch.set_grad_enabled(True):
        outputs = resnet50(inputs)
        _, preds = torch.max(outputs, 1)
        loss = criterion(outputs, labels)
        reward = reward_function(preds, labels)
        adjusted_loss = loss * (1 - reward)
        adjusted_loss.backward()
        optimizer.step()

