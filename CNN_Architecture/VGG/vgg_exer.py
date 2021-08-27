import torch
import torch.nn as nn
import torch.nn.functional as F

import utils
import VGG

# Set epoch and Device (Hyper param)
EPOCH = 10
DEVICE = 'cpu'

# Load CIFAR-10 data
train_loader, test_loader = utils.cifar10_upload()

# Check Data
for (X_train, y_train) in train_loader:
    print('X_train : ',X_train.size(), 'Type : ', X_train.type() )
    print('y_train : ',y_train.size(), 'Type : ', y_train.type() )
    break

# Load model from VGG
model = VGG.VGG_net(in_channels=3, num_classes=10).to(device = DEVICE)

# Set optimizer -> SGD and  Loss function
x = torch.randn(1, 3, 32, 32).to(device = DEVICE)
optimizer = torch.optim.SGD(model.parameters(), lr = 0.01, momentum = 0.5)
criterion = nn.CrossEntropyLoss()

# Get model info
print('Model : \n' ,model)


# Learn from CIFAR-10 train set 
for epoch in range(1, EPOCH+1):
    print('EPOCH : {} / {} '.format(epoch, EPOCH+1) )
    utils.train(model = model, train_loader = train_loader ,criterion = criterion, optimizer = optimizer)

# Evaluate from test set
test_loss, test_accuracy = utils.eval(model = model, test_loader= test_loader, criterion= criterion)
print('Test Loss : [{}] , Test Accuracy : [{}]'.format(test_loss, test_accuracy))
