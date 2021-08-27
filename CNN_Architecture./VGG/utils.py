import torch
from torchvision import datasets, transforms

def mnist_upload(batch_size = 32):
    train_dataset = datasets.MNIST(root = 'data/MNIST',
                                train = True,
                                download = False,
                                transform = transforms.ToTensor())
    test_dataset = datasets.MNIST(root = 'data/MNIST',
    train = True,
    transform = transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True)

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False)

    return train_loader ,test_loader

def cifar10_upload(batch_size = 32):
    train_dataset = datasets.CIFAR10(root = 'data/CIFAR_10',
                                train = True,
                                download = False,
                                transform = transforms.ToTensor())
    test_dataset = datasets.CIFAR10(root = 'data/CIFAR_10',
    train = True,
    transform = transforms.ToTensor())

    train_loader = torch.utils.data.DataLoader(dataset = train_dataset,
    batch_size = batch_size,
    shuffle = True)

    test_loader = torch.utils.data.DataLoader(dataset = test_dataset,
    batch_size = batch_size,
    shuffle = False)

    return train_loader ,test_loader

def train(model, train_loader ,criterion, optimizer, device = 'cpu'):

    model.train()

    for batch_idx, (data, target) in enumerate(train_loader):
        data = data.to(device  = device)
        target = target.to(device = device)

        # forward
        scores = model(data)
        loss = criterion(scores, target)

        # backward
        optimizer.zero_grad()
        loss.backward()

        #gradient descent or adam step
        optimizer.step()        
        
        if batch_idx % 200 == 0:
                print('[{} / {} ({:.0f}%)]\tTrain Loss: {:.6f}'.format(
                    batch_idx * len(data), len(train_loader.dataset), 100.*batch_idx / len(train_loader), loss.item()
                ))

def eval(model, test_loader, criterion, device = 'cpu'):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for image, label in test_loader:
            image = image.to(device)
            label = label.to(device)
            output = model(image)
            test_loss += criterion(output, label).item()
            prediction = output.max(1, keepdim = True)
            correct += prediction.eq(label.view_as(prediction)).sum().item()
    test_loss /= len(test_loader)
    test_accuracy = 100. * correct / len(test_loader.dataset)

    return test_loss , test_accuracy
