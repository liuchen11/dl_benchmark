# Obtain preprocessed data

import torch
from torchvision import datasets, transforms

def cifar10(batch_size, batch_size_test, horizontal_flip = True, random_clip = True, normalization = None):

    if normalization == None:
        normalization = transforms.Normalize(mean = [0.485, 0.456, 0.406], std = [0.229, 0.224, 0.225])

    basic_transform = [transforms.ToTensor(), normalization]
    data_augumentation = []
    if horizontal_flip == True:
        data_augumentation.append(transforms.RandomHorizontalFlip())
    if random_clip == True:
        data_augumentation.append(transforms.RandomCrop(32, 4))

    train_set = datasets.CIFAR10(root = './data', train = True, download = True,
        transform = transforms.Compose(data_augumentation + basic_transform))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    test_set = datasets.CIFAR10(root = './data', train = False, download = True,
        transform = transforms.Compose(basic_transform))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size_test, shuffle = False, num_workers = 4, pin_memory = True)

    return train_loader, test_loader

def mnist(batch_size, batch_size_test, horizontal_flip = False, random_clip = False, normalization = None):

    if normalization == None:
        normalization = transforms.Normalize(mean = [0.5, 0.5, 0.5], std = [0.5, 0.5, 0.5])

    basic_transform = [transforms.ToTensor(), normalization]
    data_augumentation = []
    if horizontal_flip == True:
        data_augumentation.append(transforms.RandomHorizontalFlip())
    if random_clip == True:
        data_augumentation.append(transforms.RandomCrop(28, 4))

    train_set = datasets.MNIST(root = './data', train = True, download = True,
        transform = transforms.Compose(data_augumentation + basic_transform))
    train_loader = torch.utils.data.DataLoader(train_set, batch_size = batch_size, shuffle = True, num_workers = 4, pin_memory = True)

    test_set = datasets.MNIST(root = './data', train = False, download = True,
        transform = transforms.Compose(basic_transform))
    test_loader = torch.utils.data.DataLoader(test_set, batch_size = batch_size_test, shuffle = False, num_workers = 4, pin_memory = True)

    return train_loader, test_loader
