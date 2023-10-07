import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict
from ..ylib.dataloader.svhn_loader import SVHN

# from util.broden_loader import BrodenDataset, broden_collate, dataloader

# from .custom_loader import CustomLoader

imagesize = 32

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize([x/255.0 for x in [125.3, 123.0, 113.9]],
    #                     [x/255.0 for x in [63.0, 62.1, 66.7]]),
])

transform_train = transforms.Compose([
    # transforms.RandomCrop(imagesize, padding=4),
    transforms.RandomResizedCrop(size=imagesize, scale=(0.2, 1.)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465),
                         (0.2023, 0.1994, 0.2010)),
    # transforms.Normalize([x / 255.0 for x in [125.3, 123.0, 113.9]],
    #                      [x / 255.0 for x in [63.0, 62.1, 66.7]]),
])

transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    #transforms.RandomSizedCrop(224),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

kwargs = {'num_workers': 2, 'pin_memory': True}
num_classes_dict = {'CIFAR-100': 100, 'CIFAR-10': 10, 'imagenet': 1000, 'cat': 1}

def get_loader_in(args, config_type='default', split=('train', 'val')):
    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'batch_size': args.batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
        },
        "eval": {
            'transform_train': transform_test,
            'transform_test': transform_test,
            'batch_size': args.batch_size,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_test_largescale,
        },
    })[config_type]

    train_loader, val_loader, lr_schedule, num_classes = None, None, [50, 75, 90], 0
    if args.in_dataset == "CIFAR-10":
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR10(root='./datasets/data', train=True, download=True, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR10(root='./datasets/data', train=False, download=True, transform=transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
    elif args.in_dataset == "CIFAR-100":
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR100(root='./datasets/data', train=True, download=True, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR100(root='./datasets/data', train=False, download=True, transform=config.transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
    elif args.in_dataset == "imagenet":
        root = args.imagenet_root
        # Data loading code
        if 'train' in split:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale),
                batch_size=config.batch_size, shuffle=False, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale),
                batch_size=config.batch_size, shuffle=False, **kwargs)
    elif args.in_dataset == "ANIMALS":
        root = "/Users/apagnoux/Downloads/Test/"
        if 'train' in split:
            train_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root, config.transform_train),
                batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            val_loader = torch.utils.data.DataLoader(
                torchvision.datasets.ImageFolder(root, config.transform_test),
                batch_size=config.batch_size, shuffle=True, **kwargs)

    print("Data_loader's type is {}".format(train_loader))
    return EasyDict({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "lr_schedule": lr_schedule,
        "num_classes": num_classes_dict[args.in_dataset],
    })

# def get_loader_out(args, dataset=('tim', 'noise'),config_type='default'):
#
#     config = EasyDict({
#         "default": {
#             'transform_train': transform_train,
#             'transform_test': transform_test,
#             'transform_test_largescale': transform_test_largescale,
#             'transform_train_largescale': transform_train_largescale,
#             'batch_size': args.batch_size
#         },
#     })[config_type]
#     test_ood_loader, val_ood_loader = None, None
#
#     val_dataset = dataset[1]
#     batch_size = args.batch_size
#     if val_dataset == 'SVHN':
#         val_ood_loader = torch.utils.data.DataLoader(
#             SVHN('datasets/ood_data/svhn/', split='test', transform=transform_test, download=False),
#             batch_size=batch_size, shuffle=False,
#             num_workers=2)
#     elif val_dataset == 'CIFAR-10':
#         val_ood_loader = torch.utils.data.DataLoader(
#             torchvision.datasets.CIFAR10(root='./datasets/data', train=False, download=True, transform=transform_test),
#             batch_size=batch_size, shuffle=True, num_workers=2)
#
#     # elif val_dataset != 'SVHN':
#     #     val_ood_loader = torch.utils.data.DataLoader(
#     #         CustomLoader(root=val_dataset, transform=transform_test),
#     #         batch_size=batch_size, shuffle=True, num_workers=2)
#
#     else:
#         val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.ImageFolder("./datasets/ood_data/{}".format(val_dataset),
#                                                           transform=transform_test), batch_size=batch_size, shuffle=False, num_workers=2)
#
#     return EasyDict({
#         "val_ood_loader": val_ood_loader,
#     })
