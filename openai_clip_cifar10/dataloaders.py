import torchvision
import torch
import torchvision.transforms as transforms


def get_original_cifar10_dataloaders(root, train_bsz=256, val_bsz=256):
    # Add resize if needed
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(
            mean=(0.48145466, 0.4578275, 0.40821073),
            std=(0.26862954, 0.26130258, 0.27577711)
        ),
    ])

    trainset = torchvision.datasets.CIFAR10(root=root, train=True,
                                            download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(trainset, batch_size=train_bsz,
                                            shuffle=True, num_workers=2)

    testset = torchvision.datasets.CIFAR10(root=root, train=False,
                                        download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(testset, batch_size=val_bsz,
                                            shuffle=False, num_workers=2)

    return trainloader, testloader
