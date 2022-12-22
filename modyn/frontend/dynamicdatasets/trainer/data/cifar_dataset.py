import torchvision.transforms as transforms
from torchvision import datasets

def get_cifar_datasets():

    transform = transforms.Compose([transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    train_dataset =  datasets.CIFAR10(root='./data', train=True,
                        download=True, transform=transform)

    val_dataset = datasets.CIFAR10(root='./data', train=False,
                        download=True, transform=transform)

    return train_dataset, val_dataset