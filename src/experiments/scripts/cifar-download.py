import torchvision.datasets as datasets
cifar_testset = datasets.CIFAR10(
    root='./data/cifar10',
    train=True,
    download=True,
    transform=None)
