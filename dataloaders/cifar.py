from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets

NORMALIZATION_TRANSFORMS = transforms.Normalize(
                mean=[x / 255.0 for x in [125.3, 123.0, 113.9]],
                std=[x / 255.0 for x in [63.0, 62.1, 66.7]]
            )

DEFAULT_TRANSFORMS = transforms.Compose([
                transforms.ToTensor(),
                NORMALIZATION_TRANSFORMS,
            ])

TRAIN_TRANSFORMS = transforms.Compose([
                transforms.RandomHorizontalFlip(),
                transforms.RandomCrop(32, padding=4, padding_mode='reflect'),
                DEFAULT_TRANSFORMS,
])


def create_dataloaders(dataset, augment, batch_size):
    transform_train = TRAIN_TRANSFORMS if augment else DEFAULT_TRANSFORMS

    kwargs = {'num_workers': 1, 'pin_memory': True}
    assert(dataset == 'cifar10' or dataset == 'cifar100')

    train_dataset = datasets.__dict__[dataset.upper()]('./data', train=True, download=True, transform=transform_train)

    val_dataset = datasets.__dict__[dataset.upper()]('./data', train=False, transform=DEFAULT_TRANSFORMS)

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False, **kwargs)

    return train_loader, val_loader
