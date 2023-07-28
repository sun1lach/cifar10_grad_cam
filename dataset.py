import ssl

import albumentations as A
import torch
import torchvision
from albumentations.pytorch import ToTensorV2

ssl._create_default_https_context = ssl._create_unverified_context


class Cifar10Dataset(torchvision.datasets.CIFAR10):
    def __init__(
        self, root="~/data/cifar10", train=True, download=True, transform=None
    ):
        super().__init__(root=root, train=train, download=download, transform=transform)

    def __getitem__(self, index):
        image, label = self.data[index], self.targets[index]

        if self.transform is not None:
            transformed = self.transform(image=image)
            image = transformed["image"]

        return image, label


class CustomDataLoader:
    @staticmethod
    def get_train_test_transforms():
        train_transforms = A.Compose(
            [
                A.RandomCrop(32, 32, p=4),
                A.CoarseDropout(
                    max_holes=1,
                    max_height=16,
                    max_width=16,
                    min_holes=1,
                    min_height=16,
                    min_width=16,
                    fill_value=(0.4914, 0.4822, 0.4465),
                    always_apply=False,
                    p=0.5,
                ),
                A.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
                ),
                ToTensorV2(),
            ]
        )

        # Test Phase transformations
        test_transforms = A.Compose(
            [
                A.Normalize(
                    mean=(0.4914, 0.4822, 0.4465), std=(0.2023, 0.1994, 0.2010)
                ),
                ToTensorV2(),
            ]
        )

        return train_transforms, test_transforms

    def get_train_test_dataset(self):
        train_transforms, test_transforms = CustomDataLoader.get_train_test_transforms()
        train = Cifar10Dataset(
            root="./data", train=True, download=True, transform=train_transforms
        )
        test = Cifar10Dataset(
            root="./data", train=False, download=True, transform=test_transforms
        )

        return train, test
