from torch.utils import data
from torchvision import transforms as T
from torchvision.datasets import ImageFolder


def get_loader(image_dir, crop, crop_size=178, image_size=128, batch_size=16, mode='train', num_workers=1):
    """
    Build and return a data loader.
    """
    transform = []
    if crop:
        transform.append(T.CenterCrop(crop_size))
    transform.append(T.Resize((image_size,image_size)))
    if mode == 'train':
        transform.append(T.RandomHorizontalFlip())
    transform.append(T.ToTensor())
    transform.append(T.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5)))
    transform = T.Compose(transform)

    dataset = ImageFolder(image_dir, transform)

    data_loader = data.DataLoader(dataset=dataset,
                                  batch_size=batch_size,
                                  drop_last=(mode == 'train'),
                                  shuffle=(mode == 'train'),
                                  num_workers=num_workers)
    return data_loader