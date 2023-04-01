###############################################################################
# Code from
# https://github.com/pytorch/vision/blob/master/torchvision/datasets/folder.py
# Modified the original code so that it also loads images from the current
# directory as well as the subdirectories
###############################################################################

import torch.utils.data as data

from PIL import Image
import os
import os.path

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG', '.pgm', '.PGM',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP', '.tiff', 
    '.txt', '.json'
]


def is_image_file(filename):
    return any(filename.endswith(extension) for extension in IMG_EXTENSIONS)


def make_dataset(dir):
    images = []
    assert os.path.isdir(dir), f'{dir} is not a valid directory'

    for root, _, fnames in sorted(os.walk(dir)):
        for fname in fnames:
            if is_image_file(fname):
                path = os.path.join(root, fname)
                images.append(path)
    return images

def make_grouped_dataset(dir):
    images = []
    assert os.path.isdir(dir), f'{dir} is not a valid directory'
    fnames = sorted(os.walk(dir))
    for fname in sorted(fnames):
        root = fname[0]
        if paths := [
            os.path.join(root, f) for f in sorted(fname[2]) if is_image_file(f)
        ]:
            images.append(paths)
    return images

def check_path_valid(A_paths, B_paths):
    assert(len(A_paths) == len(B_paths))
    for a, b in zip(A_paths, B_paths):
        assert(len(a) == len(b))

def default_loader(path):
    return Image.open(path).convert('RGB')


class ImageFolder(data.Dataset):

    def __init__(self, root, transform=None, return_paths=False,
                 loader=default_loader):
        imgs = make_dataset(root)
        if len(imgs) == 0:
            raise RuntimeError(
                (
                    f"Found 0 images in: {root}" + "\n"
                    "Supported image extensions are: "
                )
                + ",".join(IMG_EXTENSIONS)
            )

        self.root = root
        self.imgs = imgs
        self.transform = transform
        self.return_paths = return_paths
        self.loader = loader

    def __getitem__(self, index):
        path = self.imgs[index]
        img = self.loader(path)
        if self.transform is not None:
            img = self.transform(img)
        return (img, path) if self.return_paths else img

    def __len__(self):
        return len(self.imgs)
