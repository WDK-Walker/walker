from torch.utils.data import Dataset
import PIL.Image as Image
import os


def make_datast(root):
    imgs = []
    n = len(os.listdir(root))//2
    for i in range(n):
        img = os.path.join(root, '%dimg.png'%i)
        label = os.path.join(root, '%dlabel.png'%i)
        imgs.append((img, label))
    return imgs


class LiverDataset(Dataset):
    def __init__(self, root, transform=None, target_transform=None):
        imgs = make_datast(root)
        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        x_path, y_path = self.imgs[index]
        img_x = Image.open(x_path)
        img_y = Image.open(y_path)
        if self.transform is not None:
            img_x = self.target_transform(img_x)
            img_y = self.target_transform(img_y)
        return img_x, img_y
    def __len__(self):
        return len(self.imgs)