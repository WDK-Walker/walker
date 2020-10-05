import torch
from unet import Unet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# 预处理
x_transforms = transforms.Compose([
    transforms.Resize((1568, 1568)),
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# load image
img = Image.open("./data/text/01.jpg")
# plt.imshow(img)
# plt.show()
# [N, C, H, W]
img = x_transforms(img)


# expand batch dimension
img = torch.unsqueeze(img, dim=0)


# create model
model = Unet(3, 1)
# load model weights
model_weight_path = "./original_weights_20.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()

# def uint82bin(n, count=8):
#     """returns the binary of integer n, count refers to amount of bits"""
#     return ''.join([str((n >> y) & 1) for y in range(count-1, -1, -1)])


with torch.no_grad():
    y = model(img).sigmoid()
    img_y = torch.squeeze(y).numpy()
    plt.imshow(img_y,cmap='gray')
    plt.show()







