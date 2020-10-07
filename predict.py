import torch
from unet import Unet
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np

# 预处理，提高图片的输入分辨率后检测速度变慢。
x_transforms = transforms.Compose([
    transforms.Resize((1568, 1568)),         
    # transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.5,0.5,0.5], [0.5,0.5,0.5])
])

# 加载图片 
img = Image.open("./data/text/01.jpg")
# plt.imshow(img)
# plt.show()
# [N, C, H, W]
img = x_transforms(img)
 

# expand batch dimension
img = torch.unsqueeze(img, dim=0)


# 创建模型
model = Unet(3, 1)
# 加载模型权重
model_weight_path = "./original_weights_20.pth"
model.load_state_dict(torch.load(model_weight_path))
model.eval()



with torch.no_grad():
    y = model(img).sigmoid()
    img_y = torch.squeeze(y).numpy()
    plt.imshow(img_y,cmap='gray')
    plt.show()







