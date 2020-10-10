import torch
import argparse
from torch.utils.data import DataLoader
from torch import nn, optim
from torchvision.transforms import transforms
from unet import Unet
from dataset import LiverDataset
import matplotlib.pyplot as plt
import numpy as np


# 是否使用cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#改动前
# x_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor(),
#     transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
# ])
#改动后
x_transforms = transforms.Compose([
    transforms.Resize((598, 448)),
    transforms.ToTensor(),
    transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
])

# mask只需要转换为tensor
#改动前
# y_transforms = transforms.Compose([
#     transforms.Resize((224, 224)),
#     transforms.ToTensor()
# ])
# 改动后
y_transforms = transforms.Compose([
    transforms.Resize((598,448)),
    transforms.CenterCrop(448),
    transforms.ToTensor()
])
def train_model(model, optimizer, dataload, num_epochs=20):
    n = []
    for epoch in range(1, num_epochs+1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)
        dt_size = len(dataload.dataset)
        epoch_loss = 0
        step = 0
        for x, y in dataload:
            step += 1
            inputs = x.to(device)
            labels = y.to(device)
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward
            outputs = model(inputs)
            criterion = nn.BCEWithLogitsLoss()
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            print("%d/%d,train_loss:%0.3f" % (step, (dt_size - 1) // dataload.batch_size + 1, loss.item()))
        print("epoch %d loss:%0.3f" % (epoch, epoch_loss/step))
        n.append(epoch_loss/step)
        n_3f = np.round(n, 3)
        print(n_3f)
    m = list(range(1, 21))
    torch.save(model.state_dict(), 'original_weights_%d.pth' % epoch)
    plt.plot(m, n_3f)
    plt.title('train_loss vs epoch')
    plt.xlabel('epoch')
    plt.ylabel('loss')
    plt.show()
    plt.savefig('train_loss')
    return model


#训练模型
def train(args):
    model = Unet(3,1).to(device)
    batch_size = args.batch_size
    # criterion = nn.BCEWithLogitsLoss()

    optimizer = optim.Adam(model.parameters())
    liver_dataset = LiverDataset("data/train", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
    train_model(model, optimizer, dataloaders)

#显示模型的输出结果
def test(args):
    model = Unet(3,1)
    model.load_state_dict(torch.load(args.ckpt, map_location='cpu'))
    liver_dataset = LiverDataset("data/val", transform=x_transforms, target_transform=y_transforms)
    dataloaders = DataLoader(liver_dataset, batch_size=1)
    model.eval()
    plt.ion()
    with torch.no_grad():
        for x, y in dataloaders:
            y=model(x).sigmoid()
            img_y=torch.squeeze(y).numpy()
            plt.imshow(img_y, cmap='gray', interpolation='nearest')
            plt.pause(0.5)
        plt.ioff()
        plt.show()


if __name__ == '__main__':
    #参数解析
    parse = argparse.ArgumentParser()
    parse = argparse.ArgumentParser()
    parse.add_argument("--action", type=str, help="train or test")
    parse.add_argument("--batch_size", type=int, default=5)
    parse.add_argument("--ckpt", type=str, help="the path of model weight file")
    args = parse.parse_args()

    if args.action=="train":
        train(args)
    elif args.action=="test":
        test(args)

