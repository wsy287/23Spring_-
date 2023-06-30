import os
import torch
import torch.nn as nn
import sys
sys.path.append("..")
import numpy as np
from utils import *
from vgg import VGG_A_BatchNorm
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import argparse
from slbi_toolbox import SLBI_ToolBox
import torch.nn.functional as F
import matplotlib.pyplot as plt
import ssl

ssl._create_default_https_context = ssl._create_unverified_context# 全局取消证书验证
torch.backends.cudnn.benchmark = True
def get_slbi(model, lr, kappa=1, mu=20):
    layer_list = []
    name_list = []
    for name, p in model.named_parameters():
        name_list.append(name)
        if len(p.data.size()) == 4 or len(p.data.size()) == 2:
            layer_list.append(name)
    # 定义SLBI优化器
    optimizer = SLBI_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, weight_decay=0)
    optimizer.assign_name(name_list)
    optimizer.initialize_slbi(layer_list)
    return optimizer
parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", default=128, type=int)
parser.add_argument("--lr", default=1e-1, type=float)
parser.add_argument("--interval", default=20, type=int)
parser.add_argument("--kappa", default=1, type=int)
parser.add_argument("--dataset", default='MNIST', type=str)
parser.add_argument("--train", default=True, type=str2bool)
parser.add_argument("--download", default=True, type=str2bool)
parser.add_argument("--shuffle", default=True, type=str2bool)
parser.add_argument("--use_cuda", default=True, type=str2bool)
parser.add_argument("--parallel", default=False, type=str2bool)
parser.add_argument("--epoch", default=60, type=int)
parser.add_argument("--model_name", default='lenet', type=str)
parser.add_argument("--gpu_num", default='0', type=str)
parser.add_argument("--mu", default=20, type=int)
parser.add_argument("--save_path", default='model/', type=str)
args = parser.parse_args()
# define the model
name_list = []
device = torch.device("cuda" if (args.use_cuda and torch.cuda.is_available()) else "cpu")
torch.cuda.empty_cache()
model = VGG_A_BatchNorm().to(device)
if args.parallel:
    model = nn.DataParallel(model)
else:
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu_num
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    print(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)
optimizer = get_slbi(model, lr=args.lr)
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
# load data
data_root = 'data/CIFAR10/'
train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=args.batch_size, shuffle=False)


# train the model
train_accuracy_curve = [np.nan] * args.epoch
test_accuracy_curve = [np.nan] * args.epoch
loss_list = [np.nan] * args.epoch
for ep in tqdm(range(args.epoch)):
    model.train()
    descent_lr(args.lr, ep, optimizer, args.interval)
    loss_val = 0
    correct = num = 0
    for iter, pack in enumerate(train_loader):
        data, target = pack[0].to(device), pack[1].to(device)
        logits = model(data)
        loss = F.cross_entropy(logits, target)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        _, pred = logits.max(1)
        loss_val += loss.item()
        correct += pred.eq(target).sum().item()
        num += data.shape[0]
        if (iter + 1) % (len(train_loader) // 5) == 0:
            print('epoch[%d/%d]: iter=%d, loss=%f, Train_ACC=%f' % (
                ep + 1, args.epoch, iter + 1, loss_val / 100, correct / num))
    loss_val /= num
    train_ac = correct / num
    test_ac = evaluate_batch(model, test_loader, device)
    train_accuracy_curve[ep] = train_ac
    test_accuracy_curve[ep] = test_ac
    loss_list[ep] = loss_val
    optimizer.update_prune_order(ep)

if not os.path.exists(args.save_path):
    os.makedirs(args.save_path)
np.save(args.save_path + 'train_acc.npy',train_accuracy_curve)
np.save(args.save_path + 'test_acc.npy',test_accuracy_curve)
np.save(args.save_path + 'loss.npy',loss_list)
save_model_and_optimizer(model, optimizer, args.save_path+'train_vgg.pth')

# plot the result
x = range(len(train_accuracy_curve))
plt.plot(x, train_accuracy_curve, label = 'train accuracy')
plt.plot(x, test_accuracy_curve, label = 'test accuracy')
plt.xlabel('epoch')
plt.ylabel('accuracy')
plt.legend(loc='best')
plt.title('DessiLBI with adam on CIFAR10')
plt.savefig('pics/CIFAR10_DessiLBI.pdf')
plt.show()


