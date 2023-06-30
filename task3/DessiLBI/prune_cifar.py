import sys
sys.path.append("..")
from slbi_toolbox import SLBI_ToolBox
import torch
import numpy as np
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from vgg import VGG_A_BatchNorm
import random
from utils import *

# 设置超参数
batch_size = 128
epochs = 20
lr = 1e-3
kappa = 1
interval = 20
mu = 20
M = 10
N = 10
device = torch.device('cuda:0')
print(device)
print(torch.cuda.get_device_name(0))
data_root = 'data/CIFAR10/'
model_root = 'model/cifar_1/'


torch.backends.cudnn.benchmark = True
load_pth = torch.load(model_root+'train_vgg.pth')
torch.cuda.empty_cache()
model = VGG_A_BatchNorm().to(device)
model.load_state_dict(load_pth['model'])
name_list = []
layer_list = []
for name, p in model.named_parameters():
    name_list.append(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
])
test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
test_loader = DataLoader(dataset=test_dataset, batch_size=batch_size, shuffle=False)


optimizer = SLBI_ToolBox(model.parameters(), lr=lr, kappa=kappa, mu=mu, weight_decay=0)
optimizer.load_state_dict(load_pth['optimizer'])
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)

def prune_net(r, layer_name):
    print('prune the',layer_name)
    print('acc before pruning:', evaluate_batch(model, test_loader, device))
    optimizer.prune_layer_by_order_by_name(r, layer_name, True)
    print('acc after pruning', evaluate_batch(model, test_loader, device))
    optimizer.recover()
    print('acc after recovering', evaluate_batch(model, test_loader, device),'\n')

# 测试不同剪枝比例
prune_net(20, 'features.8.weight')
prune_net(40, 'features.8.weight')
prune_net(60, 'features.8.weight')
prune_net(80, 'features.8.weight')

# 可视化剪枝后
# weight = model.features[8].weight.clone().detach().cpu().numpy()
# before_weight = np.zeros((M*3,N*3))
# for i in range(M):
#     for j in range(N):
#         before_weight[i*3:i*3+3, j*3:j*3+3] = weight[i][j]
# before_weight = np.abs(before_weight)
# optimizer.prune_layer_by_order_by_name(80, 'features.8.weight', True)
# weight = model.features[8].weight.clone().detach().cpu().numpy()
# pruned_weight = np.zeros((M*3,N*3))
# for i in range(M):
#     for j in range(N):
#         pruned_weight[i*3:i*3+3, j*3:j*3+3] = weight[i][j]
# pruned_weight = np.abs(pruned_weight)
# plt.subplot(121)
# plt.imshow(before_weight, cmap='gray')
# plt.axis('off')
# plt.title('Conv3 before pruning')
# plt.subplot(122)
# plt.imshow(pruned_weight, cmap='gray')
# plt.axis('off')
# plt.title('Conv3 after pruning')
# plt.savefig('pics/vgg_prune_visu.pdf')
# plt.show()
# #
# # optimizer.recover()
# #
# # 测试剪枝不同层
#
# prune_net(80, 'features.0.weight')
# prune_net(80, 'features.4.weight')
# prune_net(80, 'features.8.weight')
# prune_net(80, 'features.15.weight')
# prune_net(80, 'features.22.weight')
# prune_net(80, 'classifier.0.weight')
# prune_net(80, 'classifier.3.weight')
# prune_net(80, 'classifier.6.weight')
#
