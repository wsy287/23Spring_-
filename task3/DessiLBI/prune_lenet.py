import os
from slbi_toolbox import SLBI_ToolBox
import torch
from utils import *
import matplotlib.pyplot as plt
import numpy as np
from data_loader import load_data
import lenet
torch.backends.cudnn.benchmark = True
current_path = os.getcwd()
load_pth = torch.load(current_path+'/model/mnist_1/lenet.pth')
torch.cuda.empty_cache()
model = lenet.Net().cuda()
model.load_state_dict(load_pth['model'])
name_list = []
layer_list = []
M = 10
N = 10
device = torch.device('cuda:0')
for name, p in model.named_parameters():
    name_list.append(name)
    if len(p.data.size()) == 4 or len(p.data.size()) == 2:
        layer_list.append(name)
optimizer = SLBI_ToolBox(model.parameters(), lr=1e-1, kappa=1, mu=20, weight_decay=0)
optimizer.load_state_dict(load_pth['optimizer'])
optimizer.assign_name(name_list)
optimizer.initialize_slbi(layer_list)
test_loader = load_data(dataset='MNIST', train=False, download=True, batch_size=64, shuffle=False)
#### test prune one layer
weight_conv3 = model.conv3.weight.clone().detach().cpu().numpy()
before_weight = np.zeros((M*5,N*5))
for i in range(M):
    for j in range(N):
        before_weight[i*5:i*5+5, j*5:j*5+5] = weight_conv3[i][j]
before_weight = np.abs(before_weight)
print('prune conv3')
print('acc before pruning')
evaluate_batch(model, test_loader, 'cuda')
print('acc after pruning')
optimizer.prune_layer_by_order_by_name(80, 'conv3.weight', True)
evaluate_batch(model, test_loader, 'cuda')
weight_conv3 = model.conv3.weight.clone().detach().cpu().numpy()
pruned_weight = np.zeros((M*5,N*5))
for i in range(M):
    for j in range(N):
        pruned_weight[i*5:i*5+5, j*5:j*5+5] = weight_conv3[i][j]
pruned_weight = np.abs(pruned_weight)
print('acc after recovering')
optimizer.recover()
evaluate_batch(model, test_loader, 'cuda')


plt.subplot(121)
plt.imshow(before_weight, cmap='gray')
plt.axis('off')
plt.title('Conv3 before pruning')
plt.subplot(122)
plt.imshow(pruned_weight, cmap='gray')
plt.axis('off')
plt.title('Conv3 after pruning')
save_path = current_path + '/pics/'
if not os.path.exists(save_path):
    os.makedirs(save_path)
plt.savefig(save_path + 'lenet_prune_visu.pdf')
plt.show()

#### test prune two layers

# print('prune conv3 and fc1')
# print('acc before pruning')
# evaluate_batch(model, test_loader, 'cuda')
# print('acc after pruning')
# optimizer.prune_layer_by_order_by_list(80, ['conv3.weight', 'fc1.weight'], True)
# evaluate_batch(model, test_loader, 'cuda')
# print('acc after recovering')
# optimizer.recover()
# evaluate_batch(model, test_loader, 'cuda')