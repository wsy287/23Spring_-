import torch.nn as nn
import pickle as pk
import os
import numpy as np
import matplotlib.pyplot as plt
import torch
from torchvision import datasets, transforms
from model import LatentModulatedSiren # 这个就是我们的模型F_theta
from utils import inner_step # 内圈循环
from cifar import CIFAR
from torch.utils.data import Dataset
import argparse
def str2bool(value):
    return value.lower() == 'true'

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
parser = argparse.ArgumentParser()
parser.add_argument("--start_outer_save", default=0, type=int)
parser.add_argument("--start_inner_save", default=0, type=int)
parser.add_argument("--train", default=True, type=str2bool)
args = parser.parse_args()
model_cfg = {
    'batch_size': 128,
    'out_channels': 3,
    'depth': 15,
    'latent_dim': 512,
    'latent_init_scale': 0.01,
    'layer_sizes': [],
    'meta_sgd_clip_range': [0, 1],
    'meta_sgd_init_range': [0.005, 0.1],
    'modulate_scale': False,
    'modulate_shift': True,
    'use_meta_sgd': True,
    'w0': 30,
    'width': 512}

bs = 9
inner_steps = 100
inner_lr = 1e-2 # Follow原作超参设置

save_dir = './eval_output/%s/' % ('trainset' if args.train else 'testset')
print(save_dir)
model = LatentModulatedSiren(**model_cfg).to(device)
# 导入部分模型参数
model_dict =  model.state_dict()
psnr_list = []

if args.start_inner_save:
    state_dict = torch.load("./eval_output/%s/model_%d_%d.pth"  % ('trainset' if args.train else 'testset',args.start_outer_save,args.start_inner_save))
    # psnr_list = np.load("./eval_output/%s/psnr_%d_%d.npy"  % ('trainset' if args.train else 'testset',args.start_outer_save,args.start_inner_save))
    # psnr_list = psnr_list.tolist()
else:
    state_dict = torch.load("./output/iter_%d/model.pth" % args.start_outer_save)
    # 初始化phi_j = [0] * 512
    nn.init.constant_(model.latent.latent_vector, 0)

state_dict = {k:v for k,v in state_dict.items() if k in model_dict.keys()}
model_dict.update(state_dict)
model.load_state_dict(model_dict)

transform = transforms.Compose([
        transforms.ToTensor(),

    ])

data_root = 'data/CIFAR10/'

dataset = CIFAR(root=data_root,train=args.train, transform=transform) # 训练集用于优化theta

criterion = nn.MSELoss(reduction='none')
inner_optim = torch.optim.SGD([model.latent.latent_vector],lr=inner_lr) # 内圈优化只调整phi

# 取几个样本
images = []
coords = []
for i in range(model_cfg['batch_size']):
    i, c, _, _ = dataset.__getitem__(i)
    images.append(i)
    coords.append(c)


images = torch.Tensor(np.stack(images)).permute(0,2,3,1).to(device)
coords = torch.Tensor(np.stack(coords)).to(device)
print(images.shape)
print(coords.shape)
model.train()

for j in range(inner_steps):
    inner_loss = inner_step(images, coords, model, inner_optim, criterion)
    psnr = -10 * np.log10(inner_loss / model_cfg['batch_size'])
    psnr_list.append(psnr)
    if (j+1) % 100 == 0:
        print("Inner step {:3}, Inner loss {:.6f}, psnr {:.6f}".format(j,inner_loss, psnr))

modulate = model.latent.latent_vector.detach() # detach
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
torch.save(model.state_dict(), os.path.join(save_dir, 'model_%d_%d.pth' % (args.start_outer_save,inner_steps+args.start_inner_save)))
model.eval()
np.save(save_dir + 'psnr_%d_%d.npy' % (args.start_outer_save,inner_steps+args.start_inner_save),np.array(psnr_list))
pp_out = model(coords, modulate) # 推理

loss = criterion(pp_out, images).mean([1, 2, 3])

_, topidx = loss.topk(k=4, dim=0, largest=False, sorted=True)
_, lastidx = loss.topk(k=4, dim=0, largest=True, sorted=True)

# images[0] = transform(images[0])
# pp_out[0] = transform(pp_out[0])
images = images.cpu().detach().numpy()
pp_out = pp_out.cpu().detach().numpy()
# np.save('eval_output/origin.npy',images[0])
np.save('eval_output/%d.npy' % args.start_inner_save,pp_out[0])
# images = images[:,:, :, ::-1]
# pp_out = pp_out[:,:, :, ::-1]

# images = images[:,:, :, [0,2,1]]
# pp_out = pp_out[:,:, :, [0,2,1]]
# plt.figure(figsize=(15, 8))
#
# for i in range(4):
#     plt.subplot(2, 4, i * 2 + 1)
#     plt.imshow(images[topidx[i]])
#     plt.axis("off")
#     plt.title("Source")
#
#     plt.subplot(2, 4, i * 2 + 2)
#     plt.imshow(pp_out[topidx[i]])
#
#     plt.axis("off")
#     plt.title("Generated")
# plt.savefig('pics/%s/cifar_eval_top_%d_%d.png' % ('trainset' if args.train else 'testset',args.start_outer_save,inner_steps+args.start_inner_save))
#
# plt.figure(figsize=(15, 8))
#
# for i in range(4):
#     plt.subplot(2, 4, i * 2 + 1)
#     plt.imshow(images[lastidx[i]])
#     plt.axis("off")
#     plt.title("Source")
#
#     plt.subplot(2, 4, i * 2 + 2)
#     plt.imshow(pp_out[lastidx[i]])
#
#     plt.axis("off")
#     plt.title("Generated")
# plt.savefig('pics/%s/cifar_eval_last_%d_%d.png' % ('trainset' if args.train else 'testset',args.start_outer_save,inner_steps+args.start_inner_save))

# fig, axes = plt.subplots(1, 2, figsize=(12, 6))
# axes[0].imshow(images[0])
# axes[1].imshow(pp_out[0])
# axes[0].axis("off")
# axes[1].axis("off")
# plt.savefig('pics/recon_0_%d_%d.png' % (args.start_outer_save,inner_steps+args.start_inner_save),dpi=600)
# plt.show()