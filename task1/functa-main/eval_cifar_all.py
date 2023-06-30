import torch.nn as nn
import os
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from model import LatentModulatedSiren # 这个就是我们的模型F_theta
from utils import inner_step # 内圈循环
from cifar import CIFAR
from torch.utils.data import Dataset
from tqdm import tqdm

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

inner_steps = 200
inner_lr = 1e-2 # Follow原作超参设置

save_dir = './eval_all_output/%s/' % ('trainset' if args.train else 'testset')
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
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
data_loader = DataLoader(dataset, batch_size=model_cfg['batch_size'],shuffle=False,drop_last=True)

criterion = nn.MSELoss(reduction='none')
inner_optim = torch.optim.SGD([model.latent.latent_vector],lr=inner_lr) # 内圈优化只调整phi
print(len(data_loader))


idx = 0
modulations = []
for images, coords, _, _ in tqdm(data_loader):
    model.train()
    idx += 1
    print(idx,'\n')
    nn.init.constant_(model.latent.latent_vector, 0)
    images, coords = images.permute(0, 2, 3, 1).to(device), coords.to(device)
    for j in range(inner_steps):
        inner_loss = inner_step(images, coords, model, inner_optim, criterion)
        psnr = -10 * np.log10(inner_loss / model_cfg['batch_size'])
        psnr_list.append(psnr)
        if (j+1) % 50 == 0:
            print("Inner step {:3}, Inner loss {:.6f}, psnr {:.6f}".format(j+1,inner_loss, psnr))
    model.eval()
    modulate = model.latent.latent_vector.detach().cpu()
    modulations.append(modulate)
    if idx%100 == 0:
        np.save(save_dir + 'modulations_%d_%d_%d.npy' % (args.start_outer_save, inner_steps + args.start_inner_save,idx),
                np.array(modulations))
np.save(save_dir + 'modulations_%d_%d.npy' % (args.start_outer_save,inner_steps+args.start_inner_save),np.array(modulations))




