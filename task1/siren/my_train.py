import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from torchvision import datasets, transforms
from PIL import Image
from torchvision.transforms import Resize, Compose, ToTensor, Normalize
import numpy as np
import skimage
import matplotlib.pyplot as plt
from collections import OrderedDict
import argparse
from utils import *




parser = argparse.ArgumentParser()
parser.add_argument("--bs_sqrt", default=1, type=int)
parser.add_argument("--length", default=32, type=int)
parser.add_argument("--steps", default=100, type=int)
parser.add_argument("--steps_til_summary", default=10, type=int)
args = parser.parse_args()
total_steps = args.steps  # Since the whole image is our dataset, this just means 500 gradient descent steps.
steps_til_summary = args.steps_til_summary
bs_sqrt = args.bs_sqrt
bs = bs_sqrt**2
length = args.length
save_output_dir = 'output/'
save_pic_dir = 'pics/'
if not os.path.exists(save_pic_dir):
    os.makedirs(save_pic_dir)
if not os.path.exists(save_output_dir):
    os.makedirs(save_output_dir)
data_root = 'data/CIFAR10/'
transform = transforms.Compose([
        Resize(length),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])

# transform = transforms.Compose([
#         transforms.ToTensor(),
#         transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
#     ])
train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=bs, shuffle=False,drop_last=True)

img_siren = Siren(in_features=2, out_features=3, hidden_features=256,
                  hidden_layers=3, outermost_linear=True)
img_siren.cuda()
optim = torch.optim.Adam(lr=1e-4, params=img_siren.parameters())

ground_truth, _ = next(iter(train_loader))

sidelength = bs_sqrt*ground_truth.shape[2]
ground_truth = ground_truth.permute(0, 2, 3, 1)
ground_truth = torch.cat([torch.cat(([ground_truth[bs_sqrt*j+i] for i in range(bs_sqrt)]),0) for j in range(bs_sqrt)],1)

print(ground_truth.shape)
ground_truth = ground_truth.view(1,-1,3)
ground_truth = torch.unsqueeze(ground_truth, dim=0)
print(ground_truth.shape)
model_input = get_mgrid(sidelength, 2)


model_input, ground_truth = model_input.cuda(), ground_truth.cuda()
min_loss = 1
for step in range(total_steps):
    model_output, coords = img_siren(model_input)
    if step == 0:
        print(model_output.shape)
    loss = ((model_output - ground_truth) ** 2).mean()
    if loss < min_loss:
        min_loss = loss
        min_output = model_output
    if not (step+1) % steps_til_summary:
        psnr = -10 * torch.log10(loss/bs)
        print("Step %d, Total loss %0.6f, PSNR %0.6f" % (step+1, loss, psnr))

    optim.zero_grad()
    loss.backward()
    optim.step()
print(-10 * torch.log10(min_loss/bs))
ground_truth = ground_truth.cpu().view(sidelength, sidelength, 3).detach().numpy()
min_output = min_output.cpu().view(sidelength, sidelength, 3).detach().numpy()
mu = np.array([0.4914, 0.4822, 0.4465])
std = np.array([0.2023, 0.1994, 0.2010])
ground_truth = unnormalized_show(ground_truth,mu,std)
min_output = unnormalized_show(min_output,mu,std)
np.save(save_output_dir+'inr_recon_%d_%d_%d.npy' % (total_steps,bs_sqrt,length),np.array([ground_truth,min_output]))

fig, axes = plt.subplots(1, 2, figsize=(12, 6))
axes[0].imshow(ground_truth)
axes[1].imshow(min_output)
axes[0].axis("off")
axes[1].axis("off")
plt.savefig(save_pic_dir+'recon_%d_%d_%d.png' % (total_steps,bs_sqrt,length),dpi=600)
plt.show()