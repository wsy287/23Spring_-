# python classify_cifar.py
import os
from torch.utils.data import DataLoader
from tqdm import tqdm
import numpy as np
from torch import nn
import torch
import torch.nn.functional as F
from model import LatentModulatedSiren
from torchvision import datasets, transforms
from mlp import MLP,MLP2
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
save_dir = 'classify_cifar/'
if not os.path.exists(save_dir):
    os.makedirs(save_dir)
transform = transforms.Compose([
        transforms.ToTensor(),
    ])

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

# state_dict = torch.load('eval_all_output/trainset/model_32300_100.pth')
# model = LatentModulatedSiren(**model_cfg).to(device)
# model_dict =  model.state_dict()
# state_dict = {k:v for k,v in state_dict.items() if k in model_dict.keys()}
# model_dict.update(state_dict)
# model.load_state_dict(model_dict)

latent_vector = np.load('eval_all_output/trainset/modulations_32300_200.npy',allow_pickle=True)
test_vector = np.load('eval_all_output/testset/modulations_32300_200.npy',allow_pickle=True)
print(latent_vector.shape)
print(latent_vector[0].shape)
print(test_vector.shape)
print(test_vector[0].shape)
# get labels
# train_loader = get_cifar_loader(train=True)
# val_loader = get_cifar_loader(train=False)
data_root = 'data/CIFAR10/'
train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
test_dataset = datasets.CIFAR10(root=data_root, train=False, download=True, transform=transform)
train_loader = DataLoader(dataset=train_dataset, batch_size=model_cfg['batch_size'], shuffle=False, drop_last=True)
test_loader = DataLoader(dataset=test_dataset, batch_size=model_cfg['batch_size'], shuffle=False, drop_last=True)


def train(model, optimizer, latent_vector, vector, train_loader, test_loader, epochs_n=100, scheduler=None):
    model.to(device)
    train_accuracy_curve = [np.nan] * epochs_n
    test_accuracy_curve = []
    loss_list = [np.nan] * epochs_n
    model.train()
    for ep in tqdm(range(epochs_n)):
        loss_val = 0
        correct = num = 0
        for iter, pack in enumerate(train_loader):
            x = latent_vector[iter].to(device)
            y = pack[1].to(device)
            prediction = model(x)
            loss = F.cross_entropy(prediction, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            _, pred = prediction.max(1)
            loss_val += loss.item()
            correct += pred.eq(y).sum().item()
            num += x.shape[0]
            if (iter + 1) % (len(train_loader) // 5) == 0:
                print('epoch[%d/%d]: iter=%d, loss=%f, Train_ACC=%f' % (
                    ep + 1, epochs_n, iter + 1, loss_val / 100, correct / num))
        loss_val /= num
        train_ac = correct / num
        train_accuracy_curve[ep] = train_ac
        loss_list[ep] = loss_val
        if (ep+1) % 100 == 0:
            test_ac = test(model, vector, test_loader, device)
            test_accuracy_curve.append(test_ac)
            torch.save(model.state_dict(), save_dir+'%d_w.pth' % ep)
            np.save(save_dir+'loss_%d.npy' % ep,loss_list)
            np.save(save_dir+'train_acc_%d.npy' % ep,train_accuracy_curve)
    torch.save(model.state_dict(), save_dir + '%d_w.pth' % ep)
    test(model, vector, test_loader, device)
def test(model, vector, data_loader, device):
    model.eval()
    correct = num = 0
    for iter, pack in enumerate(data_loader):
        x = vector[iter].to(device)
        y = pack[1].to(device)
        prediction = model(x)
        _, pred = prediction.max(1)
        correct += pred.eq(y).sum().item()
        num += x.shape[0]
    print('Correct : ', correct)
    print('Num : ', num)
    print('Test ACC : ', correct / num)
    torch.cuda.empty_cache()
    model.train()
    return correct / num
lr = 1e-3
model = MLP()
optimizer = torch.optim.Adam(model.parameters(), lr = lr, weight_decay=1e-4)
train(model, optimizer, latent_vector, test_vector,train_loader, test_loader, epochs_n=500,scheduler=None)

