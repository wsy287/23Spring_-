import os
import argparse
import torch
import torch.nn as nn
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import numpy as np
import torch.nn.functional as F
import model
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)

def inner_step(images, coords, network, optim, criterion, modulate=None, meta_sgd=False):
    """
    batch_ratio: 如果推理生成modulation时bs与训练时不同，MSE loss的mean策略会导致梯度变化
        如训练时bs=4，预测时bs=1，则预测时梯度比训练时大了4倍
    """
    recon = network(coords, modulate)

    # print("*********** recon shape:", recon.shape)
    # print("*********** images shape:", images.shape)
    # recon = recon.reshape(images.shape)
    # print("*********** recon shape:", recon.shape)
    loss = criterion(recon, images)  # N, H, W, C
    loss = loss.mean([1, 2, 3]).sum(0)

    loss.backward()

    if meta_sgd:
        print(network.latent.latent_vector.grad.sum().item(), network.meta_sgd_lrs().sum().item())
        with torch.no_grad():
            meta_lr = network.meta_sgd_lrs()
            mod_grad = modulate.grad if modulate is not None else network.latent.latent_vector.grad
            # paddle.assign(meta_lr * mod_grad, mod_grad)
        print(network.latent.latent_vector.grad.sum().item(), network.meta_sgd_lrs().sum().item())
    optim.step()
    optim.zero_grad()
    network.zero_grad()

    return loss.item()


def outer_step(images, coords, network, optim, criterion, modulate=None):
    recon = network(coords, modulate)
    # print("*********** recon shape:", recon.shape)
    # print("*********** images shape:", images.shape)
    # recon = recon.reshape(images.shape)
    # print("*********** recon shape:", recon.shape)
    loss = criterion(recon, images)
    loss = loss.mean([1, 2, 3]).sum(0)

    loss.backward()
    optim.step()
    optim.zero_grad()
    network.zero_grad()
    return loss.item()
    # return loss.detach().numpy()


def parse_args():
    """
    command args
    """
    parser = argparse.ArgumentParser(description='Model training')
    # params of training
    parser.add_argument(
        "--dset", dest="dset", help="Use dataset", default=None, type=str)

    parser.add_argument(
        "--output", dest="output", help="Rootpath of output", default="./output/", type=str)

    parser.add_argument(
        "--batch_size", dest="batch_size", default=16, type=int)

    parser.add_argument(
        "--start_save", dest="start_save", default=0, type=int)

    return parser.parse_args()


if __name__ == "__main__":
    # Args
    args = parse_args()
    # Config
    batch_size = args.batch_size
    inner_steps = 3
    outer_steps = 100000
    inner_lr = 1e-3
    outer_lr = 3e-6
    latent_init_scale = 0.01
    save_interval = 100  # save ckpt interval
    ckpt_dir = args.output

    # Dataloader
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010))
    ])
    data_root = 'data/CIFAR10/'
    train_dataset = datasets.CIFAR10(root=data_root, train=True, download=True, transform=transform)
    train_loader = DataLoader(dataset=train_dataset, batch_size=args.batch_size, shuffle=True,drop_last=True)

    # Prepare
    if not os.path.exists(ckpt_dir):
        os.makedirs(ckpt_dir)

    # Model
    model_cfg = {
        'batch_size': batch_size,
        'out_channels': 3,  # RGB
        'depth': 15,
        'latent_dim': 512,
        'latent_init_scale': 0.1,
        'layer_sizes': [],
        'meta_sgd_clip_range': [0, 1],
        'meta_sgd_init_range': [0.005, 0.1],
        'modulate_scale': False,
        'modulate_shift': True,
        'use_meta_sgd': False,
        'w0': 30,
        'width': 512}

    if args.dset == "mnist":
        model_cfg['out_channels'] = 1

    network = model.LatentModulatedSiren(**model_cfg).to(device)
    if args.start_save:
        network.load_state_dict(torch.load("output/iter_%d/model.pth" % args.start_save))
        print("Load iter_%d state_dict done!" % args.start_save)
    # Optimizer
    ## Inner optimizer
    inner_optim = torch.optim.SGD([network.latent.latent_vector],lr=inner_lr)
    # inner_optim = torch.optim.SGD([network.latent.latent_vector, network.meta_sgd_lrs.meta_sgd_lrs],lr=inner_lr)

    ## Outer optimizer
    outer_optim = torch.optim.Adam([p for n, p in network.named_parameters() if n != "latent.latent_vector"],lr=outer_lr, weight_decay=1e-4)

    # Loss
    criterion = nn.MSELoss(reduction='none')


    # Train loop
    iter = 0
    iterator = train_loader.__iter__()
    inner_loss_list = [np.nan] * outer_steps * inner_steps
    outer_loss_list = [np.nan] * outer_steps
    inner_psnr_list = [np.nan] * outer_steps * inner_steps
    outer_psnr_list = [np.nan] * outer_steps
    while iter < outer_steps:
        try:
            images, labels  = iterator.next()
            coords = utils.get_coordinate_grid(images.shape[2]).astype("float32")
        except StopIteration:
            iterator = train_loader.__iter__()
            images, labels  = iterator.next()
            coords = utils.get_coordinate_grid(images.shape[2]).astype("float32")
        coords = np.expand_dims(coords, 0).repeat(args.batch_size, axis=0)
        coords = torch.tensor(coords)
        images = images.permute(0,2,3,1)
        images,coords = images.to(device),coords.to(device)
        # print(coords.shape)
        # print(images.shape)

        iter += 1
        if iter > outer_steps: break

        # network.latent.latent_vector = torch.Tensor(np.zeros(network.latent.latent_vector.shape).astype("float32"))

        nn.init.constant_(network.latent.latent_vector, 0)

        for j in range(inner_steps):
            inner_loss = inner_step(images, coords,
                                    network, inner_optim, criterion,
                                    meta_sgd=model_cfg['use_meta_sgd'])
            psnr = -10 * np.log10(inner_loss / batch_size)
            print("Outer iter {}: [{}/{}], inner loss {:.6f}, psnr {:.6f}".format(iter + 1, j + 1,
                                                                                  inner_steps, inner_loss,
                                                                                  psnr))

        modulate = network.latent.latent_vector.detach()  # detach
        outer_loss = outer_step(images, coords, network, outer_optim, criterion,
                                modulate)

        psnr = -10 * np.log10(outer_loss / batch_size)
        outer_loss_list[iter] = outer_loss
        outer_psnr_list[iter] = psnr
        print("Outer iter {}/{}: outer loss {:.6f}, outer PSNR {:.6f}".format(iter + 1,
                                                                              outer_steps, outer_loss, psnr))

        if iter > 0 and iter % save_interval == 0:
            current_save_dir = os.path.join(ckpt_dir, "iter_{}".format(iter+args.start_save))
            if not os.path.exists(current_save_dir):
                os.makedirs(current_save_dir)
            torch.save(network.state_dict(), os.path.join(current_save_dir, 'model.pth'))
            np.save(os.path.join(current_save_dir, 'outer_loss.npy'), outer_loss_list)
            np.save(os.path.join(current_save_dir, 'outer_psnr.npy'), outer_psnr_list)
    iterator.__del__()