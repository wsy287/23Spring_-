import torch
import numpy as np
def save_model_and_optimizer(model, optimizer, path):
    save_dict = {'model': model.state_dict(), 'optimizer': optimizer.state_dict()}
    torch.save(save_dict, path)
def get_coordinate_grid(res: int, centered: bool = True):
    """Returns a normalized coordinate grid for a res by res sized image.
      Args:
        res (int): Resolution of image.
        centered (bool): If True assumes coordinates lie at pixel centers. This is
          equivalent to the align_corners argument in Pytorch. This should always be
          set to True as this ensures we are consistent across different
          resolutions, but keep False as option for backwards compatibility.

      Returns:
        Jnp array of shape (height, width, 2).

      Notes:
        Output will be in [0, 1] (i.e. coordinates are normalized to lie in [0, 1]).
    """
    if centered:
        half_pixel = 1. / (2. * res)  # Size of half a pixel in grid
        coords_one_dim = np.linspace(half_pixel, 1. - half_pixel, res)
    else:
        coords_one_dim = np.linspace(0, 1, res)
    # Array will have shape (height, width, 2)
    return np.stack(
        np.meshgrid(coords_one_dim, coords_one_dim, indexing='ij'), axis=-1)
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
            mod_grad.data = meta_lr * mod_grad
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


