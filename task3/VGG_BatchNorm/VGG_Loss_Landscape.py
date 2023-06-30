import matplotlib as mpl
mpl.use('Agg')
import matplotlib.pyplot as plt
from torch import nn
import numpy as np
import torch
import os
import random
from tqdm import tqdm as tqdm
from IPython import display

from VGG_BatchNorm.models.vgg import VGG_A
from VGG_BatchNorm.models.vgg import VGG_A_BatchNorm
from VGG_BatchNorm.data.loaders import get_cifar_loader

# ## Constants (parameters) initialization
device_id = [0,1,2,3]
num_workers = 1
batch_size = 128


# add our package dir to path 
module_path = os.path.dirname(os.getcwd())
home_path = module_path
figures_path = os.path.join(home_path, 'reports', 'figures')
models_path = os.path.join(home_path, 'reports', 'models')
print(figures_path)
print(models_path)
# Make sure you are using the right device.
# device_id = device_id
# os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
# device = torch.device("cuda:{}".format(0) if torch.cuda.is_available() else "cpu")
device = torch.device('cuda:0')
print(device)
print(torch.cuda.get_device_name(0))



# Initialize your data loader and
# make sure that dataloader works
# as expected by observing one
# sample from it.
train_loader = get_cifar_loader(train=True)
val_loader = get_cifar_loader(train=False)
# for X,y in train_loader:
#     ## --------------------
#     # Add code as needed
#     plt.imshow(np.array(X[0,0,:,:]))
#     ## --------------------
#     break



# This function is used to calculate the accuracy of model classification
def get_accuracy(model, dataloader, device='cpu'):
    ## --------------------
    # Add code as needed
    correct = 0
    l = 0
    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)
            pred = model(x).argmax(1)
            correct += torch.sum(pred == y).item()
            l += len(y)
    return correct / l
    ## --------------------

# Set a random seed to ensure reproducible results
def set_random_seeds(seed_value=0, device='cpu'):
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    random.seed(seed_value)
    if device != 'cpu': 
        torch.cuda.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


# We use this function to complete the entire
# training process. In order to plot the loss landscape,
# you need to record the loss value of each step.
# Of course, as before, you can test your model
# after drawing a training round and save the curve
# to observe the training
def train(model, optimizer, criterion, train_loader, val_loader, scheduler=None, epochs_n=100, best_model_path=None):
    model.to(device)
    # learning_curve = [np.nan] * epochs_n
    # 评估每个epoch的模型效果
    train_accuracy_curve = [np.nan] * epochs_n
    val_accuracy_curve = [np.nan] * epochs_n
    # max_val_accuracy = 0
    # max_val_accuracy_epoch = 0

    # 记录每个step的train loss、最后一层的loss梯度和最后一层的weight，作为后续可视化的数据
    # batches_n = len(train_loader)
    losses_list = []
    grads = []
    weights = []
    for epoch in tqdm(range(epochs_n), unit='epoch'):
        if scheduler is not None:
            scheduler.step()
        model.train()

        # loss_list = []  # use this to record the loss value of each step
        # grad = []  # use this to record the loss gradient of each step
        # learning_curve[epoch] = 0  # maintain this to plot the training curve

        for data in train_loader:
            x, y = data
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad()
            prediction = model(x)
            loss = criterion(prediction, y)
            loss.backward()
            # You may need to record some variable values here
            # if you want to get loss gradient, use
            # grad = model.classifier[4].weight.grad.clone()
            ## --------------------
            # 取output layer参数
            weight = model.classifier[-1].weight.clone().detach()
            grad = model.classifier[-1].weight.grad.clone().detach()
            grads.append(grad.cpu().numpy())
            weights.append(weight.cpu().numpy())
            losses_list.append(loss.item())
            ## --------------------
            optimizer.step()

        # losses_list.append(loss_list)
        # grads.append(grad)
        # display.clear_output(wait=True)
        # f, axes = plt.subplots(1, 2, figsize=(15, 3))
        #
        # learning_curve[epoch] /= batches_n
        # axes[0].plot(learning_curve)

        # Test your model and save figure here (not required)
        # remember to use model.eval()
        ## --------------------
        # Add code as needed
        model.eval()
        train_accuracy_curve[epoch] = get_accuracy(model, train_loader, device)
        val_accuracy_curve[epoch] = get_accuracy(model, val_loader, device)
        ## --------------------
    return train_accuracy_curve, val_accuracy_curve, losses_list, grads, weights


# Train your model
# feel free to modify
epo = 20
lrs = [1e-3, 2e-3, 1e-4, 5e-4]

print(os.getcwd())

set_random_seeds(901, device=device)
print('************************************************')
# standard VGG模型
for lr in lrs:
    set_random_seeds(901, device=device)
    model = VGG_A()
    print('Begin the train of VGG, the lr is',lr)
    print('************************************************')
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    train_ac, val_ac, losses_list, grads, weights = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    if lr == lrs[0]:
        vgg_train_ac = train_ac
        vgg_val_ac = val_ac
        vgg_loss = losses_list
        vgg_grad = grads
        vgg_weight = weights
    else:
        vgg_train_ac = np.row_stack((vgg_train_ac, train_ac))
        vgg_val_ac = np.row_stack((vgg_val_ac, val_ac))
        vgg_loss = np.row_stack((vgg_loss, losses_list))
        vgg_grad = np.row_stack((vgg_grad, grads))
        vgg_weight = np.row_stack((vgg_weight, weights))

np.save('vgg_train_ac.npy', vgg_train_ac)
np.save('vgg_val_ac.npy', vgg_val_ac)
np.save('vgg_loss.npy', vgg_loss)
np.save('vgg_grad.npy', vgg_grad)
np.save('vgg_weight.npy', vgg_weight)

for lr in lrs:
    set_random_seeds(901, device = device)
    model = VGG_A_BatchNorm()
    print('Begin the train of VGG_BN, the lr is',lr)
    print('************************************************')
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    criterion = nn.CrossEntropyLoss()
    train_ac, val_ac, losses_list, grads, weights = train(model, optimizer, criterion, train_loader, val_loader, epochs_n=epo)
    if lr == lrs[0]:
        vgg_bn_train_ac = train_ac
        vgg_bn_val_ac = val_ac
        vgg_bn_loss = losses_list
        vgg_bn_grad = grads
        vgg_bn_weight = weights
    else:
        vgg_bn_train_ac = np.row_stack((vgg_bn_train_ac, train_ac))
        vgg_bn_val_ac = np.row_stack((vgg_bn_val_ac, val_ac))
        vgg_bn_loss = np.row_stack((vgg_bn_loss, losses_list))
        vgg_bn_grad = np.row_stack((vgg_bn_grad, grads))
        vgg_bn_weight = np.row_stack((vgg_bn_weight, weights))

np.save('vgg_bn_train_ac.npy', vgg_bn_train_ac)
np.save('vgg_bn_val_ac.npy', vgg_bn_val_ac)
np.save('vgg_bn_loss.npy', vgg_bn_loss)
np.save('vgg_bn_grad.npy', vgg_bn_grad)
np.save('vgg_bn_weight.npy', vgg_bn_weight)

print('The end of all train!')