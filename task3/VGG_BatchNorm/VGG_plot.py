# -*- coding = utf-8 -*-
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm as tqdm

lrs = [1e-3, 2e-3, 1e-4, 5e-4]

# 导入存储的训练数据
save_path = ''
vgg_train_ac = np.load(save_path+'vgg_train_ac.npy')
vgg_val_ac = np.load(save_path+'vgg_val_ac.npy')
vgg_loss = np.load(save_path+'vgg_loss.npy')
vgg_grad = np.load(save_path+'vgg_grad.npy')
vgg_weight = np.load(save_path+'vgg_weight.npy')

vgg_bn_train_ac = np.load(save_path+'vgg_bn_train_ac.npy')
vgg_bn_val_ac = np.load(save_path+'vgg_bn_val_ac.npy')
vgg_bn_loss = np.load(save_path+'vgg_bn_loss.npy')
vgg_bn_grad = np.load(save_path+'vgg_bn_grad.npy')
vgg_bn_weight = np.load(save_path+'vgg_bn_weight.npy')

# 可视化
fig_save = 'plot_result/'
# 可视化VGG和VGG with BN的训练结果
def VGG_tra_res():
    ax = plt.gca()
    ax.patch.set_facecolor("#ECECF0")
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['bottom'].set_visible(False)  # 去掉下边框
    ax.spines['left'].set_visible(False)  # 去掉左边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.grid(axis='both', color='w', linewidth=0.5)
    x = range(1, vgg_train_ac.shape[1]+1, 1)
    plt.plot(x, vgg_train_ac[0], color='#77dfb1', label = 'Standard VGG')
    plt.plot(x, vgg_bn_train_ac[0], color='#8ea4f7', label = 'Standard VGG + BatchNorm')
    plt.title('Training results with learning rate:'+str(lrs[0]))
    plt.xlabel('Epoch')
    plt.ylabel('Train set accuracy')
    plt.legend(loc='lower right')
    plt.show()


num_lr = len(lrs)
num_step = vgg_loss.shape[1]

# 可视化gradient predictiveness
def VGG_Grad_Pred():
    bin = 40
    grad_dis = np.zeros((num_lr, int(num_step-1)))
    for lr in range(num_lr):
        for step in range(num_step-1):
            grad_dis[lr][step] = np.linalg.norm(vgg_grad[lr*num_step+step]-vgg_grad[lr*num_step+step+1])
    max_grad_pred = np.max(grad_dis, axis=0)[::bin]
    min_grad_pred = np.min(grad_dis, axis=0)[::bin]

    grad_bn_dis = np.zeros((num_lr, num_step - 1))
    for lr in range(num_lr):
        for step in range(num_step - 1):
            grad_bn_dis[lr][step] = np.linalg.norm(vgg_bn_grad[lr*num_step+step] - vgg_bn_grad[lr*num_step+step + 1])
    max_bn_grad_pred = np.max(grad_bn_dis, axis=0)[::bin]
    min_bn_grad_pred = np.min(grad_bn_dis, axis=0)[::bin]

    x = range(1, num_step-1, bin)
    begin = 1
    x = x[begin:]
    max_grad_pred = max_grad_pred[begin:]
    min_grad_pred = min_grad_pred[begin:]
    max_bn_grad_pred = max_bn_grad_pred[begin:]
    min_bn_grad_pred = min_bn_grad_pred[begin:]
    ax = plt.gca()
    ax.patch.set_facecolor("#ECECF0")
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['bottom'].set_visible(False)  # 去掉下边框
    ax.spines['left'].set_visible(False)  # 去掉左边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.grid(axis='both', color='w', linewidth=0.5)
    plt.plot(x, max_grad_pred, color='#77dfb1')
    plt.plot(x, min_grad_pred, color='#77dfb1')
    plt.fill_between(x, max_grad_pred, min_grad_pred, facecolor='#A6EBCD', label='Standard VGG')
    plt.plot(x, max_bn_grad_pred, color='#8ea4f7')
    plt.plot(x, min_bn_grad_pred, color='#8ea4f7')
    plt.fill_between(x, max_bn_grad_pred, min_bn_grad_pred, facecolor='#B8C4F3', label='Standard VGG + BatchNorm')

    plt.legend(loc='upper right')
    plt.title('Gradient Predictiveness')
    plt.xlabel('Steps')
    plt.ylabel('Gradient Predictiveness')
    plt.savefig(fig_save + 'Gradient_Pred.png', dpi=600)
    plt.show()


# 可视化beta_smoothness
def VGG_Beta_Smooth():
    bin = 20
    beta = np.zeros((num_lr, int(num_step-1)))
    for lr in range(num_lr):
        for step in range(num_step-1):
            # beta[lr][step] = np.linalg.norm(vgg_grad[lr*num_step+step]-vgg_grad[lr*num_step+step+1])/ \
            #                  (np.linalg.norm(vgg_grad[lr*num_step+step]*lrs[lr]) + 1e-3)
            beta[lr][step] = np.linalg.norm(vgg_grad[lr * num_step + step] - vgg_grad[lr * num_step + step + 1]) / \
                             np.linalg.norm(vgg_weight[lr * num_step + step] - vgg_weight[lr * num_step + step + 1])
    max_beta_smooth = np.max(beta, axis=0)[::bin]
    min_beta_smooth = np.min(beta, axis=0)[::bin]

    beta_bn = np.zeros((num_lr, num_step - 1))
    for lr in range(num_lr):
        for step in range(num_step - 1):
            # beta_bn[lr][step] = np.linalg.norm(vgg_bn_grad[lr*num_step+step] - vgg_bn_grad[lr*num_step+step + 1])/ \
            #                     (np.linalg.norm(vgg_bn_grad[lr*num_step+step]*lrs[lr])+1e-3)
            beta_bn[lr][step] = np.linalg.norm(vgg_bn_grad[lr * num_step + step] - vgg_bn_grad[lr * num_step + step + 1]) / \
                             np.linalg.norm(vgg_bn_weight[lr * num_step + step] - vgg_bn_weight[lr * num_step + step + 1])
    max_bn_beta_smooth = np.max(beta_bn, axis=0)[::bin]
    min_bn_beta_smooth = np.min(beta_bn, axis=0)[::bin]

    x = range(1, num_step-1, bin)
    begin = 10
    x = x[begin:]
    max_beta_smooth = max_beta_smooth[begin:]
    min_beta_smooth = min_beta_smooth[begin:]
    max_bn_beta_smooth = max_bn_beta_smooth[begin:]
    min_bn_beta_smooth = min_bn_beta_smooth[begin:]
    ax = plt.gca()
    ax.patch.set_facecolor("#ECECF0")
    ax.spines['top'].set_visible(False)  # 去掉上边框
    ax.spines['bottom'].set_visible(False)  # 去掉下边框
    ax.spines['left'].set_visible(False)  # 去掉左边框
    ax.spines['right'].set_visible(False)  # 去掉右边框
    plt.grid(axis='both', color='w', linewidth=0.5)
    plt.plot(x, max_beta_smooth, color='#77dfb1')
    plt.plot(x, min_beta_smooth, color='#77dfb1')
    plt.fill_between(x, max_beta_smooth, min_beta_smooth, facecolor = '#A6EBCD', label='Standard VGG')
    plt.plot(x, max_bn_beta_smooth, color='#8ea4f7')
    plt.plot(x, min_bn_beta_smooth, color='#8ea4f7')
    plt.fill_between(x, max_bn_beta_smooth, min_bn_beta_smooth, facecolor = '#B8C4F3', label='Standard VGG + BatchNorm')
    plt.legend(loc='upper right')
    plt.title('effective beta-smoothness')
    plt.xlabel('Steps')
    plt.ylabel('beta-smoothness')
    plt.savefig(fig_save+'beta-smoothness.png',dpi=600)
    plt.show()


if __name__=='__main__':
    print('Begin plot...')
    VGG_tra_res()
    print('Plot train results over!')
    VGG_Grad_Pred()
    print('Plot grad pred over!')
    VGG_Beta_Smooth()
    print('All plot works end!')