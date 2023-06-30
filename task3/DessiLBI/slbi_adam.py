# from https://github.com/DessiLBI2020/DessiLBI/blob/master/DessiLBI/code/slbi_opt.py

import torch
from torch.optim.optimizer import Optimizer, required
import copy
import torch.nn.functional as F
from typing import List, Optional
from torch import Tensor
import math
import numpy as np

# 定义SLBI优化器

DEBUG = 0


# Adam加入参数beta和eps
class SLBI(Optimizer):
    def __init__(self, params, lr=required, kappa=1, mu=100, weight_decay=0, momentum=0.9, dampening=0,
                 betas=(0.9, 0.999), eps=1e-8):
        defaults = dict(lr=lr, kappa=kappa, mu=mu, weight_decay=weight_decay, momentum=momentum, dampening=dampening,
                        betas=betas, eps=eps)
        if DEBUG is True:
            print('*******************************************')
            for key in defaults:
                print(key, ' : ', defaults[key])
            print('*******************************************')
        super(SLBI, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(SLBI, self).__setstate__(state)

    def assign_name(self, name_list):
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                param_state['name'] = name_list[iter]

    def initialize_slbi(self, layer_list=None):
        if layer_list == None:
            pass
        else:
            for group in self.param_groups:
                for p in group['params']:
                    param_state = self.state[p]
                    if param_state['name'] in layer_list:
                        param_state['z_buffer'] = torch.zeros_like(p.data)
                        param_state['gamma_buffer'] = torch.zeros_like(p.data)

    def step(self, closure=None):
        loss = None
        if closure is not None:
            loss = closure()
        for group in self.param_groups:
            mu = group['mu']
            kappa = group['kappa']
            lr_kappa = group['lr'] * group['kappa']
            lr_gamma = group['lr'] / mu
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            beta1, beta2 = group['betas']
            eps = group['eps']

            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]

                if weight_decay != 0 and len(p.data.size()) != 1 and 'bn' not in param_state['name']:
                    d_p.add_(p.data, alpha=weight_decay)

                # if 'step' not in param_state:
                #     param_state['step'] = 0
                #     param_state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                #     param_state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)

                # param_state['step'] += 1

                # #参数step,avg,avg_sq
                # step = param_state['step']
                # exp_avg = param_state['exp_avg']
                # exp_avg_sq = param_state['exp_avg_sq']

                # #以step为指数计算bias_corr
                # bias_correction1 = 1 - beta1 ** step
                # bias_correction2 = 1 - beta2 ** step

                # #使用指数加权平均计算一阶和二阶梯度量，不断地更新avg和avg_sq
                # exp_avg.mul_(beta1).add_(d_p, alpha=1 - beta1)
                # exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p, value=1 - beta2)
                # denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                # step_size = lr_kappa / bias_correction1

                # p.data.addcdiv_(exp_avg, denom, value=-step_size)

                # 字典初始化
                if 'step' not in param_state:
                    param_state['step'] = 0
                    param_state['exp_avg'] = torch.zeros_like(p)
                    param_state['exp_avg_sq'] = torch.zeros_like(p)

                param_state['step'] += 1

                # 参数step,avg,avg_sq
                step = param_state['step']
                exp_avg = param_state['exp_avg']
                exp_avg_sq = param_state['exp_avg_sq']

                # 使用指数加权平均计算一阶和二阶梯度量，不断地更新avg和avg_sq
                exp_avg.mul_(beta1).add_(d_p, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(d_p, d_p, value=1 - beta2)

                # 以step为指数计算bias_corr
                bias_correction1 = 1 - beta1 ** step
                bias_correction2 = 1 - beta2 ** step
                step_size = lr_kappa / bias_correction1
                denom = (exp_avg_sq.sqrt() / math.sqrt(bias_correction2)).add_(eps)

                p.data.addcdiv_(exp_avg, denom, value=-step_size)

                if 'z_buffer' in param_state:
                    # 计算dp的梯度，加上gamma的部分
                    new_grad = d_p * lr_kappa + (p.data - param_state['gamma_buffer']) * lr_kappa / mu
                    last_p = copy.deepcopy(p.data)
                    p.data.add_(-new_grad)

                    # 更新v并存入z_buffer中
                    param_state['z_buffer'].add_(param_state['gamma_buffer'] - last_p, alpha=-lr_gamma)
                    if len(p.data.size()) == 2:
                        param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], 1)
                    elif len(p.data.size()) == 4:
                        param_state['gamma_buffer'] = kappa * self.shrink_group(param_state['z_buffer'])
                    else:
                        pass
                else:
                    p.data.add_(d_p, alpha=-lr_kappa)

    def calculate_w_star_by_layer(self, layer_name):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'z_buffer' in param_state and param_state['name'] == layer_name:
                    if len(p.data.size()) == 2:
                        param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                    elif len(p.data.size()) == 4:
                        param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                    else:
                        pass
                else:
                    pass

    def calculate_all_w_star(self):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'z_buffer' in param_state:
                    if len(p.data.size()) == 2:
                        param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                    elif len(p.data.size()) == 4:
                        param_state['w_star'] = p.data * (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                    else:
                        pass

    def calculate_layer_residue(self, layer_name):
        diff = 0
        for group in self.param_groups:
            mu = group['mu']
            for p in group['params']:
                param_state = self.state[p]
                if param_state['name'] == layer_name:
                    if 'gamma_buffer' in param_state:
                        diff = ((p.data - param_state['gamma_buffer']) * (
                                    p.data - param_state['gamma_buffer'])).sum().item()
                    else:
                        pass
        diff /= (2 * mu)
        print('Residue of' + layer_name + ' : ', diff)

    def calculate_all_residue(self):
        diff = 0
        for group in self.param_groups:
            mu = group['mu']
            for p in group['params']:
                param_state = self.state[p]
                if 'gamma_buffer' in param_state:
                    diff += ((p.data - param_state['gamma_buffer']) * (
                                p.data - param_state['gamma_buffer'])).sum().item()
        diff /= (2 * mu)
        print('Residue : ', diff)

    # 计算闭式解

    def shrink(self, s_t, lam):
        # proximal mapping for 2-d weight(fc layer)
        gamma_t = s_t.sign() * (torch.max(s_t.abs() - (lam * torch.ones_like(s_t)), torch.zeros_like(s_t)))
        return gamma_t

    def shrink_group(self, ts):
        # shrinkage for 4-d weight(conv layer)
        ts_reshape = torch.reshape(ts, (ts.shape[0], -1))
        ts_norm = torch.norm(ts_reshape, 2, 1)
        ts_shrink = torch.max(torch.zeros_like(ts_norm),
                              torch.ones_like(ts_norm) - torch.div(torch.ones_like(ts_norm), ts_norm))
        ts_return = torch.transpose(torch.mul(torch.transpose(ts_reshape, 0, 1), ts_shrink), 0, 1)
        ts_return = torch.reshape(ts_return, ts.shape)
        return ts_return


# 定义slbi工具箱
class SLBI_ADAM_ToolBox(SLBI):
    def use_w_star(self):
        # use sparse params to replace original params
        self.calculate_all_w_star()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'w_star' in param_state:
                    if len(p.data.size()) == 4:
                        ts_reshape = torch.reshape(param_state['w_star'], (param_state['w_star'].shape[0], -1))
                        ts_norm = torch.norm(ts_reshape, 2, 1)
                        num_selected_filters = torch.sum(ts_norm != 0).item()
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data = param_state['w_star']
                        print('max:', torch.max(param_state['w_star']))
                        print('min:', torch.min(param_state['w_star']))
                        print('number of filters: ', p.data.size()[0])
                        print('number of selected filter:', num_selected_filters)
                    elif len(p.data.size()) == 2:
                        num_selected_units = (param_state['w_star'] > 0.0).sum().item()
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data = param_state['w_star']
                        print('max:', torch.max(param_state['w_star']))
                        print('min:', torch.min(param_state['w_star']))
                        print('number of filters: ', p.data.size()[0] * p.data.size()[1])
                        print('number of selected units:', num_selected_units)
                    else:
                        pass

    def calculate_proportion(self, layer_name):
        # self.calculate_all_w_star()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'w_star' in param_state and param_state['name'] == layer_name:
                    # print(layer_name)
                    self.calculate_w_star_by_layer(layer_name)
                    if len(p.data.size()) == 4:
                        ts_reshape = torch.reshape(param_state['w_star'], (param_state['w_star'].shape[0], -1))
                        ts_norm = torch.norm(ts_reshape, 2, 1)
                        num_selected_filters = torch.sum(ts_norm != 0).item()
                        return num_selected_filters / p.data.size()[0]
                    elif len(p.data.size()) == 2:
                        num_selected_units = (param_state['w_star'] > 0.0).sum().item()
                        return num_selected_units / p.data.size()[0] * p.data.size()[1]
                    else:
                        pass

    def calculate_norm(self, layer_name):
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if param_state['name'] == layer_name:
                    layer_norm = torch.norm(torch.reshape(p.data, (p.data.shape[0], -1)), 2, 1)
        return layer_norm.cpu().detach().numpy()

    def cal_prune_thre(self, percent, layer_name):
        # prune layer according to given percent and layer name
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                if param_state['name'] in layer_name and 'prune_order' in param_state:
                    print(param_state['name'])
                    order = param_state['prune_order'].cpu().detach().numpy()
                    threshold = np.percentile(order, percent)
                    print('Threshold : ', threshold)
        return threshold

    def update_prune_order(self, epoch):
        self.calculate_all_w_star()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'z_buffer' in param_state:
                    if len(p.data.size()) == 4:
                        if 'epoch_record' not in param_state:
                            param_state['epoch_record'] = torch.zeros_like(p.data).add_(2000.0)
                            mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                            param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),
                                                                    param_state['epoch_record'])
                            epoch_per_filer, _ = torch.min(
                                torch.reshape(param_state['epoch_record'], (param_state['epoch_record'].shape[0], -1)),
                                dim=1)
                            param_state['prune_order'] = torch.norm(torch.reshape(p.data, (p.data.shape[0], -1)), 2,
                                                                    1) - epoch_per_filer
                        else:
                            mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                            # print(mask.size())
                            # print(param_state['epoch_record'].size())
                            param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),
                                                                    param_state['epoch_record'])
                            epoch_per_filer, _ = torch.min(
                                torch.reshape(param_state['epoch_record'], (param_state['epoch_record'].shape[0], -1)),
                                dim=1)
                            param_state['prune_order'] = torch.norm(torch.reshape(p.data, (p.data.shape[0], -1)), 2,
                                                                    1) - epoch_per_filer
                    elif len(p.data.size()) == 2:
                        if 'epoch_record' not in param_state:
                            param_state['epoch_record'] = torch.zeros_like(p.data).add_(2000.0)
                            mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                            param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),
                                                                    param_state['epoch_record'])
                            param_state['prune_order'] = torch.abs(p.data) - param_state['epoch_record']
                        else:
                            mask = (torch.gt(torch.abs(param_state['gamma_buffer']), 0.0)).float()
                            param_state['epoch_record'] = torch.min((1.0 - mask) * 2000.0 + mask * float(epoch),
                                                                    param_state['epoch_record'])
                            # param_state['prune_order'] = torch.abs(param_state['w_star']) - param_state['epoch_record']
                            param_state['prune_order'] = torch.abs(p.data) - param_state['epoch_record']
                    else:
                        pass

    def prune_layer_by_order_by_name(self, percent, layer_name, prune_bias):
        # prune layer according to given percent and layer name
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                if param_state['name'] == layer_name and 'prune_order' in param_state:
                    if DEBUG is True:
                        print(param_state['name'])
                    order = param_state['prune_order'].cpu().detach().numpy()
                    threshold = np.percentile(order, percent)
                    if len(p.data.size()) == 4:
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data[threshold > param_state['prune_order'], :, :, :] = 0.0
                        if prune_bias:
                            for k in range(iter + 1, len(group['params'])):
                                p_n = group['params'][k]
                                param_state_n = self.state[p_n]
                                if param_state_n['name'] == layer_name.replace('weight', 'bias'):
                                    if DEBUG is True:
                                        print(param_state_n['name'])
                                    param_state_n['original_params'] = copy.deepcopy(p_n.data)
                                    p_n.data[threshold > param_state['prune_order']] = 0.0
                    elif len(p.data.size()) == 2:
                        num_selected_units = (param_state['w_star'] > 0.0).sum().item()
                        mask = (torch.gt(param_state['prune_order'], threshold)).float()
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data = p.data * mask
                    else:
                        pass
                elif param_state['name'] in layer_name and 'prune_order' not in param_state:
                    print('Please Update Order First')
                else:
                    pass

    # 剪枝
    def prune_layer_by_order_by_list(self, percent, layer_name, prune_bias):
        # prune layer according to given percent and layer name
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                if param_state['name'] in layer_name and 'prune_order' in param_state:
                    if DEBUG is True:
                        print(param_state['name'])
                    order = param_state['prune_order'].cpu().detach().numpy()
                    threshold = np.percentile(order, percent)
                    if len(p.data.size()) == 4:
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data[threshold > param_state['prune_order'], :, :, :] = 0.0
                        if prune_bias:
                            for k in range(iter + 1, len(group['params'])):
                                p_n = group['params'][k]
                                param_state_n = self.state[p_n]
                                if param_state_n['name'] == param_state['name'].replace('weight', 'bias'):
                                    if DEBUG is True:
                                        print(param_state_n['name'])
                                    param_state_n['original_params'] = copy.deepcopy(p_n.data)
                                    p_n.data[threshold > param_state['prune_order']] = 0.0
                    elif len(p.data.size()) == 2:
                        num_selected_units = (param_state['w_star'] > 0.0).sum().item()
                        mask = (torch.gt(param_state['prune_order'], threshold)).float()
                        param_state['original_params'] = copy.deepcopy(p.data)
                        p.data = p.data * mask
                    else:
                        pass
                elif param_state['name'] in layer_name and 'prune_order' not in param_state:
                    print('Please Update Order First')
                else:
                    pass

    def recover(self):
        # in use_w_star or prune_layer, params are changed. so using recover() can give params original value
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'original_params' in param_state:
                    p.data = param_state['original_params']

    def extract_layer_weights(self, layer_name, number_select):
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                print(param_state['name'])
                if len(p.data.size()) == 4 and 'prune_order' in param_state and param_state['name'] == layer_name:
                    sorted_filter, indices = torch.sort(param_state['prune_order'], descending=True)
                    selected_filters = p.data[indices[0: number_select], :, :, :]
                    return selected_filters
                elif len(p.data.size()) == 2 and 'prune_order' in param_state and param_state['name'] == layer_name:
                    sorted_filter, indices = torch.sort(param_state['prune_order'], descending=True)
                    selected_filters = p.data[:, indices[0: number_select]]
                    return selected_filters
                else:
                    pass

    def extract_conv_and_fc_weights(self, layer_name, number_select):
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                print(param_state['name'])
                if len(p.data.size()) == 4 and 'prune_order' in param_state and param_state['name'] == layer_name:
                    sorted_filter, indices = torch.sort(param_state['prune_order'], descending=True)
                    selected_filters = p.data[indices[0: number_select], :, :, :]
                    for k in range(iter + 1, len(group['params'])):
                        p_fc = group['params'][k]
                        if len(p_fc.data.size()) == 2:
                            break
                    step = int(p_fc.data.size()[1] / p.data.size()[0])
                    # print(p_fc.data.size()[1])
                    # print(p.data.size())
                    fc_indice = []
                    for j in range(len(indices[0: number_select])):
                        fc_indice.extend(range(indices[j] * step, (indices[j] + 1) * step))
                    # print(fc_indice)
                    # print(p_fc.data.size())
                    selected_weights = p_fc.data[:, fc_indice]
                    return selected_filters, selected_weights
                else:
                    pass

    # 初始化
    def ortho_init(self, init_matrix, coordinate_matrix):
        _, w, h, cin = coordinate_matrix.size()
        coordinate_matrix_v = coordinate_matrix.view(-1, w * h * cin).transpose(0, 1)
        # print(coordinate_matrix_v.size())
        q, r = torch.qr(coordinate_matrix_v)
        r_d = torch.diag(r)
        # print(r_d)
        sorted_d, indices = torch.sort(torch.abs(r_d), descending=True)
        eps = 1e-10
        rnk = min(coordinate_matrix_v.size()) - torch.ge(sorted_d, eps).sum()
        # print('rank :' ,rnk)
        basis = q[:, indices[rnk:]].transpose(0, 1)
        basis = basis.view(-1, w, h, cin)
        stdv = 1. / math.sqrt(w * h * cin)
        nll_count = 0
        for k in range(init_matrix.size()[0]):
            if nll_count < basis.size()[0]:
                init_matrix[k, :, :, :] = torch.clamp(basis[k, :, :, :], -stdv, stdv)
                nll_count += 1
            else:
                init_matrix[k].uniform_(-stdv, stdv)

    def reinitialize(self, layer_list, percent, reinitialize_threshold=0):
        # print('Reinitialize')
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                if 'prune_order' in param_state and param_state['name'] in layer_list and len(p.data.size()) == 4:
                    proportion = self.calculate_proportion(param_state['name'])
                    # print(param_state['name'])
                    # print('Proportion : ', proportion)
                    if proportion > reinitialize_threshold:
                        sorted_filter, indices = torch.sort(param_state['prune_order'], descending=True)
                        border = int((1 - percent / 100) * p.data.size()[0])
                        keep_indice = indices[range(0, border)]
                        reinit_indice = indices[range(border, p.data.size()[0])]
                        self.ortho_init(p.data[reinit_indice, :, :, :], p.data[keep_indice, :, :, :])
                        param_state['prune_order'][reinit_indice] = -2000.0
                        param_state['gamma_buffer'][reinit_indice] = 0.0
                        param_state['z_buffer'][reinit_indice] = 0.0
                        param_state['w_star'][reinit_indice] = 0.0
                        for k in range(iter + 1, len(group['params'])):
                            p_b = group['params'][k]
                            p_b_state = self.state[p_b]
                            if len(p_b.data.size()) == 1 and p_b_state['name'] == param_state['name'].replace('bias',
                                                                                                              'weight'):
                                p_b.data[reinit_indice] = 0.0

    def calculate_mask(self, layer_name):
        self.calculate_all_w_star()
        for group in self.param_groups:
            for p in group['params']:
                param_state = self.state[p]
                if 'w_star' in param_state and param_state['name'] == layer_name:
                    if len(p.data.size()) == 4:
                        ts_reshape = torch.reshape(param_state['w_star'], (param_state['w_star'].shape[0], -1))
                        ts_norm = torch.norm(ts_reshape, 2, 1)
                        num_selected_filters = torch.sum(ts_norm != 0).item()
                        mask = torch.ones_like(param_state['w_star'])
                        mask[ts_norm != 0, :, :, :] = 0
                        return mask
                    elif len(p.data.size()) == 2:
                        return torch.ones_like(param_state['w_star'])
                        # return torch.gt(torch.abs(param_state['w_star']), 0.0).float()
                    else:
                        pass

    def step_with_freeze(self, freeze=True):
        loss = None
        for group in self.param_groups:
            mu = group['mu']
            kappa = group['kappa']
            lr_kappa = group['lr'] * group['kappa']
            lr_gamma = group['lr'] / mu
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            for p in group['params']:
                if p.grad is None:
                    continue
                d_p = p.grad.data
                param_state = self.state[p]
                if momentum != 0:
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.zeros_like(p.data)
                        buf.mul_(momentum).add_(d_p)
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    d_p = buf
                if weight_decay != 0 and len(p.data.size()) != 1 and 'bn' not in param_state['name']:
                    d_p.add_(weight_decay, p.data)
                if 'z_buffer' in param_state:
                    if freeze:
                        mask = self.calculate_mask(param_state['name'])
                    else:
                        mask = torch.ones_like(d_p)
                    new_grad = d_p * lr_kappa + (p.data - param_state['gamma_buffer']) * lr_kappa / mu
                    new_grad = new_grad * mask
                    last_p = copy.deepcopy(p.data)
                    p.data.add_(-new_grad)
                    param_state['z_buffer'].add_(-lr_gamma, mask * (param_state['gamma_buffer'] - last_p))
                    if len(p.data.size()) == 2:
                        param_state['gamma_buffer'] = kappa * self.shrink(param_state['z_buffer'], 1)
                    elif len(p.data.size()) == 4:
                        param_state['gamma_buffer'] = kappa * self.shrink_group(param_state['z_buffer'])
                    else:
                        pass
                else:
                    p.data.add_(-lr_kappa, d_p)

    def print_network(self):
        print('Printing Network')
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                print(param_state['name'], p.data.size())

    def generate_dict(self):
        net_dict = {}
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                s_name = param_state['name'].replace('module.', '')
                if s_name == 'conv1.weight':
                    net_dict['conv1.out'] = p.data.size()[0]
                elif s_name == 'fc.weight':
                    net_dict['fc.in'] = p.data.size()[1]
                elif len(p.data.size()) == 4:
                    n_name = param_state['name'].replace('module.', '')
                    n_name = n_name.replace('.weight', '')
                    print(n_name)
                    net_dict[n_name + '.in'] = p.data.size()[1]
                    net_dict[n_name + '.out'] = p.data.size()[0]
                else:
                    pass
        return net_dict

    def get_size(self, layer_name):
        for group in self.param_groups:
            for iter, p in enumerate(group['params']):
                param_state = self.state[p]
                if param_state['name'] == layer_name:
                    return p.data.size()