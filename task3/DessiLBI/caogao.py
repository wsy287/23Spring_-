# python caogao.py
import numpy as np
import matplotlib.pyplot as plt
for i in range(1,4):
    save_path = 'model/cifar_%d/test_acc.npy' % i
    test_acc = np.load(save_path)
    print(max(test_acc))
x = range(20)
titles = ['DessiLBI','DessiLBI with adam','Adam']
plt.figure(figsize=(15, 5))
for i in range(1,4):
    test_save_path = 'model/cifar_%d/test_acc.npy' % i
    test_acc = np.load(test_save_path)
    train_save_path = 'model/cifar_%d/train_acc.npy' % i
    train_acc = np.load(train_save_path)
    plt.subplot(1,3,i)
    plt.plot(x, train_acc, label='train accuracy')
    plt.plot(x, test_acc, label='test accuracy')
    plt.xlabel('epoch')
    plt.ylabel('accuracy')
    plt.legend(loc='best')
    plt.title(titles[i-1])
plt.savefig('pics/CIFAR10.png',dpi=600)
plt.show()

# for i in range(1,4):
#     save_path = 'model/mnist_%d/test_acc.npy' % i
#     test_acc = np.load(save_path)
#     print(max(test_acc))

# x = range(20)
# titles = ['DessiLBI','DessiLBI with adam','Adam']
# plt.figure(figsize=(15, 5))
# for i in range(1,4):
#     test_save_path = 'model/mnist_%d/test_acc.npy' % i
#     test_acc = np.load(test_save_path)
#     train_save_path = 'model/mnist_%d/train_acc.npy' % i
#     train_acc = np.load(train_save_path)
#     plt.subplot(1,3,i)
#     plt.plot(x, train_acc, label='train accuracy')
#     plt.plot(x, test_acc, label='test accuracy')
#     plt.xlabel('epoch')
#     plt.ylabel('accuracy')
#     plt.legend(loc='best')
#     plt.title(titles[i-1])
# plt.savefig('pics/MNIST.png',dpi=600)
# plt.show()