cd /final/task3/DessiLBI

# train lenet with DessiLBI on MNIST in train_lenet.py

python train_lenet.py --batch_size 128 --lr 1e-2 --epoch 20 --interval 20 --kappa 1 --mu 20 --save_path 'model/mnist_1/'

# prune lenet in prune_lenet.py

python prune_lenet.py

|0    | 20    | 40    | 60    |80   |
|98.7 | 98.68 | 98.53 | 97.72 |93.42|
|conv1|conv2|conv3|fc1  |fc2  |
|89.42|85.59|93.42|98.67| 98.05|

# train vgg_bn with DessiLBI on CIFAR10 in train_cifar.py

python train_cifar.py --batch_size 128 --lr 1e-2 --epoch 20 --interval 20 --kappa 1 --mu 20 --save_path 'model/cifar_1/'

# prune vgg_bn in prune_cifar.py

python prune_cifar.py

|0    | 20    | 40    | 60    |80   |
|80.30| 67.52 | 34.54 | 14.67 |10.36|


| features.0 | features.4 | features.8 | features.15 | features.22 | classifier.0 | classifier.3 | classifier.6 |
|    10.05   | 11.52      | 10.36      |   58.49     |   72.38     |  72.84       |    80.34     |     80.32    |

# train vgg with DessiLBI_adam on CIFAR in train_lenet.py

python train_cifar_slbi_adam.py --batch_size 128 --lr 1e-2 --epoch 20 --interval 20 --kappa 1 --optim 'slbi_adam' --mu 20 --save_path 'model/cifar_2/'

# train vgg with adam on CIFAR in train_lenet.py
python train_cifar_slbi_adam.py --batch_size 128 --lr 1e-2 --epoch 20 --interval 20 --kappa 1 --optim 'adam' --mu 20 --save_path 'model/cifar_3/'

# train lenet with DessiLBI_adam on MNIST in train_lenet.py

python train_mnist_slbi_adam.py --batch_size 128 --lr 1e-2 --epoch 20 --interval 20 --kappa 1 --optim 'slbi_adam' --mu 20 --save_path 'model/mnist_2/'

# train lenet with adam on MNIST in train_lenet.py
python train_mnist_slbi_adam.py --batch_size 128 --lr 1e-2 --epoch 20 --interval 20 --kappa 1 --optim 'adam' --mu 20 --save_path 'model/mnist_3/'
