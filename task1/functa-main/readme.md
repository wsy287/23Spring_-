# environment prepare

conda create -n functa_env python=3.8
conda activate functa_env
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt

cd /final/task1/functa-main

# train CIFAR10

python train_CIFAR.py --batch_size 128
python train_CIFAR.py --batch_size 128 --start_save 3800
python train_CIFAR.py --batch_size 128 --start_save 6300
python train_CIFAR.py --batch_size 128 --start_save 11700
python train.py --batch_size 128

# eval recon CIFAR10 train set

python eval_cifar_demo.py --start_outer_save 11700
python eval_cifar_demo.py --start_outer_save 32300

# eval recon CIFAR10 test set

python eval_cifar_demo.py --start_outer_save 11700 --train False
python eval_cifar_demo.py --start_outer_save 32300 --train False

# eval recon CIFAR10 through inner trained model

python eval_cifar_demo.py --start_outer_save 32300 --start_inner_save 1000

# get modulations for all training set and test set

python eval_cifar_all.py --start_outer_save 32300

python eval_cifar_all.py --start_outer_save 32300 --train False

# train classify model

python classify_cifar.py

MLP without weight_decay


| epoch     | 100    | 200    | 300    | 400    | 500    | 600    | 700    |
| --------- | ------ | ------ | ------ | ------ | ------ | ------ | ------ |
| test_acc  | 0.4109 | 0.4230 | 0.4413 | 0.4377 | 0.4346 | 0.4376 | 0.4345 |
| train_acc | 0.4870 | 0.5730 | 0.6171 | 0.6485 | 0.6636 | 0.6828 | 0.6943 |

MLP2 without weight_decay
| epoch    | 100    | 200    | 300    | 400    | 500    | 600    |
| -------- | ------ | ------ | ------ | ------ |--------|--------|
| test_acc | 0.4220 | 0.4219 | 0.4354 | 0.4324 | 0.4313 | 0.4307 |
|train_acc | 0.4994 | 0.5600 | 0.5985 | 0.6297 | 0.6503 |0.6628|
MLP with weight_decay 1e-4
| epoch    | 100    | 200    | 300    | 400    | 500    | 
| -------- | ------ | ------ | ------ | ------ |--------|
| test_acc | 0.4156 | 0.4287 | 0.4251 | 0.4273 | 0.4321 | 
|train_acc | 0.4579 | 0.5017 | 0.5331 | 0.5369 | 0.5529 |