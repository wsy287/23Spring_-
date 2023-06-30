conda activate functa_env

cd /final/task1/siren

# different steps


| 100   | 200   | 400   | 800   | 2000  |
| ----- | ----- | ----- | ----- | ----- |
| 15.68 | 22.02 | 29.06 | 37.99 | 55.47 |

python my_train.py --bs_sqrt 1 --steps 20

3.57

python my_train.py --bs_sqrt 1 --steps 40

7.84

python my_train.py --bs_sqrt 1 --steps 80

13.63

python my_train.py --bs_sqrt 1 --steps 100

15.683850

python my_train.py --bs_sqrt 1 --steps 200

22.023895

python my_train.py --bs_sqrt 1 --steps 400

29.061583

python my_train.py --bs_sqrt 1 --steps 800

38.8005

python my_train.py --bs_sqrt 1 --steps 2000

55.4686

# different image size

python my_train.py --bs_sqrt 2 --steps 200

27.5486

python my_train.py --bs_sqrt 2 --steps 400

33.6051

python my_train.py --bs_sqrt 2 --steps 800

40.4774

python my_train.py --bs_sqrt 2 --steps 2000

52.764553

python my_train.py --bs_sqrt 3 --steps 200

27.1351

python my_train.py --bs_sqrt 3 --steps 400

35.3611

python my_train.py --bs_sqrt 3 --steps 800

43.7214

python my_train.py --bs_sqrt 3 --steps 2000

55.2709

python my_train.py --bs_sqrt 1 --length 64 --steps 800

37.8023

python my_train.py --bs_sqrt 1 --length 64 --steps 2000

50.1764

python my_train.py --bs_sqrt 1 --length 96 --steps 800

34.12

python my_train.py --bs_sqrt 1 --length 96 --steps 2000

44.1539

python my_train.py --bs_sqrt 1 --length 128 --steps 800

35.9966

python my_train.py --bs_sqrt 1 --length 128 --steps 2000

43.2462


| size          | 32    | 64    | 96    | 128   |
| ------------- | ----- | ----- | ----- | ----- |
| psnr_{s=800}  | 37.99 | 38.46 | 34.38 | 36.45 |
| psnr_{s=2000} | 55.47 | 50.12 | 44.34 | 42.96 |

# draw

python draw.py
