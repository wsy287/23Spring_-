# python draw.py
import numpy as np
import matplotlib.pyplot as plt

recons = []
steps = [100,400,800,2000,4000,6000,10000]
ori = np.load('eval_output/origin.npy')
for i in range(7):
    save_path = 'eval_output/%d.npy' % steps[i]
    output = np.load(save_path)
    recons.append(output)
plt.figure(figsize=(16, 8))
plt.subplot(2, 4, 1)
plt.imshow(ori)
plt.axis("off")
plt.title("Ori.")
for i in range(3):
    plt.subplot(2, 4, i+2)
    plt.imshow(recons[i])
    plt.axis("off")
    plt.title(str(steps[i]))
# plt.subplot(2, 4, 5)
# plt.imshow(np.ones(ori.shape))
# plt.axis("off")
for i in range(4):
    plt.subplot(2, 4, 5+i)
    plt.imshow(recons[i+3])
    plt.axis("off")
    plt.title(str(steps[i+3]))
plt.savefig('pics/diff_steps_all',dpi=600)
plt.show()
