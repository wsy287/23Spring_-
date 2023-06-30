# python draw.py
import numpy as np
import matplotlib.pyplot as plt
recons = []
steps = [200,400,800,2000]
for i in range(4):
    save_path = 'output/inr_recon_%d_1_32.npy' % steps[i]
    output = np.load(save_path)
    ori = output[0]
    recons.append(output[1])
plt.figure(figsize=(20, 4))
plt.subplot(1, 5, 1)
plt.imshow(ori)
plt.axis("off")
plt.title("Ori.")
for i in range(4):
    plt.subplot(1, 5, i+2)
    plt.imshow(recons[i])
    plt.axis("off")
    plt.title(str(steps[i]))
plt.savefig('write/diff_steps',dpi=600)
plt.show()
