


from data import get_all, load_data
import numpy as np



_, b, _ = get_all("/data/DATASETS/pseudoDvs/prep/","dvs")

data, _ = load_data(train=b, test=None)

x,_ = data[14]



import matplotlib.pyplot as plt

plt.figure()
plt.imshow(np.sum(x,axis=-1)[0])
plt.tight_layout()
plt.axis('off')
plt.savefig("fig.png")