import matplotlib.pyplot as plt
import numpy as np

import torch

def Plot(x):
    x = np.array(x)

    for j in range(len(x)):

        data = x[j]

        fig, ax = plt.subplots()
        ax.imshow(data)

        # for a in range(11):
        #     for b in range(11):
        #         ax.text(b, a, np.round(data[a, b], 2), ha="center", va="center", color="w")

        fig.tight_layout()

        plt.savefig(f'imgs/{j}.png')
        plt.cla()