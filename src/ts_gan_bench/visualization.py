from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import numpy as np

def plot_tsne(real_data, fake_data, filename):
    data = np.concatenate([real_data, fake_data], axis=0)
    tsne = TSNE(n_components=2, perplexity=40)

    real_res = data[:len(real_data)]
    fake_res = data[len(real_data):]

    plt.figure(figsize=(8, 6))
    plt.scatter(
        real_res[:, 0],
        real_res[:, 1],
        label='Real',
        alpha=0.5,
        s=10,
    )
    plt.scatter(
        fake_res[:, 0],
        fake_res[:, 1],
        label='Synthetic',
        alpha=0.5,
        s=10,
    )
    plt.legend()
    plt.savefig(filename)
    plt.close()
