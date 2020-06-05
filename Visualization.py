import umap
import numpy as np
import matplotlib.pyplot as plt


class Visualization:

    def __init__(self, data, target):
        self.data = data
        self.target = target
        self.ncolors = len(set(target))

    def reduce_scatter_plot(self):
        embedding = umap.UMAP().fit_transform(self.data)
        
        plt.scatter(embedding[:, 0], embedding[:, 1], c=self.target, cmap='Spectral', s=5)
        plt.gca().set_aspect('equal', 'datalim')
        plt.colorbar(boundaries=np.arange(self.ncolors + 1)-0.5).set_ticks(np.arange(self.ncolors))
        plt.title('UMAP projection of the dataset', fontsize=24);
        plt.show()
