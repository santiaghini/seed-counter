import matplotlib.pyplot as plt


def plot_full(img, cmap='jet'):
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.show()