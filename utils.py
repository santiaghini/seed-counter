import matplotlib.pyplot as plt


def plot_full(img, title, cmap='jet'):
    # add title
    plt.title(title)
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.tight_layout()
    plt.show()