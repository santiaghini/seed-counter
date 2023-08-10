import matplotlib.pyplot as plt


def plot_full(img, title='', cmap='jet'):
    # add title
    # plt.text(0, 0, title, color='white', fontsize=8, ha='left', va='top')
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap)
    plt.axis('off')
    plt.title(title)
    plt.tight_layout()
    plt.show()