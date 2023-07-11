import matplotlib.pyplot as plt

def plot_full(img):
    plt.figure(figsize=(10,10))
    plt.imshow(img, cmap='jet')
    plt.axis('off')
    plt.tight_layout()
    plt.show()