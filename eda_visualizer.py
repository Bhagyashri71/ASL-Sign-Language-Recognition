##### EDA and visualization code #############


import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

def plot_label_distribution(labels):
    label_counts = pd.Series(labels).value_counts().sort_index()
    plt.figure(figsize=(12, 6))
    sns.barplot(x=[chr(i + 65) for i in label_counts.index], y=label_counts.values)
    plt.title("Image Count per ASL Letter")
    plt.xlabel("ASL Letter")
    plt.ylabel("Number of Images")
    plt.grid(True)
    plt.show()

def show_sample_images(X, labels):
    plt.figure(figsize=(14, 10))
    shown = 0
    for cls in range(26):
        indices = np.where(np.array(labels) == cls)[0]
        if len(indices) == 0:
            continue
        plt.subplot(4, 7, shown + 1)
        plt.imshow(X[indices[0]])
        plt.title(f"Label: {chr(cls + 65)}")
        plt.axis('off')
        shown += 1
    plt.tight_layout()
    plt.show()
