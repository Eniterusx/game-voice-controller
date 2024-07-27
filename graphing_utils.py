import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

label_dict = {
    "_silence_": 0,
    "_unknown_": 1,
    "down": 2,
    "go": 3,
    "left": 4,
    "no": 5,
    "off": 6,
    "on": 7,
    "right": 8,
    "stop": 9,
    "up": 10,
    "yes": 11,
    "zero": 12,
    "one": 13,
    "two": 14,
    "three": 15,
    "four": 16,
    "five": 17,
    "six": 18,
    "seven": 19,
    "eight": 20,
    "nine": 21,
}

def GenerateConfusionMatrix(y_true, y_pred, num_classes):
    confusion_matrix = np.zeros((num_classes, num_classes))
    for i in range(len(y_true)):
        confusion_matrix[y_true[i], y_pred[i]] += 1
    return confusion_matrix

def VisualizeConfusionMatrix(y_true, y_pred, class_dict=label_dict, save_path=None):
    labels = class_dict.keys()
    confusion_matrix = GenerateConfusionMatrix(y_true, y_pred, len(labels))
    
    fig, ax = plt.subplots(figsize=(11, 9))
    cax = ax.matshow(confusion_matrix, cmap="coolwarm")
    fig.colorbar(cax)

    tick_positions = range(len(labels))
    ax.set_xticks(tick_positions)
    ax.set_yticks(tick_positions)
    ax.set_xticklabels(list(labels), rotation=90)
    ax.set_yticklabels(list(labels)) 

    for i in range(len(labels)):
        for j in range(len(labels)):
            ax.text(j, i, int(confusion_matrix[i, j]), ha="center", va="center", color="black")

    plt.xlabel("Predicted")
    plt.ylabel("True")

    if save_path:
        plt.savefig(Path(save_path))