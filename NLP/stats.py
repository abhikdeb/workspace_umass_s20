import numpy as np
from matplotlib import pyplot as plt
import os

epochs = 50
# data_dir = 'D:/UMass/Spring20/685/Project/work/Hasoc_BiLSTM_' + str(epochs) + '/'
# data_dir = 'D:/UMass/Spring20/685/Project/work/Reddit_BiLSTM_' + str(epochs) + '/'
data_dir = 'D:/UMass/Spring20/685/Project/work/Gab_BiLSTM_' + str(epochs) + '/'


def main():
    file_list = os.listdir(data_dir)
    train_losses = []
    train_accuracies = []
    val_accuracies = []
    test_accuracies = []
    legend = {}

    for file in file_list:
        key = file.split("_", 2)[2]
        if key in legend.keys():
            legend[key] += 1
        else:
            legend[key] = 1

    print(legend.keys())
    dct = list(legend.keys())

    for key in legend.keys():
        train_loss = np.load(data_dir + 'train_loss_' + key, allow_pickle=True)
        train_losses.append(train_loss)
        train_acc = np.load(data_dir + 'train_acc_' + key, allow_pickle=True)
        train_accuracies.append(train_acc)
        val_acc = np.load(data_dir + 'val_acc_' + key, allow_pickle=True)
        val_accuracies.append(val_acc)
        test_acc = np.load(data_dir + 'test_acc_' + key, allow_pickle=True)
        test_accuracies.append(test_acc)

    fig, axs = plt.subplots(2, 2)
    for i in range(len(legend.keys())):
        axs[0, 0].plot(np.arange(epochs), train_accuracies[i])
    axs[0, 0].set_title('Training F1')
    for i in range(len(legend.keys())):
        axs[0, 1].plot(np.arange(epochs), test_accuracies[i])
        print(dct[i], ' : ', np.max(test_accuracies[i]))
    axs[0, 1].set_title('Test F1')
    for i in range(len(legend.keys())):
        axs[1, 0].plot(np.arange(epochs), val_accuracies[i])
    axs[1, 0].set_title('Validation F1')
    for i in range(len(legend.keys())):
        axs[1, 1].plot(np.arange(epochs), train_losses[i])
    axs[1, 1].set_title('Training Loss')

    plt.show()

    exit(0)
    return


if __name__ == "__main__":
    main()
