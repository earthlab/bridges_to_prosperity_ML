import os
import matplotlib.pyplot as plt
import torch


def ratio_box_plot(indir, acc_key, outpath=None):
    plotting_data = {}
    ratios = []
    for dir in os.listdir(indir):
        if 'ratio' in dir and os.path.isdir(os.path.join(indir, dir)):
            ratio = float(dir.split('_')[-1])
            ratios.append(ratio)
            plotting_data[ratio] = {}
            for file in os.listdir(os.path.join(indir, dir)):
                if 'best.tar' in file or file.startswith('.') or '.tar' not in file:
                    continue
                epoch = file.split('chkpt')[1].split('.tar')[0]
                model = torch.load(os.path.join(indir, dir, file))
                print(model.keys())
                plotting_data[ratio][epoch] = model[acc_key]
    ratios = sorted(ratios)

    accuracies = []
    for ratio in ratios:
        accuracies.append([float(plotting_data[ratio][epoch]) for epoch in sorted(plotting_data[ratio].keys())])

    fig, ax = plt.subplots(figsize=(10, 7))
    bp = ax.boxplot(accuracies, patch_artist=True, notch=True)

    # Set the color of the boxes
    box_colors = ['#1f77b4'] * len(accuracies)
    for patch, color in zip(bp['boxes'], box_colors):
        patch.set_facecolor(color)

    # Set the color of the whiskers and caps
    whisker_colors = ['#7570b3'] * len(accuracies)
    for whisker, color in zip(bp['whiskers'], whisker_colors):
        whisker.set_color(color)
    for cap, color in zip(bp['caps'], whisker_colors):
        cap.set_color(color)

    # Set the color of the medians
    median_colors = ['#ff7f0e'] * len(accuracies)
    for median, color in zip(bp['medians'], median_colors):
        median.set_color(color)

    # Set the color of the fliers
    flier_colors = ['#2ca02c'] * len(accuracies)
    for flier, color in zip(bp['fliers'], flier_colors):
        flier.set(marker='o', color=color, alpha=0.5)

    # Set the x-axis labels
    ax.set_xticklabels([f"{ratio:.1f}" for ratio in ratios])
    ax.set_xlabel('no bridge / bridge')
    ax.set_ylabel(acc_key)

    # Add a title
    ax.set_title(os.path.basename(indir))

    # Save or show the plot
    if outpath is not None:
        plt.savefig(outpath)
    else:
        plt.show()
    plt.clf()


def line_plot(indir, acc_key, outpath=None):
    plotting_data = {}
    ratios = []
    for dir in os.listdir(indir):
        if 'ratio' in dir and os.path.isdir(os.path.join(indir, dir)):
            ratio = float(dir.split('_')[-1])
            ratios.append(ratio)
            plotting_data[ratio] = []
            for file in os.listdir(os.path.join(indir, dir)):
                if 'best.tar' in file or file.startswith('.') or '.tar' not in file:
                    continue
                epoch = int(file.split('chkpt')[1].split('.tar')[0])
                model = torch.load(os.path.join(indir, dir, file))
                print(model.keys())
                plotting_data[ratio].append((epoch, model[acc_key]))
    ratios = sorted(ratios)

    linestyles = [
        'solid', 'dashed', 'dashdot', 'dotted', 'solid', 'dashed', 'dashdot', 'dotted'
        ]

    for i, ratio in enumerate(ratios):
        x = []
        y = []
        for epoch in sorted(plotting_data[ratio], key=lambda x: x[0]):
            x.append(epoch[0])
            y.append(epoch[1])

        plt.plot(x, y, label=f'{ratio}', linestyle=linestyles[i])

    plt.title(os.path.basename(indir))
    plt.ylabel(acc_key)
    plt.xlabel('epoch')
    plt.legend(title='no bridge / bridge')
    # Save or show the plot
    if outpath is not None:
        plt.savefig(outpath)
    else:
        plt.show()
    plt.clf()
