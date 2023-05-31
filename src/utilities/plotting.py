import os

import matplotlib.pyplot as plt
import torch
from tqdm.auto import tqdm
import json
from glob import glob
def fixJSONKeys(x):
    if isinstance(x, dict):
        d = {}
        for k,v in x.items():
            try: 
                d[float(k)] = fixJSONKeys(v) 
            except:
                d[k] = fixJSONKeys(v) 
        return d
    return x
"""
    From a directory that has saved models, creat a box plot
    indir: directory of a type of CNN with subdirectories corresponding to ratios
    acc_key: type of acc to be plotted options ('total_acc', 'bridge_acc', 'no_bridge_acc')
"""
def ratio_box_plot(indir, acc_key, outpath=None):
    accuracies = ('total_acc', 'bridge_acc', 'no_bridge_acc')
    save_file = os.path.join(indir, "accuracy_summary.json")
    print(f"Plotting {indir}, {acc_key}")
    if os.path.isfile(save_file):
        print(f"\tLoading summary file {save_file}")
        with open(save_file,'r') as f: 
            plotting_data = fixJSONKeys(json.load(f))
    else:
        print(f"\tSummary file DNE for {indir}\n\tMust load all models from tar...")
        plotting_data = {}
        plotting_data["ratios"] = set()
        for mfile in tqdm(glob(os.path.join(indir, "**", "*chkpt*.tar"), recursive=True)):
            dir, _ = os.path.split(mfile)
            _, ratio = os.path.split(dir)
            ratio = float(ratio.split('_')[-1])
            plotting_data["ratios"].add(ratio)
            if not (ratio in plotting_data):
                plotting_data[ratio] = {}
            epoch = mfile.split('chkpt')[1].split('.tar')[0]
            model = torch.load(mfile)
            plotting_data[ratio][epoch] = {}
            plotting_data[ratio][epoch]["file"] = mfile
            for acc in accuracies:
                plotting_data[ratio][epoch][acc] = float(model[acc])
        plotting_data["ratios"] = sorted(list(plotting_data["ratios"]))
        
        with open(save_file, "w") as fp:
            json.dump(plotting_data,fp) 

    accuracies = []
    for ratio in plotting_data["ratios"]:
        accuracies.append([plotting_data[ratio][epoch][acc_key] for epoch in sorted(plotting_data[ratio].keys())])

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
    ax.set_xticklabels([f"{ratio:.1f}" for ratio in plotting_data["ratios"]])
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
