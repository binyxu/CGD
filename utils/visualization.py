import torch
import json
import numpy as np
import matplotlib.pyplot as plt
import torch.nn.functional as F
from matplotlib.ticker import NullFormatter
from matplotlib.patches import Patch
import matplotlib.gridspec as gridspec
from utils.models import initialize_kfold_classifier

def calculate_accuracy(predictions, labels):
    correct = (predictions == labels).sum().item()
    total = labels.size(0)
    return correct / total if total > 0 else 0

def main_visualization(logits_kfold, logits_clip, poisoned_indices, noise_indices, poison_labels, clean_labels, base_path):
    """
    Main visualization function integrating the revised visualization settings.
    """
    is_poisoned = torch.zeros(len(poison_labels), dtype=bool)
    is_poisoned[poisoned_indices] = True
    if noise_indices is not None:
        is_noise = torch.zeros(len(poison_labels), dtype=bool)
        is_noise[noise_indices] = True
    else:
        is_noise = torch.zeros(len(poison_labels), dtype=bool)
    is_poisoned_np = is_poisoned.detach().cpu().numpy()
    is_noise_np = is_noise.detach().cpu().numpy()

    predictions_kfold = torch.argmax(logits_kfold, dim=1)
    predictions_clip = torch.argmax(logits_clip, dim=1)

    accuracy_kfold = calculate_accuracy(predictions_kfold, poison_labels)
    accuracy_clip = calculate_accuracy(predictions_clip, poison_labels)
    if poisoned_indices.numel() > 0:
        accuracy_kfold_attack = calculate_accuracy(predictions_kfold[poisoned_indices], poison_labels[poisoned_indices])
    else:
        accuracy_kfold_attack = 0

    print(f"Accuracy of k-fold method: {accuracy_kfold * 100:.2f}%")
    print(f"Accuracy of k-fold attack method: {accuracy_kfold_attack * 100:.2f}%")
    print(f"Accuracy of CLIP method: {accuracy_clip * 100:.2f}%")

    print(f"Using CLIP accuracy {accuracy_clip} as threshold line to filter clean data!!")

    # 1. Compute Softmax Probabilities
    probs_kfold = F.softmax(logits_kfold, dim=1)
    probs_clip = F.softmax(logits_clip, dim=1)

    # Ensure poison_labels and clean_labels are tensors of shape [N]
    poison_labels = torch.tensor(poison_labels, dtype=torch.long)
    clean_labels = torch.tensor(clean_labels, dtype=torch.long)

    # 2. Compute Negative Log-Likelihoods
    NLL_kfold = -torch.log(probs_kfold[torch.arange(len(poison_labels)), poison_labels])
    NLL_clip = -torch.log(probs_clip[torch.arange(len(poison_labels)), poison_labels])

    # 3. Convert Data for Plotting
    entropy_clip_np = NLL_clip.detach().cpu().numpy()
    entropy_kfold_np = NLL_kfold.detach().cpu().numpy()

    N = len(entropy_clip_np)

    entropy_clip_ranks = np.argsort(np.argsort(entropy_clip_np)) + 1
    entropy_kfold_ranks = np.argsort(np.argsort(entropy_kfold_np)) + 1

    # Normalize ranks to range [0, 1]
    entropy_clip_norm = entropy_clip_ranks / N
    entropy_kfold_norm = entropy_kfold_ranks / N

    # 4. Randomly Select Points
    sample_size = 10000
    total_points = len(entropy_clip_norm)

    if total_points > sample_size:
        # Randomly select unique indices without replacement
        unique_indices = np.random.choice(total_points, size=sample_size, replace=False)
    else:
        # If total points are less than or equal to the sample size, use all indices
        unique_indices = np.arange(total_points)

    # Subset the data
    entropy_clip_subset = entropy_clip_norm[unique_indices]
    entropy_kfold_subset = entropy_kfold_norm[unique_indices]
    is_poisoned_subset = is_poisoned_np[unique_indices]
    is_noise_subset = is_noise_np[unique_indices]
    poison_labels_subset = poison_labels[unique_indices].detach().cpu().numpy()
    clean_labels_subset = clean_labels[unique_indices].detach().cpu().numpy()

    # 5. Identify Special Poisoned Samples (poison_labels == clean_labels)
    is_special_poisoned = poison_labels_subset == clean_labels_subset

    # Create custom grid layout with adjusted ratios
    fig = plt.figure(figsize=(3.5, 3.5))
    gs = gridspec.GridSpec(
        6, 6,
        figure=fig,
        hspace=0.05,
        wspace=0.05,
        height_ratios=[0.4, 0.1, 0.1, 0.1, 0.1, 2.4],
        width_ratios=[2.4, 0.1, 0.1, 0.1, 0.1, 0.4]
    )

    # Main scatter plot
    ax_main = fig.add_subplot(gs[1:, :-1])
    ax_xhist = fig.add_subplot(gs[0, :-1])
    ax_yhist = fig.add_subplot(gs[1:, -1])

    # Set the limits of histograms to match main plot
    ax_xhist.set_xlim(0, 1)
    ax_yhist.set_ylim(0, 1)

    # Scatter plot with no edges
    scatter_clean = ax_main.scatter(
        entropy_clip_subset[~is_poisoned_subset],
        entropy_kfold_subset[~is_poisoned_subset],
        c='blue',
        alpha=0.1,
        s=3,
        label='clean',
        edgecolors='none'  # Remove marker edges
    )
    scatter_poisoned = ax_main.scatter(
        entropy_clip_subset[is_poisoned_subset & ~is_special_poisoned & ~is_noise_subset],
        entropy_kfold_subset[is_poisoned_subset & ~is_special_poisoned & ~is_noise_subset],
        c='red',
        alpha=0.5,
        s=3,
        label='poison',
        edgecolors='none'  # Remove marker edges
    )
    scatter_clean_label_poisoned = ax_main.scatter(
        entropy_clip_subset[is_poisoned_subset & is_special_poisoned & ~is_noise_subset],
        entropy_kfold_subset[is_poisoned_subset & is_special_poisoned & ~is_noise_subset],
        c='purple',
        alpha=0.5,
        s=3,
        label='lc-poison',
        edgecolors='none'  # Remove marker edges
    )
    scatter_noise = ax_main.scatter(
        entropy_clip_subset[is_poisoned_subset & is_noise_subset],
        entropy_kfold_subset[is_poisoned_subset & is_noise_subset],
        c='green',
        alpha=0.5,
        s=3,
        label='noise',
        edgecolors='none'  # Remove marker edges
    )

    ax_main.set_xlabel('Entropy of Weak Model')
    ax_main.set_ylabel('Entropy of Suspicious Model')
    ax_main.set_xlim(0, 1)
    ax_main.set_ylim(0, 1)
    ax_main.grid(True, linestyle='--')  # Use dashed lines for the grid

    # Keep ticks and labels on the main plot
    plt.setp(ax_main.get_xticklabels(), visible=True)
    plt.setp(ax_main.get_yticklabels(), visible=True)
    ax_main.tick_params(axis='both', which='both', length=5)

    # Remove ticks and labels from histograms
    ax_xhist.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)
    ax_yhist.tick_params(axis='both', which='both', bottom=False, top=False, labelbottom=False, left=False, labelleft=False)

    # Remove numerical labels from histogram axes
    ax_xhist.xaxis.set_major_formatter(NullFormatter())
    ax_xhist.yaxis.set_major_formatter(NullFormatter())
    ax_yhist.xaxis.set_major_formatter(NullFormatter())
    ax_yhist.yaxis.set_major_formatter(NullFormatter())

    # Extend grid lines into histograms
    # For ax_xhist, draw vertical grid lines matching ax_main
    for gridline in ax_main.get_xgridlines():
        ax_xhist.axvline(gridline.get_xdata()[0], color='gray', linestyle='--', linewidth=0.5)
    # For ax_yhist, draw horizontal grid lines matching ax_main
    for gridline in ax_main.get_ygridlines():
        ax_yhist.axhline(gridline.get_ydata()[0], color='gray', linestyle='--', linewidth=0.5)

    # Histograms with matched colors
    bins = np.arange(0, 1.02, 0.02)
    ax_xhist.hist(entropy_clip_subset[~is_poisoned_subset], bins=bins, color='blue', alpha=0.3)
    ax_xhist.hist(entropy_clip_subset[is_poisoned_subset & ~is_special_poisoned & ~is_noise_subset], bins=bins, color='red', alpha=0.5)
    ax_xhist.hist(entropy_clip_subset[is_poisoned_subset & is_special_poisoned & ~is_noise_subset], bins=bins, color='purple', alpha=0.5)
    ax_xhist.hist(entropy_clip_subset[is_poisoned_subset & is_noise_subset], bins=bins, color='green', alpha=0.5)

    ax_yhist.hist(entropy_kfold_subset[~is_poisoned_subset], bins=bins, color='blue', alpha=0.3, orientation='horizontal')
    ax_yhist.hist(entropy_kfold_subset[is_poisoned_subset & ~is_special_poisoned & ~is_noise_subset], bins=bins, color='red', alpha=0.5, orientation='horizontal')
    ax_yhist.hist(entropy_kfold_subset[is_poisoned_subset & is_special_poisoned & ~is_noise_subset], bins=bins, color='purple', alpha=0.5, orientation='horizontal')
    ax_yhist.hist(entropy_kfold_subset[is_poisoned_subset & is_noise_subset], bins=bins, color='green', alpha=0.5, orientation='horizontal')

    # Adjust layout
    plt.tight_layout()
    fig.subplots_adjust(left=0.2)
    fig.subplots_adjust(bottom=0.2)

    # Create custom legend using colored patches
    legend_elements = [
        Patch(facecolor='blue', edgecolor='blue', alpha=0.5, label='clean'),
        Patch(facecolor='red', edgecolor='red', alpha=0.5, label='poison'),
        Patch(facecolor='purple', edgecolor='purple', alpha=0.5, label='lc-poison'),
        Patch(facecolor='green', edgecolor='green', alpha=0.5, label='noise')
    ]

    # Add legend to the upper-left corner with reduced transparency
    # ax_main.legend(handles=legend_elements, loc='upper left', framealpha=0.7)

    plt.savefig(f'{base_path}/vslz_with_histograms_colored.png', dpi=600)
    plt.close(fig)

    # Save entropy scores if needed
    entropy_score = {i: [entropy_clip_norm[i], entropy_kfold_norm[i]] for i in range(len(entropy_clip_norm))}
    with open(f'{base_path}/entropy_score.json', 'w') as json_file:
        json.dump(entropy_score, json_file)

    return

def process_data(logits_kfold, logits_clip, poison_labels, clean_labels, poisoned_indices, noise_indices, out_dir):
    main_visualization(
        logits_kfold=logits_kfold,
        logits_clip=logits_clip,
        poisoned_indices=poisoned_indices,
        noise_indices=noise_indices,
        poison_labels=poison_labels,
        clean_labels=clean_labels,
        base_path=out_dir
    )

    N = len(poison_labels)
    entropy_clip_np = -torch.log(F.softmax(logits_clip, dim=1)[torch.arange(len(poison_labels)), poison_labels]).detach().cpu().numpy()
    entropy_kfold_np = -torch.log(F.softmax(logits_kfold, dim=1)[torch.arange(len(poison_labels)), poison_labels]).detach().cpu().numpy()

    entropy_clip_ranks = np.argsort(np.argsort(entropy_clip_np)) + 1
    entropy_kfold_ranks = np.argsort(np.argsort(entropy_kfold_np)) + 1

    entropy_clip_norm = entropy_clip_ranks / N
    entropy_kfold_norm = entropy_kfold_ranks / N

    threshold_clip = 0.8
    threshold_kfold = 0.2

    clean_data = np.logical_and(entropy_clip_norm <= threshold_clip, entropy_kfold_norm >= threshold_kfold)
    clean_indices = np.nonzero(clean_data)[0]

    entropy_score = {i: [entropy_clip_norm[i], entropy_kfold_norm[i], i in poisoned_indices] for i in range(len(entropy_clip_norm))}
    with open(f'{out_dir}/entropy_score.json', 'w') as json_file:
        json.dump(entropy_score, json_file)

    values, counts = np.unique(poison_labels[clean_indices], return_counts=True)
    max_occurrence = counts.max()
    oversampled_indices = []
    for value, count in zip(values, counts):
        indices = np.where(poison_labels[clean_indices] == value)[0]
        if count == max_occurrence:
            oversampled_indices.extend([clean_indices[i] for i in indices])
        else:
            resampled_indices = np.random.choice([clean_indices[i] for i in indices], size=max_occurrence, replace=True)
            oversampled_indices.extend(resampled_indices)
    oversampled_indices = [int(x) for x in oversampled_indices]

    with open(f'{out_dir}/clean_indices.json', 'w') as json_file:
        json.dump(oversampled_indices, json_file)

    return oversampled_indices

def compute_logits(poison_dataset, num_classes, device):
    logits_kfold_list = []
    for run in range(3):
        print(f"Run {run + 1}/3:")
        kfold_classifier = initialize_kfold_classifier(num_classes, device)
        logits_kfold = kfold_classifier.predict(poison_dataset)
        logits_kfold_list.append(logits_kfold)

    logits_kfold_avg = torch.mean(torch.stack(logits_kfold_list), dim=0)
    return logits_kfold_avg
