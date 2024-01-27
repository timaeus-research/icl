from typing import List, Literal, Tuple, Union

import numpy as np
import seaborn as sns
from matplotlib import patches as mpatches

PRIMARY, SECONDARY, TERTIARY = sns.color_palette()[:3]
BBLUE, BORANGE, BGREEN, BRED = sns.color_palette('bright')[:4]

TransitionType = Literal["A", "B", "Other"]
Transition = Union[Tuple[int, int, str, TransitionType], Tuple[int, int, str]]


def gen_transition_colors(types: List[TransitionType]):
    """Generates a palette for transition colors. Orange-flavored for Type A. Blue-flavored for Type B."""
    num_type_a = sum([t == "A" for t in types])
    num_type_b = sum([t == "B" for t in types])
    num_other = sum([t == "Other" for t in types])

    type_a_palette = sns.color_palette("Oranges_r", num_type_a)
    type_b_palette = sns.color_palette("Blues_r", num_type_b)
    other_palette = sns.color_palette("Greys_r", num_other)

    palette = []

    for t in types:
        if t == "A":
            palette.append(type_a_palette.pop())
        elif t == "B":
            palette.append(type_b_palette.pop())
        else:
            palette.append(other_palette.pop())

    return palette


def get_transition_type(transition: Transition) -> TransitionType:
    if len(transition) == 4:
        return transition[-1]
    
    if "A" in transition[-1]:
        return "A"
    
    if "B" in transition[-1]:
        return "B"
    
    return "Other"



def plot_transitions(axes, transitions, max_step=np.inf, xlim=True, alpha=0.2, colors=None, labels=False):
    if colors is None:
        types = [get_transition_type(m) for m in transitions]
        colors = gen_transition_colors(types)

    min_step = min([t[0] for t in transitions])
    max_step = min(max_step, max([t[1] for t in transitions]))

    if not isinstance(axes, np.ndarray):
        axes = np.array([axes])

    for ax in axes.flatten():
        # Labels
        ax.set_xlabel("Training step $t$")
        ax.set_xscale("log")

        for color, (start, end, label) in zip(colors, transitions):

            if start > max_step:
                continue

            ax.axvspan(start, min(end, max_step), alpha=alpha, label=label, color=color)

        # Restrict x-axis
        if xlim:
            if isinstance(xlim, tuple):
                ax.set_xlim(*xlim)
            else:
                ax.set_xlim(min_step, max_step)

        ymin, ymax = ax.get_ylim()

        # Plot stage labels in the backgorund
        if labels:
            for i, (start, end, _) in enumerate(transitions):
                start = max(start, ymin)
                end = min(end, ymax)
                ax.text(np.exp(np.log((start+1) * (end+1))/ 2), (ymin + ymax) / 2, f"$\mathbf{{{i + 1}}}$", fontsize=16, ha='center', va='center', color=colors[i], alpha=1, zorder=-1000)

    # Add transition legend
    patch_list = []

    for color, (start, end, label) in zip(colors, transitions):
        data_key = mpatches.Patch(color=color, alpha=0.2, label=label)
        patch_list.append(data_key)

    return patch_list