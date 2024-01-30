import colorsys
from typing import List, Literal, Tuple, Union

import plotly.express as px
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
        colors = gen_LR_TRANSITION_COLORS(types)

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


def increase_saturation(rgb, saturation_factor):
    # Convert RGB to HSV
    hsv = colorsys.rgb_to_hsv(*rgb)
    
    # Increase saturation by the given factor, making sure it stays in [0, 1]
    new_s = min(max(hsv[1] * saturation_factor, 0), 1)
    
    # Convert back to RGB
    new_rgb = colorsys.hsv_to_rgb(hsv[0], new_s, hsv[2])
    return new_rgb


def increase_contrast(rgb, contrast_factor):
    # Midpoint
    midpoint = 128.0 / 255
    
    # Increase contrast
    new_rgb = [(0.5 + contrast_factor * (component - 0.5)) for component in rgb]
    
    # Clip to the range [0, 1]
    new_rgb = [min(max(component, 0), 1) for component in new_rgb]
    return new_rgb


def decrease_brightness(rgb, brightness_factor):
    # Convert RGB to HSV
    hsv = colorsys.rgb_to_hsv(*rgb)
    
    # Decrease brightness by the given factor, making sure it stays in [0, 1]
    new_v = min(max(hsv[2] * brightness_factor, 0), 1)
    
    # Convert back to RGB
    new_rgb = colorsys.hsv_to_rgb(hsv[0], hsv[1], new_v)
    return new_rgb


def rainbow(n):
    return px.colors.sample_colorscale(
        colorscale='rainbow',
        samplepoints=n,
        low=0.0,
        high=1.0,
        colortype="rgba",
    )


# LR_TRANSITION_TYPES = ['A', 'A', "A", "B", "B", "Other"]
LR_TRANSITION_TYPES = ['A', 'A', "B", "B", "Other"]
LR_TRANSITION_COLORS = gen_transition_colors(LR_TRANSITION_TYPES)

# del LR_TRANSITION_COLORS[2], LR_TRANSITION_TYPES[2]

LR_TRANSITION_COLORS[2] = decrease_brightness(LR_TRANSITION_COLORS[2], .9)
LR_TRANSITION_COLORS[2] = increase_saturation(LR_TRANSITION_COLORS[2], 2)