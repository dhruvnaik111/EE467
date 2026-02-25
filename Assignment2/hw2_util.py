import math

import numpy as np
import torch
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec
from torch import nn

class Transpose(nn.Module):
    def __init__(self, dim0, dim1):
        super().__init__()
        
        self.dim0 = dim0
        self.dim1 = dim1

    def forward(self, x):
        return x.transpose(self.dim0, self.dim1)

    def extra_repr(self):
        return f"dim0={self.dim0}, dim1={self.dim1}"

def get_torch_device():
    # Use CUDA (Nvidia/AMD GPU) if available
    if torch.cuda.is_available():
        device = "cuda"
    # Use MPS (Apple Silicon GPU) if available
    elif torch.backends.mps.is_available():
        device = "mps"
    # Fallback to CPU
    else:
        device = "cpu"

    return torch.device(device)

def flatten(it):
    """ Flatten iterable of iterables into an iterable of values. """
    return (value for sub_it in it for value in sub_it)

def group_every(iterable, n_elem=1):
    """ Group every N elements into a list and yield that list. """
    group = []

    for value in iterable:
        # Add value to group
        group.append(value)
        # Yield and re-create group if it is full
        if len(group)>=n_elem:
            yield group
            group = []

    # Yield the last group if it's not empty
    if group:
        yield group

def unzip(iterable):
    """ Unzip iterable of tuples into tuple of lists. """
    unzip_lists = None
    n_lists = None
    
    for values in iterable:
        n_values = len(values)
        # Create lists from first tuple
        if unzip_lists is None:
            unzip_lists = tuple(([value] for value in values))
            n_lists = n_values
        else:
            # Check tuple length
            if n_values!=n_lists:
                raise ValueError(f"Expect tuple of {n_lists} values, got {n_values}")
            # Append values to lists
            for value, unzip_list in zip(values, unzip_lists):
                unzip_list.append(value)

    return unzip_lists

def print_images(images, n_rows=1, n_cols=None, fig_size=(20, 5), color_map="Greys",
    texts=None, font_size=20, text_center=(0.5, -0.2)):
    """ Print multiple images in a single Matplotlib figure (with optional text description). """
    n_images = len(images)
    n_cols = n_cols or math.ceil(n_images/n_rows)
    # Make grid plot for all images
    fig = plt.figure(figsize=fig_size)
    grid_spec = GridSpec(ncols=n_cols, nrows=n_rows)

    # Print each image
    for row in range(n_rows):
        for col in range(n_cols):
            i = row*n_cols+col
            # No more image to draw
            if i>=n_images:
                break

            # Get subplot for image
            ax = fig.add_subplot(grid_spec[row, col])
            ax.axis("off")
            # Plot image
            image = np.asarray(images[i])
            ax.imshow(image, cmap=color_map)
            # Plot text if available
            if texts:
                ax.text(
                    text_center[0], text_center[1], texts[i],
                    fontsize=font_size, transform=ax.transAxes,
                    ha="center", va="center"
                )
    
    # Apply tight layout
    fig.tight_layout()
    # Print figure
    print(fig)
