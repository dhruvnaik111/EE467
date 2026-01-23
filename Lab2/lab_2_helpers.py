import math

import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec
import imutils
import cv2

import imutils
import cv2

def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size.
    
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image

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
            ax.imshow(images[i], cmap=color_map)
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