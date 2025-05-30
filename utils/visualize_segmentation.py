import sys

sys.path.append("..")
sys.path.append(".")

import numpy as np
from PIL import Image, ImageDraw, ImageFont
import cv2
import matplotlib.pyplot as plt
from matplotlib.patches import Patch



def show_mask(mask, ax, default_color, framealpha=0.5, random_color=False):
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([framealpha])], axis=0)
    else:
        color = np.concatenate([default_color, np.array([framealpha])], axis=0)
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)
    return color



def render_segmentation(class_names, colormap, seg_pred, image, show_img=True, font_size=15, framealpha=0.5):

    img_arr = np.array(image)
    visual_classes = set(seg_pred.ravel())

    if show_img:
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))  # Two subplots: original and rendered
        axs[0].imshow(img_arr)
        axs[0].axis('off')
        # Show segmentation overlay on the right
        axs[1].imshow(img_arr)
    else:
        plt.figure(figsize=(6, 4))
        plt.imshow(img_arr)

    patches = []  # To store legend patches
    for i, vis_class in enumerate(visual_classes):
        bool_mask = (seg_pred == vis_class)
        if bool_mask.sum() == 0:
            continue
        cls_color = [c / 255 for c in colormap[i]]
        color = show_mask(bool_mask, plt.gca(), default_color=cls_color, framealpha=framealpha)
        patch = Patch(color=color, label=f"{i} {class_names[vis_class]}")
        patches.append(patch)

    if show_img:
        axs[1].legend(handles=patches, loc='best', borderaxespad=0.1, fontsize=font_size, framealpha=framealpha)
        axs[1].axis('off')
    else:
        plt.legend(handles=patches, loc='best', borderaxespad=0.1, fontsize=font_size, framealpha=framealpha)

    plt.margins(tight=True)
    plt.axis('off')
    plt.tight_layout()
    plt.show()
    return