# Implementation by: Jack Shilton
# https://jack-development.github.io/

# Reference:
# Kha, N., & Nakagawa, M. (2016). Enhanced Character Segmentation for Format-Free Japanese Text Recognition.
# In Proceedings of the 15th International Conference on Frontiers in Handwriting Recognition (pp. 138-143).
# https://doi.org/10.1109/ICFHR.2016.0037

import numpy as np
import cv2
from PIL import Image
import matplotlib.pyplot as plt
from swt.swt import open_grayscale, get_edges, get_gradients, apply_swt

show_images = True

def get_average_height(img_in):
    # Format and normalise image
    img_in = np.squeeze(img_in)
    img_in = ((img_in / img_in.max()) * 255).astype(np.uint8)
    img = cv2.adaptiveThreshold(img_in,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,25)

    # Collect contours of all characters
    contours = cv2.findContours(img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]

    # Get bounding box of each contour
    heights = []
    for contour in contours:
        h = cv2.boundingRect(contour)[3]
        heights.append(h)

    # Return average height of every box
    if len(heights) > 0:
        return np.mean(heights)
    return 0

def get_average_width(img, alpha=0.75):
    # Assume AveWid = α × H_tl
    return get_average_height(img) * alpha

def get_strokes(self, xywh):
    x, y, w, h = xywh
    stroke_widths = np.array([[np.Infinity, np.Infinity]])
    for i in range(y, y + h):
        for j in range(x, x + w):
            if self.canny_img[i, j] != 0:
                gradX = self.gradsX[i, j]
                gradY = self.gradsY[i, j]

                prevX, prevY, prevX_opp, prevY_opp, step_size = i, j, i, j, 0

                if self.direction == "light":
                    go, go_opp = True, False
                elif self.direction == "dark":
                    go, go_opp = False, True
                else:
                    go, go_opp = True, True

                stroke_width = np.Infinity
                stroke_width_opp = np.Infinity
                while (go or go_opp) and (step_size < self.STEP_LIMIT):
                    step_size += 1

                    if go:
                        curX = int(np.floor(i + gradX * step_size))
                        curY = int(np.floor(j + gradY * step_size))
                        if (curX <= y or curY <= x or curX >= y + h or curY >= x + w):
                            go = False
                        if go and ((curX != prevX) or (curY != prevY)):
                            try:
                                if self.canny_img[curX, curY] != 0:
                                    if np.arccos(gradX * -self.gradsX[curX, curY] + \
                                                gradY * -self.gradsY[curX, curY]) < np.pi / 2.0:
                                        stroke_width = int(np.sqrt((curX - i) ** 2 + \
                                                                    (curY - j) ** 2))
                                        go = False
                            except IndexError:
                                go = False

                            prevX = curX
                            prevY = curY

                    if go_opp:
                        curX_opp = int(np.floor(i - gradX * step_size))
                        curY_opp = int(np.floor(j - gradY * step_size))
                        if (curX_opp <= y or \
                            curY_opp <= x or \
                            curX_opp >= y + h or \
                            curY_opp >= x + w):
                            go_opp = False
                        if go_opp and ((curX_opp != prevX_opp) or (curY_opp != prevY_opp)):
                            try:
                                if self.canny_img[curX_opp, curY_opp] != 0:
                                    if np.arccos(gradX * -self.gradsX[curX_opp, curY_opp] + \
                                            gradY * -self.gradsY[curX_opp, curY_opp]) < np.pi/2.0:
                                        stroke_width_opp = int(np.sqrt((curX_opp - i) ** 2 + \
                                                                        (curY_opp - j) ** 2))
                                        go_opp = False

                            except IndexError:
                                go_opp = False

                            prevX_opp = curX_opp
                            prevY_opp = curY_opp

                stroke_widths = np.append(stroke_widths, [(stroke_width, stroke_width_opp)],
                                            axis=0)

    stroke_widths_opp = np.delete(stroke_widths[:, 1],
                                    np.where(stroke_widths[:, 1] == np.Infinity))
    stroke_widths = np.delete(stroke_widths[:, 0],
                                np.where(stroke_widths[:, 0] == np.Infinity))
    return stroke_widths, stroke_widths_opp

def average_SWT(img, bright_on_dark=False):
    # SWT Implementation by Markus Mayer
    # https://github.com/sunsided
    # GitHub - Stroke Width Transform
    # https://github.com/sunsided/stroke-width-transform

    # Based on:
    # Epshtein, B., Ofek, E., & Wexler, Y. (2010). Detecting Text in Natural Scenes with Stroke Width Transform.
    # In Proceedings of the IEEE - Institute of Electrical and Electronics Engineers (June).
    # https://doi.org/10.1109/CVPR.2010.5540041

    # Initial formatting of image
    img_in = np.squeeze(img)
    img_in = ((img_in / img_in.max()) * 255).astype(np.uint8)
    binary_img = cv2.adaptiveThreshold(img_in,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,21,25)
    binary_img = cv2.bitwise_not(binary_img)

    # Collect swt, based on Markus Mayer's implementation
    edges = get_edges(img)
    gradients = get_gradients(img)
    swt = apply_swt(img, edges, gradients, not bright_on_dark)

    if show_images:
        cv2.imshow("swt", swt)
        cv2.imshow("binary", binary_img)
        cv2.waitKey(0)

    # Estimating the average stroke width (AW)
    stroke_sum = np.sum(swt * binary_img)
    binary_sum = np.sum(binary_img)

    if binary_sum > 0:
        AW = stroke_sum / binary_sum
        # TODO: Improve this algorithm so this doubling is not required for good performance
        return AW * 2
    else:
        return 0



# Added threshold that can allow for better groupings of non-overlapping characters
# Threshold - Minimum number of consecutive results required to make a group
# MinSplit - Minimum percentage of the bound for an AW point to be valid for a group
def get_character_splits(smeared_img, AveWid, AW, threshold=1, min_split=0.2):
    # Get projection of image along x-axis (horizontal projection)
    xProj = np.sum(smeared_img, axis=0)
    # Collect points where the projection is 0
    zero_points = np.where(xProj <= 0.01)[0]

    # Group zero points into consecutive groups, then return the mean
    mean_points = np.array([])
    if(len(zero_points) > 0):
        group = [zero_points[0]]
        consecutive_count = 0
        for i in range(1, len(zero_points)):
            if (zero_points[i] == zero_points[i - 1] + 1) and i is not len(zero_points) - 1:
                group.append(zero_points[i])
                consecutive_count += 1
            else:
                if (consecutive_count >= threshold):
                    mean_points = np.append(mean_points, np.mean(group))
                    group = [zero_points[i]]
                consecutive_count = 0

    # If no groups found, consider the whole image as a group
    if not mean_points.any():
        mean_points = [0, smeared_img.shape[1] - 1]

    # Find the width of every group
    diff = np.diff(mean_points)
    # Find oversized groups (Group with width larger than AveWid)
    large_groups = diff > AveWid

    # Find points within oversized groups less than AW
    aw_points = []
    if large_groups.any():
        group_bounds = []
        for i in range(len(large_groups)):
            if large_groups[i]:
                start = mean_points[i]
                end = mean_points[i + 1]
                group_bounds.append((int(start), int(end)))

        for bound in group_bounds:
            start, end = bound
            # Filter out points that would create a group that is too small
            size = int((end - start) * min_split)
            start += size
            end -= size
            temp_points = np.where(xProj[start:end + 1] <= AW)[0]
            if len(temp_points) == 0:
                continue

            aw_points.append(start + temp_points.mean())

    if show_images:
        plt.figure(figsize=(10, 4))
        plt.bar(np.arange(len(xProj)), xProj, width=1.0)
        plt.title('X-Axis Projection of the Image (Bar Graph)')
        plt.xlabel('X Pixels')
        plt.ylabel('Summed Intensity')

        graph_top = np.max(xProj) * 1.1
        plt.ylim(0, graph_top)
        plt.vlines(x=mean_points, ymin=0, ymax=graph_top, color='red', linestyle='--', label=f'Zero point')
        if (len(aw_points) > 1):
            plt.vlines(x=aw_points, ymin=0, ymax=graph_top, color='blue', linestyle='--', label=f'AW point')
        elif len(aw_points) == 1:
            plt.axvline(x=aw_points[0], color='blue', linestyle='--', label=f'AW point')

        plt.grid(True)
        plt.legend()
        plt.show()

    return mean_points, aw_points

def vertical_smearing(img, height=15):
    # Create vertical kernel
    # TODO: Select automatic kernel size from image shape
    vertical_kernel = np.ones((height, 1), np.uint8)
    return cv2.morphologyEx(img, cv2.MORPH_CLOSE, vertical_kernel)

def coarse_segmentation(raw_img, flip_binary=False):
    # Modified threshold value for better results
    AveWid = get_average_width(raw_img, 0.9)
    AW = average_SWT(raw_img)

    # Edit raw image to fit expected format
    raw_img = 1 - raw_img
    raw_img[raw_img < 0.01] = 0

    # Enhance characters vertically
    smeared_img = vertical_smearing(raw_img, 2)

    if show_images:
        cv2.imshow("smeared", smeared_img)
        cv2.waitKey(0)

    # Split characters, where split_points are 0-based splits and aw_points are from oversized segments
    split_points, aw_points = get_character_splits(smeared_img, AveWid, AW, 5, 0.4)

    if show_images:
        plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))

        for split in split_points:
            plt.axvline(x=split, color='red', linestyle='--', linewidth=1)
        for aw in aw_points:
            plt.axvline(x=aw, color='blue', linestyle='--', linewidth=1)
        plt.show()

    return split_points, aw_points

def remove_space(img, space=10):
    # Change image to correct format
    img = 1 - img
    img[img < 0.01] = 0

    # Get projection of image along x-axis (horizontal projection)
    xProj = np.sum(img, axis=0)
    # Collect points where the projection is 0
    zero_points = np.where(xProj <= 0.01)[0]

    # Collect all groups that cross 0
    groups = []
    if(len(zero_points) > 0):
        group = [zero_points[0]]
        for i in range(1, len(zero_points)):
            if (zero_points[i] == zero_points[i - 1] + 1) and i is not len(zero_points) - 1:
                group.append(zero_points[i])
            else:
                groups.append(group)
                group = [zero_points[i]]

    # Create a mask for which columns to keep
    column_mask = np.ones(img.shape[1], dtype=bool)
    for group in groups:
        # Find groups that are over the width limit
        if (len(group) > space):
            # Set every column over the limit to be removed
            column_mask[group[space:]] = False

    # Remove specified columns
    img = img[:, column_mask]

    # Return image to original format
    img = 1 - img
    return img

if __name__ == '__main__':
    flip_binary = True
    input_file = 'input.png'
    raw_img = open_grayscale(input_file)
    raw_img = remove_space(raw_img)
    split_points, aw_points = coarse_segmentation(raw_img, flip_binary)