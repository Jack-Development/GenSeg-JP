# Implementation by: Jack Shilton
# https://jack-development.github.io/

# Reference:
# Kha, N., & Nakagawa, M. (2016). Enhanced Character Segmentation for Format-Free Japanese Text Recognition.
# In Proceedings of the 15th International Conference on Frontiers in Handwriting Recognition (pp. 138-143).
# https://doi.org/10.1109/ICFHR.2016.0037

import numpy as np
import cv2
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import math
import heapq
from swt.swt import open_grayscale
from skimage.morphology import skeletonize
from scipy.ndimage import convolve
from scipy.spatial import Voronoi, voronoi_plot_2d
from itertools import combinations

show_images = True

global AveWid


def get_average_height(img_in):
    # Format and normalise image
    img_in = np.squeeze(img_in)
    img_in = ((img_in / img_in.max()) * 255).astype(np.uint8)
    img = cv2.adaptiveThreshold(img_in, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 21, 25)

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

# -----------------------------------------------------------------------------
# SWT Implementation by Azmi C. Özgen
# https://github.com/azmiozgen
# GitHub - text-detection
# https://github.com/azmiozgen/text-detection

# Based on:
# Epshtein, B., Ofek, E., & Wexler, Y. (2010). Detecting Text in Natural Scenes with Stroke Width Transform.
# In Proceedings of the IEEE - Institute of Electrical and Electronics Engineers (June).
# https://doi.org/10.1109/CVPR.2010.5540041


def apply_canny(img, sigma=0.33):
    v = np.median(img)
    lower = int(max(0, (1.0 - sigma) * v))
    upper = int(min(255, (1.0 + sigma) * v))
    return cv2.Canny(img, lower, upper)


def get_strokes(raw_img):
    # Maximum number of steps to along the gradient to find stroke width
    step_limit = 10

    # Convert the raw image to 8-bit grayscale
    gray_img = np.uint8(raw_img)
    canny_img = apply_canny(gray_img)

    # Compute the Sobel gradients
    sobelX = cv2.Sobel(gray_img, cv2.CV_64F, 1, 0, ksize=-1)
    sobelY = cv2.Sobel(gray_img, cv2.CV_64F, 0, 1, ksize=-1)

    # Convert gradients to integer steps
    stepsX = sobelY.astype(int)
    stepsY = sobelX.astype(int)

    # Compute gradient magnitudes and normalize gradients
    magnitudes = np.sqrt(stepsX * stepsX + stepsY * stepsY)
    gradsX = stepsX / (magnitudes + 1e-10)
    gradsY = stepsY / (magnitudes + 1e-10)

    # Initialize array of stroke widths for each pixel
    h, w = raw_img.shape[:2]
    stroke_widths = np.array([[np.inf, np.inf]])

    # Loop over every pixel
    for i in range(h):
        for j in range(w):
            # Check if the current pixel is an edge
            if canny_img[i, j] != 0:
                # Collect gradient direction
                gradX = gradsX[i, j]
                gradY = gradsY[i, j]

                # Initialize variables for following gradient direction
                prevX, prevY, prevX_opp, prevY_opp, step_size = i, j, i, j, 0
                # Flags for following gradient direction (Both directions)
                go, go_opp = True, True

                # Initialize stroke width variables
                stroke_width = np.inf
                stroke_width_opp = np.inf
                while (go or go_opp) and (step_size < step_limit):
                    step_size += 1

                    # Follow gradient direction
                    if go:
                        curX = int(np.floor(i + gradX * step_size))
                        curY = int(np.floor(j + gradY * step_size))
                        # Check if the current pixel is out of bounds
                        if (curX < 0 or curY < 0 or curX >= h or curY >= w):
                            go = False
                        if go and ((curX != prevX) or (curY != prevY)):
                            try:
                                # Check if next pixel is an edge
                                if canny_img[curX, curY] != 0:
                                    # Check if edge is opposite to gradient direction
                                    if np.arccos(gradX * -gradsX[curX, curY] + gradY * -gradsY[curX, curY]) < np.pi / 2.0:
                                        # Calculate stroke width
                                        stroke_width = int(np.sqrt((curX - i) ** 2 + (curY - j) ** 2))
                                        go = False
                            except IndexError:
                                go = False

                            prevX = curX
                            prevY = curY

                    # Traverse in the opposite gradient direction
                    if go_opp:
                        curX_opp = int(np.floor(i - gradX * step_size))
                        curY_opp = int(np.floor(j - gradY * step_size))
                        if (curX_opp < 0 or curY_opp < 0 or curX_opp >= h or curY_opp >= w):
                            go_opp = False
                        if go_opp and ((curX_opp != prevX_opp) or (curY_opp != prevY_opp)):
                            try:
                                if canny_img[curX_opp, curY_opp] != 0:
                                    if np.arccos(gradX * -gradsX[curX_opp, curY_opp] + gradY * -gradsY[curX_opp, curY_opp]) < np.pi / 2.0:
                                        stroke_width_opp = int(np.sqrt((curX_opp - i) ** 2 + (curY_opp - j) ** 2))
                                        go_opp = False

                            except IndexError:
                                go_opp = False

                            prevX_opp = curX_opp
                            prevY_opp = curY_opp

                # Append stroke widths in both directions to array
                stroke_widths = np.append(stroke_widths, [(stroke_width, stroke_width_opp)], axis=0)

    # Remove infinite stroke width
    stroke_widths_opp = np.delete(stroke_widths[:, 1], np.where(stroke_widths[:, 1] == np.inf))
    stroke_widths = np.delete(stroke_widths[:, 0], np.where(stroke_widths[:, 0] == np.inf))

    return stroke_widths, stroke_widths_opp

# -----------------------------------------------------------------------------


def average_SWT(raw_img):
    stroke_widths, _ = get_strokes(raw_img)
    # Filter out infinite values
    stroke_widths = stroke_widths[stroke_widths != np.inf]

    if len(stroke_widths) == 0:
        return 0

    return np.mean(stroke_widths)


# Added threshold that can allow for better groupings of non-overlapping characters
# Threshold - Minimum number of consecutive results required to make a group
# MinSplit - Minimum percentage of the bound for an AW point to be valid for a group
def get_character_splits(smeared_img, AW, threshold=1, min_split=0.2):
    global AveWid
    # Get projection of image along x-axis (horizontal projection)
    xProj = np.sum(smeared_img, axis=0)
    # Collect points where the projection is 0
    zero_points = np.where(xProj <= 0.01)[0]

    # Group zero points into consecutive groups, then return the mean
    mean_points = np.array([])
    if (len(zero_points) > 0):
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
        plt.vlines(x=mean_points, ymin=0, ymax=graph_top, color='red', linestyle='--', label='Zero point')
        if (len(aw_points) > 1):
            plt.vlines(x=aw_points, ymin=0, ymax=graph_top, color='blue', linestyle='--', label='AW point')
        elif len(aw_points) == 1:
            plt.axvline(x=aw_points[0], color='blue', linestyle='--', label='AW point')

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
    global AveWid
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
    split_points, aw_points = get_character_splits(smeared_img, AW, 5, 0.4)

    if show_images:
        plt.imshow(cv2.cvtColor(raw_img, cv2.COLOR_BGR2RGB))

        for split in split_points:
            plt.axvline(x=split, color='red', linestyle='--', linewidth=1)
        for aw in aw_points:
            plt.axvline(x=aw, color='blue', linestyle='--', linewidth=1)
        plt.show()

    return split_points, aw_points


def show_cropped_sections(cropped_images):
    num_images = len(cropped_images)
    fig, axes = plt.subplots(1, num_images, figsize=(15, 5))

    if num_images == 1:
        axes = [axes]

    for ax, cropped_img in zip(axes, cropped_images):
        ax.imshow(cropped_img, cmap='gray')
        ax.axis('off')

    plt.show()


def draw_local_peak(cropped, img):
    global AveWid
    # Get projection of image along x-axis (vertical projection)
    yProj = np.sum(cropped, axis=0)

    # Find local peaks (points where the derivative is negative)
    peaks = []
    for i in range(1, len(yProj) - 1):
        if (yProj[i - 1] < yProj[i]) and (yProj[i] > yProj[i + 1]):
            peaks.append(i)

    threshold = AveWid * 0.5
    merged_peaks = [peaks[0]]
    for peak in peaks[1:]:
        if peak - merged_peaks[-1] <= threshold:
            if yProj[peak] > yProj[merged_peaks[-1]]:
                merged_peaks[-1] = peak
        else:
            merged_peaks.append(peak)
    peaks = merged_peaks

    # Calculate the ratio between the image width and AveWid
    num_peaks = round(cropped.shape[1] / AveWid)

    peaks = sorted(peaks, key=lambda x: yProj[x], reverse=True)

    plt.plot(yProj)
    plt.vlines(peaks[:num_peaks], ymin=min(yProj), ymax=max(yProj), colors='r', linestyles='dashed')
    plt.title('yProjection Visualization')
    plt.xlabel('Index')
    plt.ylabel('Sum')
    plt.show()

    # Draw lines at the largest local peaks
    for peak in peaks[:num_peaks]:
        img[:, peak, 0] = 255

    return img, peaks[:num_peaks]


def generate_voronoi_diagram(skeleton_img, vertical_lines):
    vor_diagrams = []
    for current, next in zip(vertical_lines, vertical_lines[1:]):
        if current > next:
            temp = current
            current = next
            next = temp
        current += 1
        segment = skeleton_img[:, current:next]
        show_cropped_sections([segment])
        endpoints = find_endpoints(segment)
        junctions = find_junctions(segment)

        endpoints = [(y, x) for y, x in endpoints]
        junctions = [(y, x) for y, x in junctions]

        critical_points = endpoints + junctions

        plt.imshow(segment, cmap='gray', origin='upper')
        for point in endpoints:
            y, x = point
            plt.plot(x, y, 'ro')
        for point in junctions:
            y, x = point
            plt.plot(x, y, 'go')
        plt.show()

        if (len(critical_points) < 1):
            print("\033[91m" + "WARNING: No critical points found" + "\033[0m")
            return

        vor_diagram = Voronoi(critical_points)
        vor_diagrams.append(vor_diagram)

    return vor_diagrams


def create_cut(skeleton_img, vertical_lines, vor_diagrams):
    for current, next in zip(vertical_lines, vertical_lines[1:]):
        current += 1
        if current > next:
            temp = current
            current = next
            next = temp
        segment = skeleton_img[:, current:next]

        vor_diagram = vor_diagrams.pop()
        finite_segments, infinite_segments = process_voronoi(vor_diagram, segment)
        segment_fitness = get_segment_fitness([*finite_segments, *infinite_segments], segment)
        finite_fitness = segment_fitness[:len(finite_segments)]
        infinite_fitness = segment_fitness[len(finite_segments):]

        all_segments = [*finite_segments, *infinite_segments]
        all_fitness = [*finite_fitness, *infinite_fitness]
        display_segments(segment, all_segments, all_fitness)

        path = a_star(segment, all_segments, all_fitness)
        display_path = path_to_segments(path)
        display_segments(segment, display_path, all_fitness, False)

        path = [(current + x, y) for (x, y) in path]

        return path


def a_star(image, segments, fitness):
    height, width = image.shape[:2]
    height, width = height - 1, width - 1

    start_positions = [seg[0] if seg[0][1] < 0 else seg[1] for seg in segments if seg[0][1] < 0 or seg[1][1] < 0]
    end_positions = [seg[0] if seg[0][1] > height else seg[1] for seg in segments if seg[0][1] > height or seg[1][1] > height]

    start_positions = list(set(start_positions))
    end_positions = list(set(end_positions))

    # Algorithm from OVRDOZE
    # https://github.com/dille12/OVRDOZE
    # Original code by dille12
    # https://github.com/dille12

    adjacency = {}
    for i, (node_a, node_b) in enumerate(segments):
        cost = fitness[i]
        adjacency.setdefault(node_a, []).append((node_b, cost))
        adjacency.setdefault(node_b, []).append((node_a, cost))

    open_set = []   # priority queue of (f_score, node)
    came_from = {}  # track best parent to reconstruct path
    g_score = {}    # cost from start to current node
    f_score = {}    # estimated cost from start to goal through current node

    for s in start_positions:
        g_score[s] = 0.0
        f_score[s] = heuristic(s, end_positions)
        heapq.heappush(open_set, (f_score[s], s))

    # Standard A* loop
    while open_set:
        current_f, current = heapq.heappop(open_set)

        # If current is an end node, reconstruct the path and return
        if current in end_positions:
            return reconstruct_path(came_from, current)

        # If this is an outdated entry in the priority queue, skip
        if current_f > f_score.get(current, float('inf')):
            continue

        # For each neighbor, check if we've found a better path
        for neighbor, edge_cost in adjacency.get(current, []):
            tentative_g = g_score[current] + edge_cost
            if tentative_g < g_score.get(neighbor, float('inf')):
                # This is a better path. Record it.
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g
                f_score[neighbor] = tentative_g + heuristic(neighbor, end_positions)
                heapq.heappush(open_set, (f_score[neighbor], neighbor))

    # If we exit the loop, there's no path
    print("No path could be found from any start to any end.")
    return None


def heuristic(current_node, end_nodes):
    """
    For multi-end, we use the minimum Euclidean distance to any of the end nodes.
    """
    if not end_nodes:
        return 0.0
    (x1, y1) = current_node
    # Compute the min distance from current_node to any end_node
    return min(
        math.hypot(x1 - x2, y1 - y2)
        for (x2, y2) in end_nodes
    )


def reconstruct_path(came_from, current):
    path = [current]
    while current in came_from:
        current = came_from[current]
        path.append(current)
    path.reverse()
    return path


def path_to_segments(path):
    if not path or len(path) < 2:
        return []
    return [[path[i], path[i + 1]] for i in range(len(path) - 1)]


def display_segments(image, seg, fit, show_fitness=True):
    min_fitness = min(fit)
    max_fitness = max(fit)

    height, width = image.shape[:2]
    height, width = height - 1, width - 1
    plt.imshow(image, cmap='gray', origin='upper')
    cmap = mcolors.LinearSegmentedColormap.from_list("", ["green", "red"])

    for i, segment in enumerate(seg):
        p1, p2 = segment
        percentage = (fit[i] - min_fitness) / (max_fitness - min_fitness)
        color = cmap(percentage)
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], color=color, lw=1)

    if show_fitness:
        for i, fitness in enumerate(fit):
            percentage = (fit[i] - min_fitness) / (max_fitness - min_fitness)
            color = cmap(percentage)
            plt.text(seg[i][0][0], seg[i][0][1], str(round(fitness, 2)), color=color)

    plt.xlim(0, width)
    plt.ylim(height, 0)
    plt.show()


def get_segment_fitness(segment_list, image):
    kernel = [[1, 2, 1], [2, 3, 2], [1, 2, 1]]
    fitness_values = []

    for segment in segment_list:
        (x1, y1), (x2, y2) = segment
        line_pixels = bresenham_line(x1, y1, x2, y2)

        fitness = 0
        for (x, y) in line_pixels:
            fitness += apply_kernel(image, kernel, x, y)

        line_length = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        fitness += line_length

        fitness_values.append(fitness)

    return fitness_values


def apply_kernel(image, kernel, x, y):
    kernel_size = len(kernel)
    offset = kernel_size // 2

    # If the kernel cannot be fully applied at boundaries return 0
    rows = len(image)
    cols = len(image[0]) if rows > 0 else 0

    # Check boundaries
    if (x - offset < 0 or x + offset >= cols or y - offset < 0 or y + offset >= rows):
        return 0

    # Accumulate the sum within the 3×3 region
    kernel_sum = 0
    for ky in range(kernel_size):
        for kx in range(kernel_size):
            # Map kernel coordinates to image coordinates
            img_x = x + (kx - offset)
            img_y = y + (ky - offset)
            kernel_sum += image[img_y][img_x] * kernel[ky][kx]

    return kernel_sum[0]


def bresenham_line(x1, y1, x2, y2):
    # Algorithm from python-bresenham
    # https://github.com/Azrood/python-bresenham
    # Original code by Azrood
    # https://github.com/Azrood
    points = []

    # Calculate deltas
    dx = x2 - x1
    dy = y2 - y1

    # Determine how steep the line is
    is_steep = abs(dy) > abs(dx)

    # If the line is steep, transpose coordinates
    if is_steep:
        x1, y1 = y1, x1
        x2, y2 = y2, x2

    # Swap if starting x is greater than ending x
    swapped = False
    if x1 > x2:
        x1, x2 = x2, x1
        y1, y2 = y2, y1
        swapped = True

    dx = x2 - x1
    dy = y2 - y1
    error = int(dx / 2.0)
    ystep = 1 if y1 < y2 else -1

    y = y1
    for x in range(x1, x2 + 1):
        # Transpose coordinates back if needed
        if is_steep:
            points.append((y, x))
        else:
            points.append((x, y))

        error -= abs(dy)
        if error < 0:
            y += ystep
            error += dx

    # If swapped, reverse the result
    if swapped:
        points.reverse()

    return points


def process_voronoi(vor, image):
    height, width = image.shape[:2]
    height, width = height - 1, width - 1
    plt.imshow(image, cmap='gray', origin='upper')

    # Modified from scipy source code
    # https://github.com/scipy/scipy/blob/v1.15.1/scipy/spatial/_plotutils.py
    # Original code by Pauli Virtanen
    # https://github.com/pv

    center = vor.points.mean(axis=0)
    ptp_bound = np.ptp(vor.points, axis=0)

    finite_segments = []
    infinite_segments = []
    for pointidx, simplex in zip(vor.ridge_points, vor.ridge_vertices):
        simplex = np.asarray(simplex)
        if np.all(simplex >= 0):
            p1 = vor.vertices[simplex[0]]
            p2 = vor.vertices[simplex[1]]

            # Apply rotation and flipping
            p1_transformed = (round(p1[1]), round(p1[0]))
            p2_transformed = (round(p2[1]), round(p2[0]))

            finite_segments.append([p1_transformed, p2_transformed])
        else:
            i = simplex[simplex >= 0][0]  # finite end Voronoi vertex
            t = vor.points[pointidx[1]] - vor.points[pointidx[0]]  # tangent
            t /= np.linalg.norm(t)
            n = np.array([-t[1], t[0]])  # normal

            midpoint = vor.points[pointidx].mean(axis=0)
            direction = np.sign(np.dot(midpoint - center, n)) * n
            if (vor.furthest_site):
                direction = -direction
            aspect_factor = abs(ptp_bound.max() / ptp_bound.min())
            far_point = vor.vertices[i] + direction * ptp_bound.max() * aspect_factor

            i_transformed = (round(vor.vertices[i][1]), round(vor.vertices[i][0]))
            far_point_transformed = (round(far_point[1]), round(far_point[0]))

            infinite_segments.append([i_transformed, far_point_transformed])

    # Plot finite segments
    for segment in finite_segments:
        p1, p2 = segment
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r-', lw=1)

    # Plot infinite segments
    for segment in infinite_segments:
        p1, p2 = segment
        plt.plot([p1[0], p2[0]], [p1[1], p2[1]], 'r--', lw=1)

    # Highlight critical points with transformations
    for point in vor.points:
        transformed_point = (point[1], point[0])
        plt.plot(transformed_point[0], transformed_point[1], 'go', markersize=3)

    plt.xlim(0, width)
    plt.ylim(height, 0)

    plt.title('Voronoi Diagram')
    plt.legend()
    plt.axis('off')
    plt.show()

    return finite_segments, infinite_segments


def find_endpoints(img):
    bin_img = (img.reshape(img.shape[0], img.shape[1]) > 0).astype(int)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0

    neighbor_counts = convolve(bin_img, kernel, mode='constant', cval=0)
    endpoints = np.argwhere((bin_img == 1) & (neighbor_counts == 1))

    endpoint_list = [tuple(loc) for loc in endpoints]
    return endpoint_list


def find_junctions(img):
    bin_img = (img.reshape(img.shape[0], img.shape[1]) > 0).astype(int)
    kernel = np.ones((3, 3), dtype=int)
    kernel[1, 1] = 0

    neighbor_counts = convolve(bin_img, kernel, mode='constant', cval=0)

    candidates = np.argwhere((bin_img == 1) & (neighbor_counts >= 3))
    junctions = []
    for y, x in candidates:
        if non_adjacent_neighbors(bin_img, y, x):
            junctions.append((y, x))

    return [tuple(loc) for loc in junctions]


def non_adjacent_neighbors(bin_img, y, x):
    # Collect all valid 'on' neighbors (8 directions)
    nbrs = []
    for dy in [-1, 0, 1]:
        for dx in [-1, 0, 1]:
            if (dy, dx) != (0, 0):
                ny, nx = y + dy, x + dx
                if (0 <= ny < bin_img.shape[0] and 0 <= nx < bin_img.shape[1] and bin_img[ny, nx] == 1):
                    nbrs.append((ny, nx))

    # Check pairwise adjacency among neighbors
    for (y1, x1), (y2, x2) in combinations(nbrs, 2):
        # If they lie within 1 step of each other (including diagonals), they're adjacent
        if abs(y1 - y2) <= 1 and abs(x1 - x2) <= 1:
            return False

    return True


def fine_segmentation(img, split_points, aw_points):
    if (len(aw_points) < 1):
        final_images = []
        for i in range(len(split_points) - 1):
            lower_bound = math.floor(split_points[i])
            upper_bound = math.ceil(split_points[i + 1])
            final_images.append(img[:, lower_bound:upper_bound])
            return final_images

    oversized_bounds = []
    for aw_point in aw_points:
        lower_bound = max([point for point in split_points if point < aw_point])
        upper_bound = min([point for point in split_points if point > aw_point])
        oversized_bounds.append((math.floor(lower_bound), math.ceil(upper_bound)))

    cropped_images = []
    skeleton_images = []
    for lower_bound, upper_bound in oversized_bounds:
        cropped_img = [img[:, lower_bound:upper_bound] < 0.9, lower_bound, upper_bound]
        cropped_images.append(cropped_img)
        skeleton = skeletonize(cropped_img[0])
        plt.imshow(skeleton, cmap="gray")
        plt.show()
        skeleton_images.append(skeleton)

    show_cropped_sections([img for img, _, _ in cropped_images])

    divided_images = []
    for i in range(len(skeleton_images)):
        skeleton_images[i], vl = draw_local_peak(cropped_images[i][0], skeleton_images[i])
        show_cropped_sections([skeleton_images[i]])
        voronoi_diagrams = generate_voronoi_diagram(skeleton_images[i], vl)
        cut_path = create_cut(skeleton_images[i], vl, voronoi_diagrams)
        divided_images.append(cut_image(img[:, cropped_images[i][1]:cropped_images[i][2]], cut_path))

    final_images = []
    for i in range(len(split_points) - 1):
        lower_bound = math.floor(split_points[i])
        if (lower_bound in [bound[0] for bound in oversized_bounds]):
            final_images.extend(divided_images.pop(0))
            continue
        upper_bound = math.ceil(split_points[i + 1])
        final_images.append(img[:, lower_bound:upper_bound])

    return final_images


def cut_image(cropped_img, cut_path):
    height, width = cropped_img.shape[:2]
    cut_path.insert(0, (0, 0))
    cut_path.append((0, height))

    mask_pos = np.ones(cropped_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_pos, [np.array(cut_path, dtype=np.int32)], 0)
    mask_neg = np.zeros(cropped_img.shape[:2], dtype=np.uint8)
    cv2.fillPoly(mask_neg, [np.array(cut_path, dtype=np.int32)], 1)

    plt.imshow(mask_pos, cmap='gray')
    plt.show()

    plt.imshow(mask_neg, cmap='gray')
    plt.show()

    img1 = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_neg)
    img2 = cv2.bitwise_and(cropped_img, cropped_img, mask=mask_pos)
    img1[mask_neg == 0] = 1
    img2[mask_pos == 0] = 1

    plt.imshow(img1, cmap='gray')
    plt.show()

    plt.imshow(img2, cmap='gray')
    plt.show()

    x, y, w, h = cv2.boundingRect(mask_neg)
    img1_cropped = img1[y : y + h, x : x + w]

    x, y, w, h = cv2.boundingRect(mask_pos)
    img2_cropped = img2[y : y + h, x : x + w]

    plt.imshow(img1_cropped, cmap='gray')
    plt.show()

    plt.imshow(img2_cropped, cmap='gray')
    plt.show()

    return img1_cropped, img2_cropped


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
    if (len(zero_points) > 0):
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
    input_file = 'images/input.png'
    raw_img = open_grayscale(input_file)
    raw_img = remove_space(raw_img)
    split_points, aw_points = coarse_segmentation(raw_img, flip_binary)
    output = fine_segmentation(raw_img, split_points, aw_points)

    for img in output:
        plt.imshow(img, cmap='gray')
        plt.show()
