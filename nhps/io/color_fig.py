from matplotlib import pyplot as plt
import numpy as np
import os
import pickle
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator


def normal_kernel(sigma):
    def normal_function(x):
        return np.exp(-x**2 / sigma**2) / np.sqrt(2*np.pi*sigma**2)
    return normal_function


def get_density_matrix(pts, xs, ys, kernel_function):
    """

    :param np.ndarray pts: shape=[n_pts, 2]
    :param np.ndarray xs: shape=[n_x]
    :param np.ndarray ys: shape=[n_y]
    :param func kernel_function:
    :rtype: np.ndarray
    """
    n_pts, dim = pts.shape
    assert dim == 2
    n_x, n_y = xs.shape[0], ys.shape[0]
    # xs: shape=[n_x, n_pts]
    xs = xs.repeat(n_pts).reshape(n_x, n_pts)
    xs = np.abs(xs - pts[:, 0])
    # ys: shape=[n_y, n_pts]
    ys = ys.repeat(n_pts).reshape(n_y, n_pts)
    ys = np.abs(ys - pts[:, 1])
    # xs: shape=[n_pts, n_x, n_y]
    xs = xs.repeat(n_y).reshape(n_x, n_pts, n_y).transpose([1, 0, 2])
    # ys: shape=[n_pts, n_x, n_y]
    ys = ys.repeat(n_x).reshape(n_y, n_pts, n_x).transpose([1, 2, 0])
    # distances: shape=[n_pts, n_x, n_y]
    distances = np.sqrt(xs ** 2 + ys ** 2)
    values = kernel_function(distances)
    return values.sum(axis=0)


def get_range_helper(xs, ys, get_edge, n_step):

    left_most, right_most = get_edge(xs)
    down_most, up_most = get_edge(ys)
    left_down_most = min(left_most, down_most)
    right_up_most = max(right_most, up_most)
    step = (right_up_most - left_down_most) / n_step
    eps = 1e-1
    step *= 1 - 1 / n_step * eps
    x_range = y_range = np.arange(left_down_most, right_up_most, step)
    assert len(x_range) == n_step + 1

    return x_range, y_range


def get_range_for_multiple_centroid(centroid_x, centroid_y, n_step, margin):

    def get_edge(centroid):
        leftmost, rightmost = centroid.min(), centroid.max()
        width = rightmost - leftmost
        leftmost -= width * margin
        rightmost += width * margin
        return leftmost, rightmost

    return get_range_helper(centroid_x, centroid_y, get_edge, n_step)


def get_range_for_single_centroid(pts, n_step, margin):
    xs, ys = pts.T

    def get_edge(xy):
        left_edge = np.percentile(xy, 20)
        right_edge = np.percentile(xy, 80)
        width = right_edge - left_edge
        left_edge -= width * margin
        right_edge += width * margin

        return left_edge, right_edge

    return get_range_helper(xs, ys, get_edge, n_step)


def draw_colored_fig(pts, sigma=0.1, centroids=None, n_step=500, margin=0.3, vertical=True, label_size=None, anno_size=None):
    """
    :param list[np.ndarray] pts: each shape=[n_pts, 2]
    :param float sigma:
    :param np.ndarray centroids:
    :param int n_step:
    :param float margin:
    :param bool vertical:
    :return:
    """
    n_centers = len(pts)
    if centroids is None:
        centroid_x = np.zeros(shape=[n_centers], dtype=np.float32)
        centroid_y = centroid_x.copy()
        for idx_center, pts_ in enumerate(pts):
            centroid_x[idx_center], centroid_y[idx_center] = np.average(pts_, axis=0)
    else:
        centroid_x, centroid_y = centroids

    if len(pts) == 1:
        x_range, y_range = get_range_for_single_centroid(pts[0], n_step, margin)
    else:
        x_range, y_range = get_range_for_multiple_centroid(centroid_x, centroid_y, n_step, margin)

    grid_x, grid_y = np.meshgrid(x_range, y_range)

    normal_kernel_func = normal_kernel(sigma)
    normalization_factor = 1.0
    value_matrix = np.zeros(shape=[len(pts), n_step, n_step], dtype=np.float32)
    for pt_idx, pts_ in enumerate(pts):
        print('Doing with {}/{}-th dataset.'.format(pt_idx+1, len(pts)))
        single_matrix = get_density_matrix(pts_, x_range[:-1], y_range[:-1], normal_kernel_func)
        single_matrix -= single_matrix.min()
        single_matrix /= single_matrix.max() * normalization_factor
        value_matrix[pt_idx] = single_matrix

    value_matrix = value_matrix

    value_matrix = np.max(value_matrix, axis=0)
    value_matrix -= value_matrix.min()
    value_matrix /= value_matrix.max() * normalization_factor

    levels = MaxNLocator(nbins=256).tick_values(0.0, 1.0)
    cmap = plt.get_cmap('Blues')
    norm = BoundaryNorm(levels, ncolors=cmap.N, clip=True)
    f, ax = plt.subplots(figsize=[6, 6])
    ax.pcolormesh(grid_x, grid_y, value_matrix, cmap=cmap, norm=norm)
    ax.plot(ax.get_xlim(), ax.get_ylim(), ls="--", c=".1")
    ax.scatter(centroid_x, centroid_y, 40, color='black')

    plt.xlabel('particle filtering', size=label_size)
    plt.ylabel('particle smoothing', size=label_size)

    # draw vertical line
    draw_avg_line_if_possible = False
    if vertical:
        if len(centroid_x) == 1 or not draw_avg_line_if_possible:
            leftmost_idx = np.argmin(centroid_x)
            beg_x = centroid_x[leftmost_idx]
            beg_y = centroid_y[leftmost_idx]
        else:
            x_avg, y_avg = np.average([centroid_x, centroid_y], axis=1)
            ax.plot(ax.get_xlim(), [y_avg] * 2, '--b')
            beg_x = x_avg
            beg_y = y_avg

        end_x = end_y = beg_x

        y_delta = abs(beg_y - end_y)

        line_prop = dict(
            arrowstyle='<|-|>',
            shrinkA=5,
            shrinkB=5,
            linewidth=2,
            color='black'
        )

        ax.annotate('', xytext=[beg_x, beg_y], xy=[end_x, end_y],
                    arrowprops=line_prop,
                    size=15)

        text_position = [end_x, (beg_y+end_y)/2]
        ax.annotate(' {:.3f} nats'.format(y_delta), xytext=text_position,
                    xy=text_position,
                    size=anno_size)
        # ax.plot([beg_x, end_x], [beg_y, end_y])

    return ax
