import numpy as np


def max_triangle_1d(width, time_stamps, heights):
    """

    :param float width:
    :param np.ndarray time_stamps:
    :param np.ndarray heights:
    """
    distances = np.outer(time_stamps, np.ones(shape=[len(time_stamps)], dtype=np.float32))
    distances = np.abs(distances - time_stamps)
    effective_distances = width - distances
    effective_distances[effective_distances < 0] = 0.0
    gain = heights / width
    values = (effective_distances * gain).sum(axis=1)

    highest_idx = np.argmax(values)
    highest_value = values[highest_idx]

    return int(highest_idx), highest_value


def max_triangle_2d(width, time_stamps, heights, lens):
    """

    :param float width:
    :param np.ndarray time_stamps: shape=[n, m], dtype=np.float32
    :param np.ndarray heights: shape=[n], dtype=np.float32
    :param np.ndarray lens: shape=[n], dtype=np.int32
    """
    n, m = time_stamps.shape
    flatten_time_stamps = time_stamps.reshape(n * m)
    distances = np.outer(flatten_time_stamps, np.ones(shape=[n*m], dtype=np.float32))
    distances = np.abs(distances - flatten_time_stamps)
    distances = width - distances
    distances[distances < 0] = 0.0

    # shape=[n1, m1, n2, m2]
    distances = distances.reshape(n, m, n, m)
    # shape=[n1, m1, m2, n2]
    distances = distances.transpose([0, 1, 3, 2])
    gain = heights / width
    distances = distances * gain
    # shape=[n1, m1, n2, m2]
    distances = distances.transpose([0, 1, 3, 2])
    # shape=[m, n]
    len_mask = np.outer(np.arange(m, dtype=np.int32), np.ones(shape=[n], dtype=np.int32))
    len_mask = len_mask < lens
    # shape=[n, m]
    len_mask = len_mask.transpose([1, 0])

    distances[:, :, ~len_mask] = 0.0
    # shape=[n1, m1, n2]
    distances_each = np.max(distances, axis=3)
    # shape=[n1, m1]
    distances_sum = distances_each.sum(axis=2)
    distances_sum[~len_mask] = 0.0
    # shape=[n1 * m1]
    distances_sum = distances_sum.reshape(n * m)

    max_idx = np.argmax(distances_sum)
    max_value = distances_sum[max_idx]
    max_idx1 = max_idx // m
    max_idx2 = max_idx % m

    # shape=[n2, m2]
    distances_sub_mat = distances[max_idx1, max_idx2]
    choices = distances_sub_mat.argmax(axis=1)
    choice_mask = distances_sub_mat.max(axis=1) <= 0.0
    choices[choice_mask] = -1

    return [max_idx1, max_idx2], max_value, choices


def concat_pad_mat(a, pad=0):
    """

    :param list[np.ndarray] a:
    :param float pad:
    :rtype: np.ndarray
    """
    max_len = max([len(item) for item in a])
    n = len(a)
    rst = np.full(shape=[n, max_len], fill_value=pad, dtype=a[0].dtype)
    for row_idx, row in enumerate(a):
        rst[row_idx, :len(row)] = row
    return rst
