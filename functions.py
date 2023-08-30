import warnings
from cmath import log10, sqrt, inf
from collections import Counter

import cv2
import numpy as np
from bitarray import bitarray


MAX_VAL_SQ = 255.0**2
DEPTH = 8
MAX_MSE_FOR_PREDICTION = 50
BLOCK_SIZE = 8
FUTURE_REFERENCE_SIGN = "f"

QTY = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],  # luminance quantization table
        [12, 12, 14, 19, 26, 48, 60, 55],
        [14, 13, 16, 24, 40, 57, 69, 56],
        [14, 17, 22, 29, 51, 87, 80, 62],
        [18, 22, 37, 56, 68, 109, 103, 77],
        [24, 35, 55, 64, 81, 104, 113, 92],
        [49, 64, 78, 87, 103, 121, 120, 101],
        [72, 92, 95, 98, 112, 100, 103, 99],
    ]
)

QTC = np.array(
    [
        [1, 9, 8, 13, 19, 32, 41, 49],
        [10, 10, 11, 15, 21, 46, 48, inf],
        [11, 10, 13, 19, 32, 46, 55, 45],
        [11, 14, 18, 23, 41, inf, inf, inf],
        [14, 18, 30, 45, 54, inf, inf, inf],
        [19, 28, 44, 51, inf, inf, inf, inf],
        [39, 51, 62, inf, inf, inf, inf, inf],
        [58, inf, inf, inf, inf, inf, inf, inf],
    ]
)

ZIGZAGINVERSE = np.array(
    [
        [0, 1, 5, 6, 14, 15, 27, 28],
        [2, 4, 7, 13, 16, 26, 29, 42],
        [3, 8, 12, 17, 25, 30, 41, 43],
        [9, 11, 18, 24, 31, 40, 44, 53],
        [10, 19, 23, 32, 39, 45, 52, 54],
        [20, 22, 33, 38, 46, 51, 55, 60],
        [21, 34, 37, 47, 50, 56, 59, 61],
        [35, 36, 48, 49, 57, 58, 62, 63],
    ]
)

ZIGZAGFLATINVERSE = ZIGZAGINVERSE.flatten()
ZIGZAGFLAT = np.argsort(ZIGZAGFLATINVERSE)


def dezigzag_dequantize_idct(
    new, unrled, i: int, j: int, block_num: int, mode="luma", block_size=BLOCK_SIZE
):
    dezigzag = np.asarray(inverse_zigzag_single(unrled[block_num])).astype(np.float32)

    new[
        i * block_size : i * block_size + block_size,
        j * block_size : j * block_size + block_size,
    ] = np.ceil(cv2.idct(dequantize(dezigzag, mode)))

    return


def ycbcr2rgb(im):
    xform = np.array([[1, 1.772, 0], [1, -0.34414, -0.71414], [1, 0, 1.402]])
    rgb = im.astype(float)
    rgb[:, :, [1, 2]] -= 128
    rgb = rgb.dot(xform.T)
    np.putmask(rgb, rgb > 255, 255)
    np.putmask(rgb, rgb < 0, 0)
    return np.uint8(rgb)


def idct_single(arr):
    return cv2.idct(arr)


def dct(arr: np.ndarray, i: int, j: int, block_size=BLOCK_SIZE):
    return cv2.dct(
        arr[
            i * block_size : i * block_size + block_size,
            j * block_size : j * block_size + block_size,
        ]
    )


def pad2(arr1, arr2, l, w, block_size=BLOCK_SIZE):
    if (len(arr1[0]) % block_size == 0) and (len(arr1) % block_size == 0):
        arrPadded1 = arr1.copy()
        arrPadded2 = arr2.copy()
    else:
        arrPadded1 = np.zeros((l, w))
        arrPadded2 = np.zeros((l, w))
        for i in range(len(arr1)):
            for j in range(len(arr1[0])):
                arrPadded1[i, j] += arr1[i, j]
                arrPadded2[i, j] += arr2[i, j]
    return arrPadded1, arrPadded2


def pad(arr, l, w, block_size=BLOCK_SIZE):
    if (len(arr[0]) % block_size == 0) and (len(arr) % block_size == 0):
        arrPadded = arr.copy()
    else:
        arrPadded = np.zeros((l, w))
        for i in range(len(arr)):
            for j in range(len(arr[0])):
                arrPadded[i, j] += arr[i, j]

    return arrPadded


def quantize(arr, mode="luma"):
    if mode == "luma":
        QT = QTY
    else:
        QT = QTC
    return np.nan_to_num(np.ceil(arr / QT))


def dequantize(arr, mode="luma"):
    if mode == "luma":
        QT = QTY
    else:
        QT = QTC
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return np.nan_to_num(arr * QT)


def decode_I_frame_block(decoded, dct_quantized, i: int, j: int, block_size=BLOCK_SIZE):
    decoded[
        i * block_size : i * block_size + block_size,
        j * block_size : j * block_size + block_size,
    ] = idct_single(dequantize(dct_quantized))


def get_referenced_block(
    reference_frame_1, vector, reference_frame_2=None, block_size=BLOCK_SIZE
):
    return reference_frame_1[
        vector[0] : vector[0] + block_size, vector[1] : vector[1] + block_size
    ]


def trim(array: np.ndarray) -> np.ndarray:
    """
    in case the trim_zeros function returns an empty array, add a zero to the array to use as the DC component
    :param numpy.ndarray array: array to be trimmed
    :return numpy.ndarray:
    """
    trimmed = np.trim_zeros(array, "b")
    if len(trimmed) == 0:
        trimmed = np.zeros(1)
    return trimmed


def mean_square_error(arr1, arr2):
    return np.mean((arr1 - arr2) ** 2)


def PSNR(original, compressed):
    mse = mean_square_error(original, compressed)
    if mse == 0:  # MSE is zero means no noise is present in the signal .
        # Therefore PSNR have no importance.
        return inf
    psnr = 20 * log10(MAX_VAL_SQ / mse)
    return psnr.__abs__()


def run_length_encoding(array: np.ndarray) -> list:
    """
    finds the intermediary stream representing the zigzags
    format for DC components is <size><amplitude>
    format for AC components is <run_length, size> <Amplitude of non-zero>
    :param numpy.ndarray array: zigzag vectors in array
    :returns: run length encoded values as an array of tuples
    """
    encoded = list()
    run_length = 0
    eob = ("EOB",)

    for i in range(len(array)):
        for j in range(len(array[i])):
            trimmed = trim(array[i])
            if j == len(trimmed):
                encoded.append(eob)  # EOB
                break
            if i == 0 and j == 0:  # for the first DC component
                encoded.append((int(trimmed[j]).bit_length(), trimmed[j]))
            elif j == 0:  # to compute the difference between DC components
                diff = int(array[i][j] - array[i - 1][j])
                if diff != 0:
                    encoded.append((diff.bit_length(), diff))
                else:
                    encoded.append((1, diff))
                run_length = 0
            elif trimmed[j] == 0:  # increment run_length by one in case of a zero
                run_length += 1
            else:  # intermediary steam representation of the AC components
                encoded.append((run_length, int(trimmed[j]).bit_length(), trimmed[j]))
                run_length = 0
            # send EOB
        if not (encoded[len(encoded) - 1] == eob):
            encoded.append(eob)
    return encoded


def rle_motion_vectors(vectors):
    encoded = list()
    rl = 0
    for i, v in enumerate(vectors):
        if i == 0:
            encoded.append(v)
        elif v == [0, 0]:
            rl += 1
        else:
            encoded.append([rl, v])
            rl = 0

    return encoded


def unrle(ls: list):
    zigzag = []
    row = []
    dc = 0
    count = 0
    for i in ls:
        if len(i) == 2:
            dc = dc + i[1]
            row.append(dc)
            count += 1
        elif i == ("EOB",):
            while count < 64:
                row.append(0)
                count += 1
            zigzag.append(row)
            row = []
            count = 0
        else:
            for c in range(i[0]):
                row.append(0)
                count += 1
            row.append(i[2])
            count += 1

    return zigzag


def get_freq_dict(array: list) -> dict:
    """
    returns a dict where the keys are the values of the array, and the values are their frequencies
    :param numpy.ndarray array: intermediary stream as array
    :return: frequency table
    """
    #
    data = Counter(array)
    result = {k: d / len(array) for k, d in data.items()}
    return result


def find_huffman_bitarray(p: dict) -> dict:
    """
    returns a Huffman code for an ensemble with distribution p
    :param dict p: frequency table
    :returns: huffman code for each symbol
    """
    # Base case of empty dict.
    if len(p) == 0:
        return None

    # Base case of empty dict.
    if len(p) == 1:
        return {p.keys(): bitarray("0")}

    # Base case of only two symbols, assign 0 or 1 arbitrarily; frequency does not matter
    if len(p) == 2:
        return dict(zip(p.keys(), [bitarray("0"), bitarray("1")]))

    # Create a new distribution by merging lowest probable pair
    p_prime = p.copy()
    a1, a2 = lowest_prob_pair(p)
    p1, p2 = p_prime.pop(a1), p_prime.pop(a2)
    p_prime[a1 + a2] = p1 + p2

    # Recurse and construct code on new distribution
    c = find_huffman_bitarray(p_prime)
    ca1a2 = c.pop(a1 + a2)
    c[a1], c[a2] = ca1a2 + bitarray("0"), ca1a2 + bitarray("1")

    return c


def lowest_prob_pair(p):
    # Return pair of symbols from distribution p with lowest probabilities
    sorted_p = sorted(p.items(), key=lambda x: x[1])
    return sorted_p[0][0], sorted_p[1][0]


def zigzag_single(block):
    """
    ZigZag scan over a 8x8 2D array into a 64-element 1D array.
    Args:
        numpy.ndarray: 8x8 2D array
    Returns:
        numpy.ndarray: 64-element 1D array
    """
    return block.flatten()[ZIGZAGFLAT]


def inverse_zigzag_single(array):
    """
    Inverse ZigZag scan over 64-element 1D array into a 8x8 2D array.
    Args:
        numpy.ndarray: 64-element 1D array
    Returns:
        numpy.ndarray: 8x8 2D array
    """
    return array[ZIGZAGFLATINVERSE].reshape([8, 8])


def downsample_avg(arr):
    original_width = arr.shape[1]
    original_height = arr.shape[0]
    width = original_width // 2
    height = original_height // 2

    resized_image = np.zeros(shape=(height, width), dtype=np.int16)
    scale = 2

    for i in range(0, original_height, scale):
        for j in range(0, original_width, scale):
            resized_image[i // scale, j // scale] = np.mean(
                arr[i : i + scale, j : j + scale], axis=(0, 1)
            )

    return resized_image


def find_match(target_block, reference_frame, block_num, block_size=BLOCK_SIZE):
    frame_v = reference_frame.shape[0]
    frame_h = reference_frame.shape[1]

    target_block_ds1 = downsample_avg(target_block)
    target_block_ds2 = downsample_avg(target_block_ds1)

    reference_frame_ds1 = downsample_avg(reference_frame)
    reference_frame_ds2 = downsample_avg(reference_frame_ds1)

    cur_block_size = block_size // 4
    # v_blocks = len(reference_frame) // block_size
    h_blocks = len(reference_frame[0]) // block_size

    v = (block_num // h_blocks) * block_size
    v_ds2 = v // 4
    v_min_ds2 = max(v_ds2 - DEPTH, 0)
    v_max_ds2 = min(frame_v // 4 - 1 - cur_block_size, v_ds2 + DEPTH)

    h = (block_num % h_blocks) * block_size
    h_ds2 = h // 4
    h_min_ds2 = max(h_ds2 - DEPTH, 0)
    h_max_ds2 = min(frame_h // 4 - 1 - cur_block_size, h_ds2 + DEPTH)

    idx_initial = np.array([v, h])

    min_mse = inf
    idx = []
    for i in range(v_min_ds2, v_max_ds2):
        for j in range(h_min_ds2, h_max_ds2):
            mse = mean_square_error(
                target_block_ds2,
                reference_frame_ds2[
                    i : i + cur_block_size,
                    j : j + cur_block_size,
                ],
            )
            if mse < min_mse:
                min_mse = mse
                idx = [i, j]

    if min_mse > MAX_MSE_FOR_PREDICTION:
        # TODO ??
        return None

    cur_block_size = block_size // 2
    idx[0] = idx[0] * 2
    idx[1] = idx[1] * 2
    min_mse = inf
    v_min_ds1 = max(idx[0] - DEPTH, 0)
    v_max_ds1 = min(frame_v // 2 - 1 - cur_block_size, idx[0] + DEPTH)
    h_min_ds1 = max(idx[1] - DEPTH, 0)
    h_max_ds1 = min(frame_h // 2 - 1 - cur_block_size, idx[1] + DEPTH)
    for i in range(v_min_ds1, v_max_ds1):
        for j in range(h_min_ds1, h_max_ds1):
            mse = mean_square_error(
                target_block_ds1,
                reference_frame_ds1[
                    i : i + cur_block_size,
                    j : j + cur_block_size,
                ],
            )
            if mse < min_mse:
                min_mse = mse
                idx = [i, j]

    if min_mse > MAX_MSE_FOR_PREDICTION:
        # TODO ??
        return None

    cur_block_size = block_size
    idx[0] = idx[0] * 2
    idx[1] = idx[1] * 2
    min_mse = inf
    v_min = max(idx[0] - DEPTH, 0)
    v_max = min(frame_v - 1 - cur_block_size, idx[0] + DEPTH)
    h_min = max(idx[1] - DEPTH, 0)
    h_max = min(frame_h - 1 - cur_block_size, idx[1] + DEPTH)
    for i in range(v_min, v_max):
        for j in range(h_min, h_max):
            mse = mean_square_error(
                target_block,
                reference_frame[
                    i : i + cur_block_size,
                    j : j + cur_block_size,
                ],
            )
            if mse < min_mse:
                min_mse = mse
                idx = [i, j]
                if mse == 0:
                    break

    return np.subtract(idx, idx_initial)


def motion_compensation(
    target_frame, reference_frame_1, reference_frame_2=None, block_size=BLOCK_SIZE
):
    v_blocks = len(target_frame) // block_size
    h_blocks = len(target_frame[0]) // block_size

    motion_vectors = []

    block_num = 0
    for i in range(v_blocks):
        for j in range(h_blocks):
            target_block = target_frame[
                i * block_size : i * block_size + block_size,
                j * block_size : j * block_size + block_size,
            ]

            motion_vector = find_match(target_block, reference_frame_1, block_num)

            if motion_vector is None and reference_frame_2 is not None:
                motion_vector_2 = find_match(target_block, reference_frame_2, block_num)
                if motion_vector_2 is not None:
                    motion_vector = [FUTURE_REFERENCE_SIGN, motion_vector_2]

            motion_vectors.append(motion_vector)

            block_num += 1

    return motion_vectors
