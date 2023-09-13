import csv
import time
from array import array
from math import ceil
from random import random

import cv2
import numpy as np
from matplotlib import pyplot as plt
from numpy import ones

from buffer import FrameBufferV1
from encoder import Encoder
from functions import *

QTY = np.array(
    [
        [16, 11, 10, 16, 24, 40, 51, 61],
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
        [17, 18, 24, 47, inf, inf, inf, inf],
        [18, 21, 26, 66, inf, inf, inf, inf],
        [24, 26, 56, inf, inf, inf, inf, inf],
        [47, 66, inf, inf, inf, inf, inf, inf],
        [inf, inf, inf, inf, inf, inf, inf, inf],
        [inf, inf, inf, inf, inf, inf, inf, inf],
        [inf, inf, inf, inf, inf, inf, inf, inf],
        [inf, inf, inf, inf, inf, inf, inf, inf],
    ]
)

buf = FrameBufferV1()
buf_1 = FrameBufferV1()
buf_2 = FrameBufferV1()
encoder = Encoder(buf_1, buf_2)

BLOCK_SIZE = 8
frame = np.asarray(list(csv.reader(open("./cbPadded.csv", "r")))).astype(float)


def test_b_frame_enc_dec():
    frame_1 = np.asarray(
        cv2.imread(
            "../animation_frames/cat/frame_0_delay-0.5s.png", cv2.IMREAD_GRAYSCALE
        )
    ).astype(float)

    yWidth, yLength = (
        ceil(len(frame_1[0]) / BLOCK_SIZE) * BLOCK_SIZE,
        ceil(len(frame_1) / BLOCK_SIZE) * BLOCK_SIZE,
    )

    frame_1 = pad(frame_1, BLOCK_SIZE, yLength, yWidth)

    plt.imshow(frame_1, cmap="gray")
    plt.show()

    frame_2 = frame_1.copy()
    frame_1 = frame_1 - 128

    rle, hufman, decoded = encoder.encode_intra(frame_1)

    decoded = decoded.astype(np.int16) + 128

    plt.imshow(decoded, cmap="gray", vmin=0, vmax=256)
    plt.show()

    buf_1.enqueue(decoded)
    buf_1.enqueue(decoded)

    plt.imshow(frame_2, cmap="gray", vmin=0, vmax=256)
    plt.show()

    (
        rle,
        huffman,
        intra_encoded_blocks,
        motion_vectors_diff,
        decoded_frame,
    ) = encoder.encode_b_frame(frame_2)

    decoded_frame += 256
    plt.imshow(decoded_frame, cmap="gray")
    bla = 0
    plt.show()


def test_p_frame_enc_dec():
    frame_1 = np.asarray(
        cv2.imread(
            "../animation_frames/skyline/0.png",
            cv2.IMREAD_GRAYSCALE
            # "../animation_frames/skyline/frame_2_delay-0.1s.png", cv2.IMREAD_GRAYSCALE
        )
    ).astype(float)

    yWidth, yLength = (
        ceil(len(frame_1[0]) / BLOCK_SIZE) * BLOCK_SIZE,
        ceil(len(frame_1) / BLOCK_SIZE) * BLOCK_SIZE,
    )

    frame_1 = pad(frame_1, BLOCK_SIZE, yLength, yWidth)

    plt.imshow(frame_1, cmap="gray")
    plt.show()

    frame_1 = frame_1 - 128

    rle, hufman, decoded = encoder.encode_intra(frame_1)

    decoded = decoded.astype(np.int16) + 128

    plt.imshow(decoded, cmap="gray", vmin=0, vmax=256)
    plt.show()

    buf_1.enqueue(frame_1)

    frame_2 = np.asarray(
        cv2.imread(
            "./marbles.bmp",
            cv2.IMREAD_GRAYSCALE
            # "../animation_frames/skyline/frame_3_delay-0.1s.png", cv2.IMREAD_GRAYSCALE
        )
    ).astype(float)

    frame_2 = frame_1.copy()
    frame_2 = frame_2 + 128

    plt.imshow(frame_2, cmap="gray", vmin=0, vmax=256)
    plt.show()

    frame_2 = frame_2 - 128

    (
        rle,
        huffman,
        intra_encoded_blocks,
        motion_vectors_diff,
        decoded_frame,
    ) = encoder.encode_p_frame(frame_2)

    decoded_frame += 128
    plt.imshow(decoded_frame, cmap="gray", vmin=0, vmax=256)
    bla = 0
    plt.show()


def test_find_match_no_motion():
    h = 10
    v = 10

    random_block = frame[
        v * BLOCK_SIZE : v * BLOCK_SIZE + BLOCK_SIZE,
        h * BLOCK_SIZE : h * BLOCK_SIZE + BLOCK_SIZE,
    ]

    h_blocks = len(frame[0]) // BLOCK_SIZE
    block_num = h_blocks * v + h
    assert False not in (
        find_match(random_block, frame, block_num, BLOCK_SIZE) == np.array([0, 0])
    )


def find_match_with_motion(v=17, h=15, v_motion=4, h_motion=-20, frame=frame):
    random_block = frame[
        v * BLOCK_SIZE + v_motion : v * BLOCK_SIZE + BLOCK_SIZE + v_motion,
        h * BLOCK_SIZE + h_motion : h * BLOCK_SIZE + BLOCK_SIZE + h_motion,
    ]

    h_blocks = len(frame[0]) // BLOCK_SIZE
    block_num = h_blocks * v + h
    return find_match(random_block, frame, block_num, BLOCK_SIZE)


def test_find_match_with_motion():
    v = 21
    h = 86

    v_motion = 12
    h_motion = -13

    assert False not in (
        find_match_with_motion(v, h, v_motion, h_motion, frame)
        == np.array([v_motion, h_motion])
    )


def random_sgn():
    if random() < 0.5:
        return 1
    else:
        return -1


def motion_compensation_success_rate(sample_size=20):
    print(frame.shape)
    success = 0
    for i in range(sample_size):
        v = int(random() * (frame.shape[0] - 8) // 8)
        h = int(random() * (frame.shape[1] - 8) // 8)

        v_motion = min(
            max(random_sgn() * int(random() * 32), -v), frame.shape[0] - v * 8 - 8
        )
        h_motion = min(
            max(random_sgn() * int(random() * 32), -h), frame.shape[1] - h * 8 - 8
        )
        print(v, h, v_motion, h_motion)

        motion_vector = find_match_with_motion(v, h, v_motion, h_motion, frame)
        if False not in (motion_vector == np.array([v_motion, h_motion])):
            success += 1
        else:
            block = frame[
                v * BLOCK_SIZE + v_motion : v * BLOCK_SIZE + BLOCK_SIZE + v_motion,
                h * BLOCK_SIZE + h_motion : h * BLOCK_SIZE + BLOCK_SIZE + h_motion,
            ]
            print(motion_vector)
            predicted_block = frame[
                v * BLOCK_SIZE
                + motion_vector[0] : v * BLOCK_SIZE
                + BLOCK_SIZE
                + motion_vector[0],
                h * BLOCK_SIZE
                + motion_vector[1] : h * BLOCK_SIZE
                + BLOCK_SIZE
                + motion_vector[1],
            ]

            mse = mean_square_error(block, predicted_block)

            if mse < 5:
                success += 1
            else:
                print(mse)

    success_rate = success / sample_size
    print(f"success rate: {success_rate}")

    return success_rate


def test_motion_compensation_success_rate():
    assert motion_compensation_success_rate(1000) > 0.5


def test_dct_idct():
    print(frame.shape)
    v_blocks = len(frame) // BLOCK_SIZE
    h_blocks = len(frame[0]) // BLOCK_SIZE
    cbZigzag = np.zeros([v_blocks * h_blocks, BLOCK_SIZE * BLOCK_SIZE])
    cbDecoded = np.zeros(frame.shape)
    block_num = 0
    for i in range(v_blocks):
        for j in range(h_blocks):
            cbZigzag[block_num] = zigzag_single(dct(frame, i, j))

            cbDecoded[
                i * BLOCK_SIZE : (i + 1) * BLOCK_SIZE,
                j * BLOCK_SIZE : (j + 1) * BLOCK_SIZE,
            ] = idct_single(inverse_zigzag_single(cbZigzag[block_num]))

            block_num += 1
    assert np.allclose(frame, cbDecoded)


def test_frame_buffer():
    buf.enqueue(ones((1, 10)))
    assert buf.prev().shape == (1, 10)
    buf.enqueue(ones((1, 20)))
    assert buf.next().shape == (1, 20)

    first_deq = buf.dequeue()
    assert first_deq.shape == (1, 10)

    assert buf.prev().shape == (1, 20)

    assert buf.nxt is None


def test_intra_enc_dec():
    reference_frame_buffer = FrameBufferV1()
    B_frame_buffer = FrameBufferV1()

    encoder = Encoder(
        reference_frame_buffer=reference_frame_buffer, B_frame_buffer=B_frame_buffer
    )

    frame = np.asarray(
        cv2.imread(
            "../animation_frames/sea/frame_00_delay-0.08s.png", cv2.IMREAD_GRAYSCALE
        )
    ).astype(np.float32)

    w, l = (
        ceil(len(frame[0]) / BLOCK_SIZE) * BLOCK_SIZE,
        ceil(len(frame) / BLOCK_SIZE) * BLOCK_SIZE,
    )

    frame = pad(frame, BLOCK_SIZE, l, w) - 128

    rle_bytes, huffman_bytes, decoded = encoder.encode_intra(frame)

    frame += 128
    plt.imshow(frame, cmap="gray", vmin=0, vmax=256)
    plt.show()

    decoded += 128
    plt.imshow(decoded, cmap="gray", vmin=0, vmax=256)
    plt.show()

    diff = np.subtract(decoded, frame)
    plt.imshow(diff**2, cmap="gray", vmin=0, vmax=256**2)
    plt.show()
    assert PSNR(frame, decoded) > 25
