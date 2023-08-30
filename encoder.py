import os
from dataclasses import dataclass

import sys
import json
from bitarray import bitarray
from matplotlib import pyplot as plt

from buffer import FrameBuffer
from functions import *
import numpy as np


QT = np.array(
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

BLOCK_SIZE = 8


@dataclass
class Encoder:
    reference_frame_buffer: FrameBuffer
    B_frame_buffer: FrameBuffer

    def encode(self, path) -> bitarray:
        # out: bitarray = bitarray([])
        out = 0
        input_len: int = 0

        for frame_count, frame_filename in enumerate(sorted(os.listdir(path))):
            try:
                frame_filename_noext = frame_filename.split(".")[0]
                print(f"processing {frame_filename}")
                if int(frame_filename_noext) != frame_count:
                    raise AssertionError("missing frame")
            except ValueError:
                raise AssertionError("frame_filename is not in correct format")

            frame_count_mod = frame_count % 9

            frame = cv2.imread(path + frame_filename, cv2.IMREAD_GRAYSCALE)
            # normalise values around 0
            frame = np.float32(frame) - 128

            input_len += frame.shape[0] * frame.shape[1] * 8

            if frame_count_mod in [0, 3, 6]:
                (
                    rle,
                    huffman,
                    intra_encoded_blocks,
                    motion_vectors_diff,
                    decoded_frame,
                ) = self.encode_frame(frame, frame_count_mod)
                self.reference_frame_buffer.enqueue(decoded_frame)

                frame_size = 0
                frame_size += len(rle)
                frame_size += sys.getsizeof(huffman) * 8
                if intra_encoded_blocks is not None:
                    frame_size += len(intra_encoded_blocks)
                    frame_size += len(motion_vectors_diff) * 8 * 2
                    print(
                        f"frame size {len(rle), sys.getsizeof(huffman) * 8, len(intra_encoded_blocks),len(motion_vectors_diff) * 8 * 2 + 3, frame_size}"
                    )
                out += frame_size
                print(f"frame size {frame_size}")

                plt.imshow(decoded_frame, cmap="gray")
                plt.show()

                # TODO add current frame to output stream
                # out.extend(frame_bits)
                # out.extend(huffman_bits)
                if not self.B_frame_buffer.empty():
                    (
                        rle,
                        huffman,
                        intra_encoded_blocks,
                        motion_vectors_diff,
                        decoded_frame,
                    ) = self.encode_frame(
                        self.B_frame_buffer.dequeue(), frame_count_mod - 2
                    )
                    # TODO add current frame to output stream
                    frame_size = 0
                    frame_size += len(rle)
                    frame_size += sys.getsizeof(huffman) * 8
                    frame_size += len(intra_encoded_blocks)
                    frame_size += len(motion_vectors_diff) * 8 * 2
                    out += frame_size
                    print(f"frame size {frame_size}")

                    plt.imshow(decoded_frame, cmap="gray")
                    plt.show()

                    (
                        rle,
                        huffman,
                        intra_encoded_blocks,
                        motion_vectors_diff,
                        decoded_frame,
                    ) = self.encode_frame(
                        self.B_frame_buffer.dequeue(), frame_count_mod - 1
                    )
                    # TODO add current frame to output stream

                    frame_size = 0
                    frame_size += len(rle)
                    frame_size += sys.getsizeof(huffman) * 8
                    frame_size += len(intra_encoded_blocks)
                    frame_size += len(motion_vectors_diff) * 8 * 2 + 3
                    out += frame_size
                    print(
                        f"frame size {len(rle),sys.getsizeof(huffman) * 8, len(intra_encoded_blocks),len(motion_vectors_diff) * 8 * 2 + 3, frame_size}"
                    )

                    plt.imshow(decoded_frame, cmap="gray")
                    plt.show()

                    self.reference_frame_buffer.dequeue()
            else:
                self.B_frame_buffer.enqueue(frame)

        print(f"compression ratio is: {input_len/out}")

        return

    def encode_frame(
        self, frame: np.ndarray, frame_count_mod
    ) -> [list, dict, bitarray, list, np.ndarray]:
        if frame_count_mod == 0:
            return self.encode_intra(frame)
        elif frame_count_mod in [3, 6]:
            return self.encode_p_frame(frame)
        else:
            return self.encode_b_frame(frame)

    def encode_intra(
        self, frame: np.ndarray
    ) -> [list, dict, bitarray, list, np.ndarray]:
        v_blocks = len(frame) // BLOCK_SIZE
        h_blocks = len(frame[0]) // BLOCK_SIZE
        decoded = np.zeros(frame.shape)
        zigzag_ordered = np.zeros([v_blocks * h_blocks, BLOCK_SIZE * BLOCK_SIZE])
        block_num = 0
        for i in range(v_blocks):
            for j in range(h_blocks):
                dct_quantized = quantize(dct(frame, i, j), mode="luma")
                decode_I_frame_block(decoded, dct_quantized, i, j)
                zigzag_ordered[block_num] = zigzag_single(dct_quantized)
                block_num += 1
        rle = run_length_encoding(zigzag_ordered)
        huffman = find_huffman_bitarray(get_freq_dict(rle))
        return rle, huffman, None, None, decoded

    def encode_p_frame(self, target_frame) -> [list, dict, bitarray, list, np.ndarray]:
        reference_frame = self.reference_frame_buffer.prev()
        motion_vectors = motion_compensation(target_frame, reference_frame)

        v_blocks = target_frame.shape[0] // BLOCK_SIZE
        h_blocks = target_frame.shape[1] // BLOCK_SIZE

        decoded_frame = np.zeros(target_frame.shape)
        intra_encoded_blocks = bitarray(v_blocks * h_blocks)
        zigzag_ordered = []
        block_num = 0
        for i in range(v_blocks):
            for j in range(h_blocks):
                vector = motion_vectors[block_num]
                if vector is None:
                    intra_encoded_blocks[block_num] = 1

                    coded_block = quantize(dct(target_frame, i, j), mode="luma")

                    zigzag_ordered.append(zigzag_single(coded_block))

                    decoded_frame[
                        i * BLOCK_SIZE : i * BLOCK_SIZE + BLOCK_SIZE,
                        j * BLOCK_SIZE : j * BLOCK_SIZE + BLOCK_SIZE,
                    ] = idct_single(dequantize(coded_block, mode="luma"))
                else:
                    intra_encoded_blocks[block_num] = 0

                    block_vector = [
                        i * BLOCK_SIZE + vector[0],
                        j * BLOCK_SIZE + vector[1],
                    ]

                    decoded_frame[
                        i * BLOCK_SIZE : i * BLOCK_SIZE + BLOCK_SIZE,
                        j * BLOCK_SIZE : j * BLOCK_SIZE + BLOCK_SIZE,
                    ] = get_referenced_block(reference_frame, block_vector)

                block_num += 1

        prev = [0, 0]
        motion_vectors_diff = []
        for v in motion_vectors:
            if v is not None:
                motion_vectors_diff.append([v[0] - prev[0], v[1] - prev[1]])
                prev = v

        rle = run_length_encoding(np.asarray(zigzag_ordered))
        huffman = find_huffman_bitarray(get_freq_dict(rle))

        return rle, huffman, intra_encoded_blocks, motion_vectors_diff, decoded_frame

    def encode_b_frame(self, target_frame) -> [list, dict, bitarray, list, np.ndarray]:
        reference_frame_1 = self.reference_frame_buffer.prev()
        reference_frame_2 = self.reference_frame_buffer.next()
        motion_vectors = motion_compensation(
            target_frame, reference_frame_1, reference_frame_2
        )

        v_blocks = target_frame.shape[0] // BLOCK_SIZE
        h_blocks = target_frame.shape[1] // BLOCK_SIZE

        decoded_frame = np.zeros(target_frame.shape)
        intra_encoded_blocks = bitarray(v_blocks * h_blocks)
        zigzag_ordered = []
        block_num = 0
        for i in range(v_blocks):
            for j in range(h_blocks):
                vector = motion_vectors[block_num]
                if vector is None:
                    intra_encoded_blocks[block_num] = 1

                    coded_block = quantize(dct(target_frame, i, j), mode="luma")

                    zigzag_ordered.append(zigzag_single(coded_block))

                    decoded_frame[
                        i * BLOCK_SIZE : i * BLOCK_SIZE + BLOCK_SIZE,
                        j * BLOCK_SIZE : j * BLOCK_SIZE + BLOCK_SIZE,
                    ] = idct_single(dequantize(coded_block))
                else:
                    intra_encoded_blocks[block_num] = 0

                    if vector[0] == FUTURE_REFERENCE_SIGN:
                        vector = vector[1]
                        reference_frame = reference_frame_2
                    else:
                        reference_frame = reference_frame_1

                    block_vector = [
                        i * BLOCK_SIZE + vector[0],
                        j * BLOCK_SIZE + vector[1],
                    ]

                    decoded_frame[
                        i * BLOCK_SIZE : i * BLOCK_SIZE + BLOCK_SIZE,
                        j * BLOCK_SIZE : j * BLOCK_SIZE + BLOCK_SIZE,
                    ] = get_referenced_block(reference_frame, block_vector)

                block_num += 1

        prev = [0, 0]
        motion_vectors_diff = []
        for v in motion_vectors:
            if v is not None:
                if v[0] == FUTURE_REFERENCE_SIGN:
                    v = v[1]
                    motion_vectors_diff.append(["f", [v[0] - prev[0], v[1] - prev[1]]])
                else:
                    motion_vectors_diff.append([v[0] - prev[0], v[1] - prev[1]])

                prev = v

        rle = run_length_encoding(np.asarray(zigzag_ordered))
        huffman = find_huffman_bitarray(get_freq_dict(rle))

        return rle, huffman, intra_encoded_blocks, motion_vectors_diff, decoded_frame
