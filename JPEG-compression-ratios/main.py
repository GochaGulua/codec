# python 3.9.5

import sys
import time
from math import ceil
import matplotlib.pyplot as plt

from functions import *


CHANNEL_SPLITTER = b"@channel_splitter@"

# define quantization tables
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
        [17, 18, 24, 47, inf, inf, inf, inf],  # chrominance quantization table
        [18, 21, 26, 66, inf, inf, inf, inf],
        [24, 26, 56, inf, inf, inf, inf, inf],
        [47, 66, inf, inf, inf, inf, inf, inf],
        [inf, inf, inf, inf, inf, inf, inf, inf],
        [inf, inf, inf, inf, inf, inf, inf, inf],
        [inf, inf, inf, inf, inf, inf, inf, inf],
        [inf, inf, inf, inf, inf, inf, inf, inf],
    ]
)
# define window size
windowSize = len(QTY)

t0 = time.time()
# read image
imgOriginal = cv2.imread("img.tif", cv2.IMREAD_COLOR)

# convert BGR to YCrCb
img = cv2.cvtColor(imgOriginal, cv2.COLOR_BGR2YCR_CB)
rgb = cv2.cvtColor(img, cv2.COLOR_YCR_CB2RGB)
width = len(img[0])
height = len(img)
y = np.zeros((height, width), np.float32) + img[:, :, 0]
cr = np.zeros((height, width), np.float32) + img[:, :, 1]
cb = np.zeros((height, width), np.float32) + img[:, :, 2]

plt.imshow(y, cmap="gray")
plt.show()

# size of the image in bits before compression
totalNumberOfBitsWithoutCompression = (y.size + cr.size + cb.size) * 8
# channel values should be normalized, hence subtract 128
y = y - 128
cr = cr - 128
cb = cb - 128

SSH, SSV = 2, 2
crf = cv2.boxFilter(cr, ddepth=-1, ksize=(2, 2))
cbf = cv2.boxFilter(cb, ddepth=-1, ksize=(2, 2))
crSub = crf[::SSV, ::SSH]
cbSub = cbf[::SSV, ::SSH]

ycrcb = np.dstack((y, crf, cbf))

rgb_image_data1 = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB).astype(np.int16) + 128
plt.imshow(rgb_image_data1, vmin=0, vmax=256)
plt.show()

# check if padding is needed,
# if yes define empty arrays to pad each channel DCT with zeros if necessary
yWidth, yLength = (
    ceil(len(y[0]) / windowSize) * windowSize,
    ceil(len(y) / windowSize) * windowSize,
)

yPadded = pad(y, yLength, yWidth)

# chrominance channels have the same dimensions, meaning both can be padded in one loop
cWidth, cLength = (
    ceil(len(cbSub[0]) / windowSize) * windowSize,
    ceil(len(cbSub) / windowSize) * windowSize,
)

crPadded, cbPadded = pad2(crSub, cbSub, cLength, cWidth)


t1 = time.time()
diff = t1 - t0
print(f"arrays initialized. %.2fs elapsed" % diff)

v_blocks = len(yPadded) // windowSize
h_blocks = len(yPadded[0]) // windowSize
yZigzag = np.zeros([v_blocks * h_blocks, windowSize * windowSize])
block_num = 0
for i in range(v_blocks):
    for j in range(h_blocks):
        yZigzag[block_num] = zigzag_single(quantize(dct(yPadded, i, j), mode="luma"))
        block_num += 1

yZigzag = yZigzag.astype(np.int16)

# plt.imshow(yPadded_rev, cmap="gray", vmin=-128, vmax=128)
# plt.show()

t2 = time.time()
diff = t2 - t1
print(f"luma array formatted. %.2fs elapsed" % diff)

v_blocks = len(crPadded) // windowSize
h_blocks = len(crPadded[0]) // windowSize
crZigzag = np.zeros([v_blocks * h_blocks, windowSize * windowSize])
cbZigzag = np.zeros([v_blocks * h_blocks, windowSize * windowSize])
block_num = 0
for i in range(v_blocks):
    for j in range(h_blocks):
        cbZigzag[block_num] = zigzag_single(
            quantize(dct(cbPadded, i, j), mode="chroma")
        )
        crZigzag[block_num] = zigzag_single(
            quantize(dct(crPadded, i, j), mode="chroma")
        )
        block_num += 1

crZigzag = crZigzag.astype(np.int16)
cbZigzag = cbZigzag.astype(np.int16)

t3 = time.time()
diff = t3 - t2
print(f"chroma arrays formatted. %.2fs elapsed" % diff)

yEncoded = run_length_encoding(yZigzag)
crEncoded = run_length_encoding(crZigzag)
cbEncoded = run_length_encoding(cbZigzag)

t4 = time.time()
diff = t4 - t3
print(f"rle finished. %.2fs elapsed" % diff)

yFrequencyTable = get_freq_dict(yEncoded)
crFrequencyTable = get_freq_dict(crEncoded)
cbFrequencyTable = get_freq_dict(cbEncoded)

t5 = time.time()
diff = t5 - t4
print(f"frequency tables calculated. %.2fs elapsed" % diff)

yHuffman_bitarray = find_huffman_bitarray(yFrequencyTable)
crHuffman_bitarray = find_huffman_bitarray(crFrequencyTable)
cbHuffman_bitarray = find_huffman_bitarray(cbFrequencyTable)

t6 = time.time()
diff = t6 - t5
print(f"huffman dictionarys built. %.2fs elapsed" % diff)

# calculate the number of bits to transmit for each channel
# and write them to an output file
file = open("CompressedImage.asfh", "wb")

a_start = time.time()
yBitsToTransmit = bitarray()
yBitsToTransmit.encode(yHuffman_bitarray, yEncoded)
a_finish = time.time()

crBitsToTransmit = bitarray()
crBitsToTransmit.encode(crHuffman_bitarray, crEncoded)

cbBitsToTransmit = bitarray()
cbBitsToTransmit.encode(cbHuffman_bitarray, cbEncoded)

t7 = time.time()
diff = t7 - t6
print(f"converted to bits. %.2fs elapsed" % diff)

if file.writable():
    file.write(bytes(yBitsToTransmit))
    file.write(CHANNEL_SPLITTER)
    file.write(bytes(crBitsToTransmit))
    file.write(CHANNEL_SPLITTER)
    file.write(bytes(cbBitsToTransmit))
file.close()

t = time.time()
diff = t - t0
print(f"total time to compress and write: %.2fs" % diff)


totalNumberOfBitsAfterCompression = (
    len(yBitsToTransmit)
    + len(crBitsToTransmit)
    + len(cbBitsToTransmit)
    + (
        sys.getsizeof(cbHuffman_bitarray)
        + sys.getsizeof(crHuffman_bitarray)
        + sys.getsizeof(yHuffman_bitarray)
    )
    * 8
)

print(
    "Compression Ratio is "
    + str(
        np.round(
            totalNumberOfBitsWithoutCompression / totalNumberOfBitsAfterCompression, 1
        )
    )
)


t7 = time.time()
print(f"opening.")

with open("CompressedImage.asfh", "rb") as f:
    y_bytes, cr_bytes, cb_bytes = f.read().split(CHANNEL_SPLITTER)
    y_bits = bitarray()
    [y_bits.extend(format(byte, "08b")) for byte in y_bytes]
    if len(y_bits) != len(yBitsToTransmit):
        y_bits = y_bits[: -(len(y_bits) - len(yBitsToTransmit))]
    yDecoded = y_bits.decode(yHuffman_bitarray)
    cr_bits = bitarray()
    [cr_bits.extend(format(byte, "08b")) for byte in cr_bytes]
    if len(cr_bits) != len(crBitsToTransmit):
        cr_bits = cr_bits[: -(len(cr_bits) - len(crBitsToTransmit))]
    crDecoded = cr_bits.decode(crHuffman_bitarray)
    cb_bits = bitarray()
    [cb_bits.extend(format(byte, "08b")) for byte in cb_bytes]
    if len(cb_bits) != len(cbBitsToTransmit):
        cb_bits = cb_bits[: -(len(cb_bits) - len(cbBitsToTransmit))]
    cbDecoded = cb_bits.decode(cbHuffman_bitarray)

    y_unrle = np.asarray(unrle(yDecoded))
    cb_unrle = np.asarray(unrle(cbDecoded))
    cr_unrle = np.asarray(unrle(crDecoded))

    y_dezigzag = np.zeros((yLength, yWidth))
    cb_dezigzag = np.zeros((cLength, cWidth))
    cr_dezigzag = np.zeros((cLength, cWidth))

    v_blocks = int(np.floor(len(yPadded) / windowSize))
    h_blocks = int(np.floor(len(yPadded[0]) / windowSize))
    y_new = np.zeros((yLength, yWidth))
    block_num = 0
    for i in range(v_blocks):
        for j in range(h_blocks):
            dezigzag_dequantize_idct(y_new, y_unrle, i, j, block_num)
            block_num += 1

    v_blocks = int(np.floor(len(crPadded) / windowSize))
    h_blocks = int(np.floor(len(crPadded[0]) / windowSize))
    cr_new = np.zeros((cLength, cWidth))
    cb_new = np.zeros((cLength, cWidth))
    block_num = 0
    for i in range(v_blocks):
        for j in range(h_blocks):
            dezigzag_dequantize_idct(cr_new, cr_unrle, i, j, block_num, mode="chroma")
            dezigzag_dequantize_idct(cb_new, cb_unrle, i, j, block_num, mode="chroma")
            block_num += 1

    y_new += 128
    imgplot = plt.imshow(y_new, cmap="gray", vmin=0, vmax=256)
    plt.show()

    cr_new = np.repeat(np.repeat(cr_new, 2, axis=0), 2, axis=1)
    cr_new += 128

    cb_new = np.repeat(np.repeat(cb_new, 2, axis=0), 2, axis=1)
    cb_new += 128

    if y_new.shape != cr_new.shape:
        v = y_new.shape[0]
        h = y_new.shape[1]
        cr_new = cr_new[:v, 0:h]
        cb_new = cb_new[:v, 0:h]

    ycrcb = np.dstack((y_new, cr_new, cb_new))

    # rgb_image_data = cv2.cvtColor(ycrcb, cv2.COLOR_YCrCb2RGB)
    rgb_image_data = ycbcr2rgb(ycrcb)
    plt.imshow(rgb_image_data)
    plt.show()

t8 = time.time()
diff = t8 - t7
print(f"opened. %.2fs elapsed" % diff)

t8 = time.time()
diff = t8 - t0
print(f"total: %.2fs elapsed" % diff)

yPadded += 128
print(f"PSNR of Y channel: %.2f db" % PSNR(yPadded, y_new))

cr, cb = pad2(cr, cb, yLength, yWidth)
cr += 128
cb += 128
print(f"PSNR of Cr channel: %.2f db" % PSNR(cr, cr_new))
print(f"PSNR of Cb channel: %.2f db" % PSNR(cb, cb_new))
