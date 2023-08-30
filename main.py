from __future__ import annotations

from buffer import FrameBufferV1
from encoder import Encoder

reference_frame_buffer = FrameBufferV1()
B_frame_buffer = FrameBufferV1()

encoder = Encoder(
    reference_frame_buffer=reference_frame_buffer, B_frame_buffer=B_frame_buffer
)


output_bits = encoder.encode("./animation_frames/cat_enumerated/")
