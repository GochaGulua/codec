from dataclasses import dataclass, field
from typing import Optional, Protocol

from numpy import ndarray


class FrameBuffer(Protocol):
    def prev(self) -> ndarray:
        pass

    def next(self) -> ndarray:
        pass

    def enqueue(self, frame: ndarray) -> None:
        pass

    def dequeue(self) -> ndarray:
        pass

    def empty(self) -> bool:
        pass


@dataclass
class FrameBufferV1:
    pre: Optional[ndarray] = field(default=None)
    nxt: Optional[ndarray] = field(default=None)

    def prev(self) -> ndarray:
        if self.pre is None:
            raise IndexError
        return self.pre

    def next(self) -> ndarray:
        if self.nxt is None:
            raise IndexError
        return self.nxt

    def enqueue(self, frame: ndarray) -> None:
        if self.pre is None:
            self.pre = frame
        elif self.nxt is None:
            self.nxt = frame
        else:
            self.pre = self.nxt
            self.nxt = frame

    def dequeue(self) -> ndarray:
        if self.pre is None:
            raise IndexError("buffer in empty")
        else:
            ret = self.pre
            if self.nxt is not None:
                self.pre = self.nxt
                self.nxt = None
            return ret

    def empty(self) -> bool:
        return self.pre is None
