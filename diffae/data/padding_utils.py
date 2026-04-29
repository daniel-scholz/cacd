from typing import Literal

import numpy as np


def to_closest_divider(size_: int, divider: int, mode: Literal["small", "big", "closest"]) -> int:
    small = size_ // divider * divider
    large = small + divider
    match mode:
        case "small":
            return small
        case "big":
            return large
        case "closest":
            new_size = small if abs(size_ - small) < abs(size_ - large) else large
    return new_size


# to closest power of 2 below or above
def to_closest_pow2(size_: int, mode: Literal["small", "big", "closest"]) -> int:
    # print("min", x_["img"].min())
    small2 = 2 ** int(np.log2(size_))
    large2 = 2 ** (int(np.log2(size_)) + 1)
    match mode:
        case "small":
            return small2
        case "big":
            return large2
        case "closest":
            new_size = small2 if abs(size_ - small2) < abs(size_ - large2) else large2
    # print(new_size)
    return new_size
