from typing import Tuple


def get_limit_offset(item_slice: slice, seq_len: int) -> Tuple:
    start, stop, stride = item_slice.indices(seq_len)
    offset = start - 1
    limit = stop - start + 1
    return limit, offset
