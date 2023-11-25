import numpy as np

def rle_decode(mask_rle: str, img_shape: tuple = None) -> np.ndarray:
    seq = mask_rle.split()
    starts = np.array(list(map(int, seq[0::2])))
    lengths = np.array(list(map(int, seq[1::2])))
    assert len(starts) == len(lengths)
    ends = starts + lengths
    img = np.zeros((np.product(img_shape),), dtype=np.uint8)
    for begin, end in zip(starts, ends):
        img[begin:end] = 1
    # https://stackoverflow.com/a/46574906/4521646
    img.shape = img_shape
    mask = np.zeros((img_shape[0], img_shape[1], 2), dtype=np.uint8)
    mask[..., 1] = img
    return mask

def rle_encode(mask, bg = 0) -> dict:
    vec = mask.flatten()
    nb = len(vec)
    where = np.flatnonzero
    starts = np.r_[0, where(~np.isclose(vec[1:], vec[:-1], equal_nan=True)) + 1]
    lengths = np.diff(np.r_[starts, nb])
    values = vec[starts]
    assert len(starts) == len(lengths) == len(values)
    rle = []
    for start, length, val in zip(starts, lengths, values):
        if val == bg:
            continue
        rle += [str(start), length]
    # post-processing
    return " ".join(map(str, rle))