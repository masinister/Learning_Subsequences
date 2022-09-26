def bin_to_int(a):
    res = 0
    for b in a:
        res = (res << 1) | b
    return res
