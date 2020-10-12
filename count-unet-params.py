def unet_nparams(enc, dec):
    n = 0
    for i, (ks, f) in enumerate(enc):
        if i == 0:
            in_chan = 1
        else:
            in_chan = enc[i-1][1]
        n += ks * f * in_chan
    for i, (ks, f) in enumerate(dec):
        if i == 0:
            in_skip = 0
            in_prev = enc[-1][1]
        else:
            in_skip = enc[i-1][1]
            in_prev = dec[i-1][1]
        n += ks * f * (in_prev + in_skip)
    return n * 2 # complex ops

print("DCUNet-20", unet_nparams([
    (7*1, 32),
    (7*1, 32),
    (7*5, 64), (7*5, 64),
    (5*3, 64), (5*3, 64), (5*3, 64), (5*3, 64), (5*3, 64),
    (5*3, 90),
], [
    (5*3, 64), (5*3, 64), (5*3, 64), (5*3, 64), (5*3, 64),
    (7*5, 64),
    (7*5, 64),
    (7*5, 32),
    (7*1, 32),
    (7*1, 1),
]) / 1e6)
