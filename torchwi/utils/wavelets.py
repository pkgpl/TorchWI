import numpy as np

def ricker(freq, dt, nt, dtype=np.float64):
    a = np.zeros(nt, dtype=dtype)
    vm = freq / 2. / 1.08
    vm2 = vm ** 2
    pi2 = np.pi ** 2
    for i in range(nt):
        t = i * dt
        arg = pi2 * t ** 2
        a[i] = (1. - 2. * vm2 * arg) * np.exp(-vm2 * arg)
    icut = 0
    for i in range(nt):
        if (np.abs(a[i]) < a[0] * 1.e-3):
            icut = i
            break
    t0 = icut * dt
    for i in range(nt):
        t = i * dt
        t = t0 - t
        arg = pi2 * t ** 2
        a[i] = (1. - 2. * vm2 * arg) * np.exp(-vm2 * arg)
    return a


def fdgaus(cutoff, dt, nt, dtype=np.float64):
    w = np.zeros(nt, dtype=dtype)
    a = np.pi * (5. * cutoff / 8.) ** 2
    amp = np.sqrt(a / np.pi)
    for i in range(nt):
        t = i * dt
        arg = -a * t ** 2
        if arg < -32.:
            arg = -32.
        w[i] = amp * np.exp(arg)
    t0 = 0.
    threshold = 0.001 * w[0]
    for i in range(nt):
        if w[i] < threshold:
            icut = i
            t0 = (icut - 1) * dt
            break
    sqpi = np.sqrt(np.pi)
    for i in range(nt):
        t = i * dt
        t = t - t0
        arg = -a * t ** 2
        if arg < -32.:
            arg = -32.
        w[i] = -2. * np.sqrt(a) * a * t * np.exp(arg) / sqpi
    w /= w.max()
    return w
