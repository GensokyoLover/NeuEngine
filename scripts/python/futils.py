import numpy as np

def f3_to_numpy(x):
    return np.array([x.x,x.y,x.z])


def normalize(x):
    x = np.array(x)
    x = x / np.linalg.norm(x)
    return x