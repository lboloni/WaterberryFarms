import numpy as np
import scipy

# Miscellaneous helper functions

def shifted_add(a, b, x, y):
    """Add matrix b to matrix a shifted by positions x and y. 
    Ignore overflow at the margins.
    """
    xA, yA = a.shape
    xB, yB = b.shape

    fromAX = max(0, x)
    toAX = min(xB + x, xA)
    fromAY = max(0, y)
    toAY = min(yB + y, yA)

    fromBX = 0 + max(0, -x)
    toBX = xB + min(0, xA - x - xB)  
    fromBY = 0 + max(0, -y)
    toBY = yB + min(0, yA - y - yB)

    if toBX <= 0 or toBY <= 0:
        return
    if fromBX >= xB or fromBY >= yB:
        return

    a[fromAX:toAX, fromAY:toAY] = a[fromAX:toAX, fromAY:toAY] + b[fromBX:toBX, fromBY:toBY]
    return

def create_gaussian_hump(width, height, mean = [0,0], cov = [[1, 0], [0, 1]]):
    """Fill an array of size width times height with an Gaussian function of 
    the specified mean and covariance matrix."""
    xv = np.linspace(0, width, width)
    yv = np.linspace(0, height, height)
    x, y = np.meshgrid(xv, yv)
    positions = np.column_stack((x.ravel(),y.ravel()))
    value = scipy.stats.multivariate_normal.pdf(positions, mean, cov)
    vals = np.reshape(value, [height, width])
    return vals