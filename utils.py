import numpy as np
from numba import jit


def linear(x, y):
    return(np.column_stack((x, y)))


def swirl(x, y):
    r = 3 + np.sqrt(x * x + y * y)
    return(np.column_stack((x * np.sin(r * r) - y * np.cos(r * r),
                            x * np.cos(r * r) + y * np.sin(r * r))))


def spherical(x, y):
    omega = np.arctan((x + 1) / (y + 1))
    r = np.sqrt(x * x + y * y)
    return(np.column_stack((np.sin(np.pi * r) * omega / np.pi,
                            np.cos(np.pi * r) * omega / np.pi)))


def expinj(x, y):
    xx = np.exp(-x * x)
    yy = np.exp(-y * y)
    return(np.column_stack((xx, yy)))


def pdj(x, y, p1=.7, p2=3.14, p3=.7, p4=.2):
    xx = np.sin(p1 * y) - np.cos(p2 * x)
    yy = np.sin(p3 * x) - np.cos(p4 * y)
    return(np.column_stack((xx, yy)))


def bubble(x, y):
    r = np.sqrt(x * x + y * y)
    coef = 4 / (r * r + 4)
    return(np.column_stack((coef * x, coef * y)))


def rotation(ncuts, angle, resF, r, coef=1):
    """ Util function to apply a rotation several times.

    Parameters :
    - ncuts is the number of times you want to apply the rotation
    - angle is the angle of the rotation.
    - resF is a batch of points, np.array of size (N,2)
    - r is a np.array of rands, size (N,1)
    - coef is a float

    for clean rotations, ncuts time angles should make a 360 angle.
    for funny things, feel free to enter other values.
    for decreasing or increasing rotations, change coef.
    """
    rot = np.matrix(
        [[np.cos(angle), - np.sin(angle)], [np.sin(angle), np.cos(angle)]])
    for i in range(ncuts):
        cut = (i + 1) / ncuts
        sel = np.where(r < cut)[0]
        resF[sel, :] = coef * np.dot(resF[sel, :], rot)
    return(resF)


@jit
def renderImage(F_loc, C, bitmap, intensity, goods, coef_forget):
    ''' this renders the image with a jit compilation
    '''
    for i in goods:
        ad0 = F_loc[i, 0]
        ad1 = F_loc[i, 1]
        sto = bitmap[ad0, ad1]
        a = (C[i, :] * coef_forget + sto) / (coef_forget + 1)
        bitmap[ad0, ad1] = a
        intensity[ad0, ad1, :] += 1
    print("    end loop bitmap and intensity")
    return bitmap, intensity
