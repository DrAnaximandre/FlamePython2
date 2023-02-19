import numpy as np
from numba import njit
from tqdm import tqdm

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


@njit(fastmath=True)
def renderImage(F_loc, C, bitmap, intensity, goods, coef_forget):
    ''' this renders the image
        '''
    cf1 = coef_forget + 1
    C = C * coef_forget / cf1
    for i in goods:
        ad0 = F_loc[i, 0]
        ad1 = F_loc[i, 1]
        bitmap[ad0, ad1] /= cf1
        bitmap[ad0, ad1] += C[i, :]
        # a = (C[i, :] * coef_forget + sto) / (coef_forget + 1)
        intensity[ad0, ad1] += 1
    #print("    end loop bitmap and intensity")
    return bitmap, intensity
