import numpy as np


def linear(x, y):
    return(np.column_stack((x, y)))


def swirl(x, y):
    r = np.sqrt(x * x + y * y)
    return(np.column_stack((x * np.sin(r * r) - y * np.cos(r * r),
                            x * np.cos(r * r) + y * np.sin(r * r))))


def Rswirl(x, y):
    r = np.random.rand() + np.sqrt(x * x + y * y)
    return(np.column_stack((x * np.sin(r * r) - y * np.cos(r * r),
                            x * np.cos(r * r) + y * np.sin(r * r))))


def sinmoche(x, y):
    return (np.column_stack((np.sin(y - x), np.sin(x + y))))


def sinmoche2(x, y):
    return (np.column_stack((np.sin(x * np.pi),
                             np.sin(y * np.pi))))


def Rsinmoche(x, y):
    return (np.column_stack((np.sin(y - x + np.random.rand()),
                             np.sin(x + y + np.random.rand()))))


def spherical(x, y):
    r = 0.001
    omega = np.arctan((x + r) / (y + r))
    r = np.sqrt(x * x + y * y)
    return(np.column_stack((np.sin(np.pi * r) * omega / np.pi,
                            np.cos(np.pi * r) * omega / np.pi)))


def Rspherical(x, y):
    r = np.random.rand()
    omega = np.arctan((x + r) / (y + r))
    r = np.sqrt(x * x + y * y)
    return(np.column_stack((np.sin(np.pi * r) * omega / np.pi,
                            np.cos(np.pi * r) * omega / np.pi)))


def expinj(x, y):
    xx = np.exp(-x * x)
    yy = np.exp(-y * y)
    return(np.column_stack((xx, yy)))


def Rexpinj(x, y):
    xx = np.exp(-x * x - np.random.rand())
    yy = np.exp(-y * y - np.random.rand())
    return(np.column_stack((xx, yy)))


def pdj(x, y, p1=.7, p2=0.9, p3=.7, p4=.2):
    xx = np.sin(p1 * y) * np.cos(p2 * x)
    yy = np.sin(p3 * x) * np.cos(p4 * y)
    return(np.column_stack((xx, yy)))


def bubble(x, y):
    r = np.sqrt(x * x + y * y)
    coef = 1 / (r * r + 1e-8)
    return(np.column_stack((coef * x, coef * y)))


def Rbubble(x, y):
    r = np.sqrt(x * x + y * y)
    coef = 1 / (r * r + np.random.rand())
    return(np.column_stack((coef * x, coef * y)))
