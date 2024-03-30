import numpy as np
import cv2

rac = cv2.imread("spsmall.png")
size = 1440
rac = cv2.resize(rac, (size,size))

# invert rac
rac = 255 - rac
imgray = cv2.cvtColor(rac, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, 0)

# get the contours  
contour,_ = cv2.findContours(thresh, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE)

h = list(range(len(contour)))
#np.random.shuffle(h)
contour = [contour[i] for i in h]
contour = np.concatenate(contour)
cont = contour
cont = np.concatenate((cont[:,:,1],cont[:,:,0]),1)

subset = 200 # adapt to your image
smaller_contour = cont[::subset]
smaller_contour = smaller_contour
print(smaller_contour.shape)

def raton(x,y):
    
    # x, y are between -1 and 1
    x2 = (size*(x + 1) / 2).astype(np.int32)
    y2 = (size* (y + 1) / 2).astype(np.int32)

    #clip between 0 and size
    x2 = np.clip(x2, 0, size-1)
    y2 = np.clip(y2, 0, size-1)


    mask = rac[x2,y2][:,0:2] > 125

    # where mask, return xy, else return random that lives into the mask
    # sample an index from the True values of mask

    # map it to the closest contour point
    dismat = np.zeros((x2.shape[0], smaller_contour.shape[0]))
   
    for i in range(dismat.shape[0]):
        if not mask[i,0]:
            dismat[i] = np.sqrt((x2[i] - smaller_contour[:,0])**2 + (y2[i] - smaller_contour[:,1])**2)
    
  
    idx = (np.argmin(dismat, axis = 1)*subset+ np.random.randint(-subset*2, subset*2, size=x2.shape[0])) % cont.shape[0] 
    
    #noise = cont[idx] / 666 * 2 - 1 + np.random.uniform(-0.01,0.01,size=cont[idx].shape)
    #idx = np.random.choice(np.arange(cont.shape[0]), size = mask.shape[0], replace = True)

    return np.where(mask, np.column_stack((x,y)), cont[idx] / size * 2 - 1)


    
 #    return np.where(mask, np.column_stack((x,y)), np.random.rand(mask.shape[0],2) * 2 - 1)


def linear(x, y):
    return(np.column_stack((x,y)))


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
    r = 0.1
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


def Wexpinj(x, y):
    xx = np.maximum(0.99, np.exp(y - x*x))
    yy = np.maximum(0.7, np.exp(x- y*y))
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
