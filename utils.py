import numpy as np

def rescale255(tuple_bm3,rha):
    ## Ugly. will fix that later.
    a=int(tuple_bm3[0]*rha)
    b=int(tuple_bm3[1]*rha)
    c=int(tuple_bm3[2]*rha)
    return (a,b,c)
  

def linear(x,y):
    a=np.column_stack((x,y))
    return(np.column_stack((x,y)))

def swirl(x,y):
    r=1+np.sqrt(x*x+y*y)
    return(np.column_stack((x*np.sin(r*r)-y*np.cos(r*r),x*np.cos(r*r)+y*np.sin(r*r))))

def spherical(x,y):
    omega=np.arctan((x+10)/(y+10))
    r=np.sqrt(x*x+y*y)
    return(np.column_stack((np.sin(np.pi*r)*omega/np.pi,np.cos(np.pi*r)*omega/np.pi)))

def expinj(x,y):
    xx=np.exp(-x*x)
    yy=np.exp(-y*y)
    return(np.column_stack((xx,yy)))

def pdj(x,y,p1=.7,p2=3.14,p3=.7,p4=.2):
    xx=np.sin(p1*y)-np.cos(p2*x)
    yy=np.sin(p3*x)-np.cos(p4*y)
    return(np.column_stack((xx,yy)))

def bubble(x,y):
    r=np.sqrt(x*x+y*y)
    coef=4/(r*r+4)
    return(np.column_stack((coef*x,coef*y)))
