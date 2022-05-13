# FlamePython2

<p align="center"> <img src="images/mess.png"></p>

### Dependencies

Requirements:
```
numpy
pillow
matplotlib
tqdm
```

### Generate your first images

Run the following command in a terminal:
```
python src/helpers.py 
```

An image will be saved locally.
Your action on the parameters that were used to generate this image is required in the terminal.   

One first suggestion is to increase N from the default to 1000000.
The time needed to render an image will increase.


### UserGuide Example

```
import class_fractale.py
burn = 20
niter = 50
zoom = 1
N = 10000

F1 = Fractale(burn, niter, zoom)

```
Declaration of a fractale object. It will run "burn" times without saving the points, then run "niter" times saving the points. 
The zoom parameter is just for convenience. These are not parameters you should play with as a start.

```
a1 = np.array([0, 1, 0, 0, 0, 1])
a2 = np.array([1, 1, 0, 0, 0, 1])
a3 = np.array([0, 1, 0, 1, 0, 1])
```
These vectors will be used further in the Functions. You can modify these values but as a rule of thumb you should keep the values between 1.2 and -1.2.


```
v1 = Variation()
v1.addFunction([.5], a1, [linear], .2, [255, 0, 0])
v1.addFunction([.5], a2, [linear], .2, [0, 255, 0])
v1.addFunction([.5], a3, [linear], .2, [0, 0, 255])

```

Here we declare a Variation with 3 Functions. For instance, with the values of `a`s this will give a Serpinski's triangle.

Each Function has a number of scales (here the first parameter that still needs to be in a list). It also has a vector (one of the `a`s).
Then it has some additives (here it's `linear`). You can add additives (see `utils.py` for the ones that are implemented) but more than 3 additives per Function gives messy images. 
Then one can see that the Functions also have the same probability to appear (0.2, but the probs are normalised to sum to one).
Finally, each Function has a different color (the last list of 3 values). 


We then add the variation to the Fractale object, with N points.
```
F1.addVariation(v1, N)

```

We can then build the fractale and run it!
```
F1.build()
F1.runAll()
   
```
This is pretty fast. The coslty part is to go from the coordinate space to the image space:
```
print("Generating the image")
out = F1.toImage(600, optional_kernel_filtering=False)
   
```

Saving is pretty fast:
```
out.save("serp.png")
```

Taddaaa!


### History of the repo

Second version of the Flame Fractals in Python - inspired by Draves and Reckase (http://flam3.com/flame_draves.pdf)

I tried to make it easy to add Functions and Variations to the Fractals. 

<p align="center"> <img src="images/Serp.png"></p>

