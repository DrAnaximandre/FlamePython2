
from examples.Demo import DemoImageHolder
from examples.Sierpinski import Sierpinski
from examples.Sierpinskies import Sierpinskies

if __name__ == "__main__":
    
    serp = Sierpinskies(5,100, name="serp", size=1024)
    serp.run()