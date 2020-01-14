import numpy as np

def quizz(param, iteration, out):


    print("--- Actions ---")
    print("Change intensity coefficient? (c)")
    print("Change forget coefficient? (f)")
    print("Change Clip? (C)")
    print("Change A? (A)")
    print("add small noise to A? (a)")
    print("change N ? (N)")
    print("change niter(n)")
    out.save(f"{param.name}-{iteration}.png")
    action = input("Your action ?")
    
    end = False
    if action == "c":
        param.ci = float(input("Coef intensity ? "))
    elif action == "f":
        param.fi = float(input("Coef forget ? "))
    elif action == "C":
        param.clip = float(input(f"Curent clip : {param.clip}, new clip? "))
    elif action == "A":
        print(param.A)
        coord = int(input("row")), int(input("col"))
        current_value = param.A[coord[0], coord[1]]
        new_value =  float(input(f"Current value {current_value} new value ?"))
        param.A[coord[0], coord[1]] = new_value
        print(param.A)
    elif action == "a":
        noise = np.random.multivariate_normal(np.zeros(param.W),1/6*np.random.uniform(size = (param.W,param.W)),(6)).T
        param.A = noise
    elif action == "N":
        param.N = int(input("N? "))
    elif action == "n":
        param.niter = int(input("niter ? "))
    elif action == "exit":
        end=True
    else:
        print("please choose again")
        action = input("Your action ?")



    return(param, end)