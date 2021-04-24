import numpy as np


def quizz(param, iteration, out):

    print("--- Actions ---")
    print("Change intensity coefficient? (c)")
    print("Change forget coefficient? (f)")
    print("Change Clip? (C)")
    print("Change A? (A)")
    print("multiply A? (M)")
    print("add small noise to A? (a)")
    print("change N ? (N)")
    print("change niter(n)")
    print("Exit (exit)?")
    out.save(f"images/{param.name}-{iteration}.png")
    action = input("Your action ?")

    end = False
    if action == "c":
        param.ci = float(input("Coef intensity ? "))
    elif action == "f":
        param.fi = float(input("Coef forget ? "))
    elif action == "C":
        param.clip = float(input(f"Curent clip : {param.clip}, new clip? "))
    elif action == "M":
        coef = float(input("choose a value to multiply A"))
        param.A *= coef
    elif action == "A":
        print(param.A)
        coord = int(input("row")), int(input("col"))
        current_value = param.A[coord[0], coord[1]]
        new_value = float(input(f"Current value {current_value} new value ?"))
        param.A[coord[0], coord[1]] = new_value
        print(param.A)
    elif action == "a":
        noise = np.random.uniform(-1 / 6, 1 / 60, (6, param.W)).T
        param.A = param.A + noise
    elif action == "N":
        param.variation_parameters.N = int(input(f" current: {str(param.variation_parameters)}, new N? "))
    elif action == "n":
        param.niter = int(input("niter ? "))
    elif action == "exit":
        end = True
    else:
        print("please choose again")
        action = input("Your action ?")

    return(param, end)
