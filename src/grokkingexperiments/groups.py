import numpy as np 

def operation(n, op, i, j):
    if op == "mod_n":
        return (i + j) % n 
    elif op == "bitwise_and":
        return (i & j)


def generate_data(n, op):
    data = np.zeros(shape=(n**2, 3))

    for i in range(n):
        for j in range(n):
            data[(i *n) + j] = [i, j, operation(n, op, i,j)]
    
    return np.long(data)