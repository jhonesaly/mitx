import numpy as np

def randomization(n):
    """
    Arg:
      n - an integer
    Returns:
      A - a randomly-generated nx1 Numpy array.
    """
    A = np.random.rand(n, 1)
    return A

def operations(h, w):
    """
    Takes two inputs, h and w, and makes two Numpy arrays A and B of size
    h x w, and returns A, B, and s, the sum of A and B.

    Arg:
      h - an integer describing the height of A and B
      w - an integer describing the width of A and B
    Returns (in this order):
      A - a randomly-generated h x w Numpy array.
      B - a randomly-generated h x w Numpy array.
      s - the sum of A and B.
    """
    A = np.random.rand(h, w)
    B = np.random.rand(h, w)
    s = A + B
    return A, B, s


def norm(A, B):
    """
    Takes two Numpy column arrays, A and B, and returns the L2 norm of their
    sum.

    Arg:
      A - a Numpy array
      B - a Numpy array
    Returns:
      s - the L2 norm of A+B.
    """
    return np.linalg.norm(A + B)


def neural_network(inputs, weights):
    """
     Takes an input vector and runs it through a 1-layer neural network
     with a given weight matrix and returns the output.

     Arg:
       inputs - 2 x 1 NumPy array
       weights - 2 x 1 NumPy array
     Returns (in this order):
       out - a 1 x 1 NumPy array, representing the output of the neural network
    """
    return np.tanh(weights.T @ inputs)

def scalar_function(x, y):
    """
    Returns the f(x,y) defined in the problem statement.
    """
    if x <= y:
        return x * y
    else:
        return x / y

def vector_function(x, y):
    """
    Make sure vector_function can deal with vector input x,y 
    """
    #Your code here
    raise NotImplementedError

if __name__ == "__main__":
    print("--- Testando a função scalar_function(x, y) ---")
    # Caso 1: x <= y (3 <= 5 -> deve retornar 15)
    r1 = scalar_function(3, 5)
    print(f"scalar_function(3, 5) -> Resultado: {r1} (Esperado: 15)")

    # Caso 2: x > y (6 > 2 -> deve retornar 3.0)
    r2 = scalar_function(6, 2)
    print(f"scalar_function(6, 2) -> Resultado: {r2} (Esperado: 3.0)")


