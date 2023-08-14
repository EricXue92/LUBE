import math 
import numpy as np 
import pandas as pd

# nonconstant noise 
def fn(x1, x2, x3, x4, x5): 
    y = 10*(np.sin(np.pi*x1*x2)) + 20 * np.power((x3-0.5),2) + 10 * x4 + 5 * x5 
    return y

def generate_data2():
    x1 = np.array( [np.random.uniform() for i in range(500)] )
    x2 = np.array( [np.random.uniform() for i in range(500)] )
    x3 = np.array( [np.random.uniform() for i in range(500)] )
    x4 = np.array( [np.random.uniform() for i in range(500)] )
    x5 = np.array( [np.random.uniform() for i in range(500)] )
    
    y = fn(x1, x2, x3, x4, x5)
    constant_noise = np.array( [x1, x2, x3, x4, x5, y]).T
    constant_noise = pd.DataFrame(constant_noise, columns = ['x1','x2','x3','x4','x5','y'])
    constant_noise.to_csv('dataset/2_nonconstant_noise.csv', index = None)
    




