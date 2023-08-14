import math 
import numpy as np 
import pandas as pd 

# constant noise 
def fn(x1, x2, x3, x4, x5, noise1, noise2): 
    y = 0.0647*(12+3*x1-3.5*np.power(x2, 2) + 7.2*np.power(x3,3)) * (1+np.cos(4*(np.pi)*x4))*(1+0.8 * np.sin(3*np.pi*x5)) + noise1 + noise2
    return y

def generate_data1():
	noise1 = np.random.normal(0, 0.25, 300)
	noise2 = np.random.normal(0, 0.5, 300)
	x1 = np.array( [np.random.uniform() for i in range(300) ] )
	x2 = np.array( [np.random.uniform() for i in range(300) ] )
	x3 = np.array( [np.random.uniform() for i in range(300) ] )
	x4 = np.array( [np.random.uniform() for i in range(300) ] )
	x5 = np.array( [np.random.uniform() for i in range(300) ] )
    
	y = fn(x1, x2, x3, x4, x5, noise1, noise2)
	constant_noise = np.array( [x1, x2, x3, x4, x5, y]).T
	constant_noise = pd.DataFrame(constant_noise, columns = ['x1','x2','x3','x4','x5','y'])
	constant_noise.to_csv('dataset/1_constant_noise.csv', index = None)



