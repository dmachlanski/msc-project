import numpy as np
from utils import extend_data

data = np.array([[0,1,1,0,1], [1,0,0,1,0]])

#extended = extend_data(data, 2)
extended = extend_data(data, 2, 3)

print("Before:")
print(data)
print()
print("After:")
print(extended)