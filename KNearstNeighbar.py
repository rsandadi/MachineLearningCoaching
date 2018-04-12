from sklearn.datasets import *
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
import numpy as np
def displayImage(i):
 plt.imshow(digit['images'][i],cmap='Greys_r')
 plt.show()
digit = load_digits()
dig = pd.DataFrame(digit['data'][0:1700])
#print(digit['target'][:100])
train_x = digit['data'][:1700]
train_y = digit['target'][:1700]
KNN = KNeighborsClassifier(10)
KNN.fit(train_x,train_y)
test = np.array(digit['data'][1726])
test1 = test.reshape(1,-1)
displayImage(1726)

#print(KNN.predict(test1))
print(digit['target_names'])