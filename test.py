import numpy as np
from sklearn import datasets

data = datasets.load_iris()
print(data.target)
arik = np.full(10,"arik")
idrisy = np.full(10,"idrisy")

print(np.append(
    np.append(arik,idrisy,),
    arik))