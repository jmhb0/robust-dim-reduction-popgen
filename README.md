The Euro data file is pickled. To read:
```
import pickle
with open('./data/POPRES_non-reduced_phased_20.dat', 'rb') as pf:
        data = pickle.load(pf)
````


