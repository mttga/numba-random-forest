# numba-random-forest
Implementation of random forest (and decision forest) algorithm in python, optimized with numba

- cart_utils.py: contains the numba optimized functions
- deciion_forest.py: contains Cart and Decision Forest classes
- random_forest.py: contains RandomCart and Random Forest classes

To test it, add the three scripts in the same folder of your project. Then you can import and use RandomForest in the scikit-learn fashion:

```Python
from random_forest import RandomForest
# possible random methods:
# 'sigle' (one feature), 'tripe' (three features), 'log' (log of n of attribtues)
rf = RandomForest(n_trees=n, random_method='sqrt') 
rf.fit(x_train, y_train)
y_pred = rf.predict(x_test)
```
