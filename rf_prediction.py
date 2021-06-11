
#import numpy as np
#import pymatgen as mg
#from pymatgen import MPRester
#from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from sklearn.model_selection import cross_val_score
from includes import features_gen as fg
import joblib


tree_K = joblib.load("rf_bulk_modulus.joblib")
tree_G = joblib.load("rf_shear_modulus.joblib")

pred_BCN=[]
formula_list=["BC10N","B4C5N3","B2C3N1","B6N4O3","Fe2O3","LaBi"]
for i in formula_list:
    print(fg.features_generator(i))
    pred_BCN.append(fg.features_generator(i))
pred_tree_K = tree_K.predict(pred_BCN)
pred_tree_G = tree_G.predict(pred_BCN)
H_from_K_G = 0.92*(pred_tree_G/pred_tree_K)**1.137*pred_tree_G**0.708 
print("formula list", formula_list)
print("predicted hardness:", H_from_K_G)
print("predicted shear modulus:", pred_tree_G)
print("predicted bulk modulus:", pred_tree_K)
