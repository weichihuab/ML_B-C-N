### This code first loads data.json and uses bulk modulus (K_VRH) as the target. After data selection, the features are generated. 
### ML data including the target and features are then saved in formula_K.json and features.csv, respectively.
### After training, the ML model for bulk modulus is saved as rf_bulk_modulus.joblib.
### NOTE: The random seed can be randomly chosen or be specified explicitly by "random_state".

import numpy as np
import pymatgen as mg
from pymatgen import MPRester
from pymatgen.analysis.phase_diagram import PhaseDiagram, PDPlotter
from sklearn.model_selection import cross_val_score
import json
from includes import features_gen as fg


# uncomment the following block for downloading data:

'''
# block start
# One should first login to the materials project website and get an API key:
# https://materialsproject.org/dashboard

PMG_MAPI_KEY = 'ETBG2F3ESap2hvgo7wZN'
m=MPRester(PMG_MAPI_KEY)
data=m.query(criteria={'elements':{"$nin": ["He","Ne","Ar","Kr","Xe","Rn"]}, "elasticity": {"$exists": True}},
             properties=["material_id","full_formula","elasticity","formation_energy_per_atom"])

with open('data.json', 'w') as outfile:
    json.dump(data, outfile)

# block end
'''

with open('data.json') as json_file:
    data = json.load(json_file)


max_diff_K_Reuss_K_Voigt = 50
max_K_VRH = 550
min_K_VRH = 0.1
max_K_over_G = 4
max_G_over_K = 4
max_diff_G_Reuss_G_Voigt = max_diff_K_Reuss_K_Voigt
max_G_VRH = max_K_VRH
min_G_VRH = min_K_VRH
min_H = 0


feature_list=[]
label_list=[]
ML_data_list=[]
num_data=len(data)
for i in range(num_data):
    if(data[i]['elasticity']['K_VRH']==None or data[i]['elasticity']['G_VRH']==None):
        continue
    elif(data[i]['formation_energy_per_atom'] > 0.2 ):
        continue
    elif(data[i]['elasticity']['K_Reuss'] < 0 or data[i]['elasticity']['K_Voigt'] < 0 or data[i]['elasticity']['G_Reuss'] < 0 or data[i]['elasticity']['G_Voigt'] < 0):
        continue
    elif(abs(data[i]['elasticity']['K_Reuss'] - data[i]['elasticity']['K_Voigt']) > max_diff_K_Reuss_K_Voigt ):
        continue
    elif(abs(data[i]['elasticity']['G_Reuss'] - data[i]['elasticity']['G_Voigt']) > max_diff_G_Reuss_G_Voigt ):
        continue
    elif(data[i]['elasticity']['K_VRH'] > max_K_VRH):
        continue
    elif(data[i]['elasticity']['K_VRH'] < min_K_VRH):
        continue
    elif(data[i]['elasticity']['K_VRH'] / (data[i]['elasticity']['G_VRH']+0.001) >  max_K_over_G):
        continue
    elif(data[i]['elasticity']['G_VRH'] / (data[i]['elasticity']['K_VRH']+0.001) >  max_G_over_K):
        continue
    else:
        K = data[i]['elasticity']["K_VRH"]
        G = data[i]['elasticity']["G_VRH"]
        #H = 2*(G**3/K**2)**0.585-3    # Chen's model
        H = 0.92*(G/K)**1.137*G**0.708 # Tian's model
        feature_list.append(fg.features_generator(data[i]["full_formula"]))
        label_list.append(data[i]['elasticity']["K_VRH"])
        ML_data_list.append(data[i]["full_formula"])
		
# output features into a csv file
from sklearn import datasets
import pandas as pd 

feature_names=['L0','L2','L3',
             'atomic_number_min','atomic_number_max','atomic_number_range',
             'atomic_number_fwm','atomic_number_ad',
             'atomic_mass_min','atomic_mass_max','atomic_mass_range',
             'atomic_mass_fwm','atomic_mass_ad',
             'column_min','column_max','column_range',
             'column_fwm','column_ad',
             'row_min','row_max','row_range',
             'row_fwm','row_ad',
             'atomic_radius_min','atomic_radius_max','atomic_radius_range',
             'atomic_radius_fwm','atomic_radius_ad',
             'X_min','X_max','X_range',
             'X_fwm','X_ad',
             's_val_min','s_val_max','s_val_range',
             's_val_fwm','s_val_ad',
             'p_val_min','p_val_max','p_val_range',
             'p_val_fwm','p_val_ad',
             'd_val_min','d_val_max','d_val_range',
             'd_val_fwm','d_val_ad',
             'f_val_min','f_val_max','f_val_range',
             'f_val_fwm','f_val_ad',
             's_val_occ','p_val_occ','d_val_occ','f_val_occ',
             'isionic','IC_max','IC_mean'
             ]

df = pd.DataFrame(data=feature_list, columns = feature_names)
df.to_csv('features.csv', sep = ',', index = False)

# output chemical formulae and Ks to a json file

result = []

for i in range(len(ML_data_list)):
    result.append({ML_data_list[i]: label_list[i]})

import json
with open('formula_K.json', 'w') as outfile:
    json.dump(result, outfile, indent=4)


#### start ML #####

print("num_of_samples : ", len(feature_list))
X=np.array(feature_list)
y=np.array(label_list)

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt
import matplotlib as mpl
mpl.use('Agg')

indices = np.arange(len(feature_list))

# split data into train+validation set and test set 
X_trainval, X_test, y_trainval, y_test, I_X_trainval, I_X_test = train_test_split(
    X, y, indices, random_state=0, test_size=.1)


# split train+validation set into training and validation sets 
X_train, X_valid, y_train, y_valid = train_test_split(
    X_trainval, y_trainval, random_state=1)

# for data distribution
#plt.plot(y_trainval,color='r')
#plt.savefig('y_train.png',dpi=300,bbox_inches='tight')
#plt.clf()


# sklearn GridSearchCV:

#param_grid = {'max_depth': [20,30,40], 'n_estimators':[20,40,80,120,160], 'n_jobs':[-1]}
param_grid = {'max_depth': [12], 'n_estimators':[100], 'n_jobs':[-1]}
print("Parameter grid:\n{}".format(param_grid))


#grid_search = GridSearchCV(RandomForestRegressor(random_state=1111), param_grid, cv=10)
grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=10)
tree = grid_search.fit(X_trainval, y_trainval)
# save model using joblib
import joblib
joblib.dump(tree, 'rf_bulk_modulus.joblib')

pred_tree = tree.predict(X_test)

print('')
print("Best parameters: {}".format(grid_search.best_params_))
print("Best cross-validation score: {:.3f}".format(grid_search.best_score_))
#print("cross-validation ", grid_search.cv_results_)
print("Test set score: {:.3f}".format(grid_search.score(X_test, y_test)))
print("Test set Pearson score: {:.3f}".format(np.corrcoef(pred_tree, y_test)[0, 1]))

