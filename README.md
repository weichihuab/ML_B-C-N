#Description

This respository is the machine learning implementation to predict mechanical properties of a given compound.
Find more information in the preprint: https://arxiv.org/abs/2011.02038

#Data

The data used in our paper is from Materials Project DFT database.

The python code could be executed using python 3.7.

1. The folder "includes" has a module for generating 60 features used in this study. (The version of pymatgen used in features_gen.py is 2021.2.16)

2. The data.json file contains samples from Materials Project with their chemical formula and mechanical properties like bulk and shear modulus. 

3. rf_bulk_modulus.py and rf_shear_modulus.py, can generate the following files:


        formula_K.json               --  data of chemical formula and bulk modulus
        formula_G.json               --  data of chemical formual and shear modulus
        features.csv                 --  60 features of corresponding chemical formula in formula_K.json an formula_G.json
        rf_bulk_modulus.joblib       --  trained random forests model for bulk modulus
        rf_shear_modulus.joblib      --  trained random forests model for shear modulus

    NOTE: The trained random forests models would depend on the random seed, which can be set manually in rf_bulk_modulus.py and rf_shear_modulus.py using argument "random_state". 


4. The code rf_prediction.py loads rf_bulk_modulus.joblib and rf_shear_modulus.joblib to predict harness, bulk modulus, and shear modulus of given chemical compositions in "formula_list".
