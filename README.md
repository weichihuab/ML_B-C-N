# Description

This respository is the machine learning implementation to predict mechanical properties (bulk modulus, shear modulus, and Vickers hardness) for a given compound.
Find more information in our paper "Machine learning and evolutionary prediction of superhard B-C-N compounds": https://www.nature.com/articles/s41524-021-00585-7

The folder "includes" has a module for generating 60 features used in our study.

# Data

The data used in our paper is from Materials Project DFT database (https://materialsproject.org/). The data.json can be found in https://github.com/weichihuab/ML_B-C-N_data.

# Codes

### Building ML model

  - python rf_bulk_modulus.py
  - python rf_shear_modulus.py

rf_bulk_modulus.py and rf_shear_modulus.py, can generate the following files:

        formula_K.json               --  data of chemical formula and bulk modulus
        formula_G.json               --  data of chemical formual and shear modulus
        features.csv                 --  60 features of corresponding chemical formula in formula_K.json an formula_G.json
        rf_bulk_modulus.joblib       --  trained random forests model for bulk modulus
        rf_shear_modulus.joblib      --  trained random forests model for shear modulus

    NOTE: The trained random forests models would depend on the random seed, which can be set manually in rf_bulk_modulus.py and rf_shear_modulus.py using argument "random_state". 

### ML model prediction

  - python rf_prediction.py

The code rf_prediction.py loads rf_bulk_modulus.joblib and rf_shear_modulus.joblib to predict hardness, bulk modulus, and shear modulus of given chemical compositions in "formula_list".

# Citation

Wei-Chih Chen, Joanna N. Schmidt, Da Yan, Yogesh K. Vohra, and Cheng-Chien Chen, "Machine Learning and Evolutionary Prediction of Superhard B-C-N Compounds", npj Comput Mater 7, 114 (2021). https://doi.org/10.1038/s41524-021-00585-7

# Environment
  - python 3.7
  - pymatgen 2021.2.16
  - sklearn 0.23.2
