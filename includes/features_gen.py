# Please find detailed information for features in the preprint: https://arxiv.org/abs/2011.02038


import numpy as np
import pymatgen as mg
from pymatgen import MPRester


def features_generator(string):
    #elements, ratio = preprocessing(chemical)
    comp = mg.Composition(string)
    num_elements=len(comp.element_composition)
    elements=[]   # elements
    ratio=[]      # fraction
    oxydation=[]  # oxidation states
    tempcount=1

    for ii in range(num_elements):
        etemp=str(comp.elements[ii])
        elements.append(etemp)
        etemp=mg.Element(etemp)
        
        rtemp=comp.get_atomic_fraction(etemp)
        ratio.append(rtemp)
        
        oxydation.append(rtemp*np.array(etemp.common_oxidation_states))
        tempcount=tempcount*len(oxydation[ii])
    
    composition = np.array(ratio)/sum(ratio)
    # for (a)
    L0 = len(elements)
    L2 = np.linalg.norm(composition,ord=2)
    L3 = np.linalg.norm(composition,ord=3)
    
    # for (b.1) atomic number
    atomic_number= []
    for atom in elements:
        atomic_number.append(mg.Element(atom).number)
        
    atomic_number_min = float(min(atomic_number))
    atomic_number_max = float(max(atomic_number))
    atomic_number_range = atomic_number_max-atomic_number_min
    temp = np.array(atomic_number)
    atomic_number_fwm = np.dot(composition,temp)
    temp = abs(temp-atomic_number_fwm) 
    atomic_number_ad = np.dot(composition,temp)

    # for (b.2) atomic mass
    atomic_mass= []
    for atom in elements:
        atomic_mass.append(mg.Element(atom).atomic_mass)
        
    atomic_mass_min = float(min(atomic_mass))
    atomic_mass_max = float(max(atomic_mass))
    atomic_mass_range = atomic_mass_max-atomic_mass_min
    temp = np.array(atomic_mass)
    atomic_mass_fwm = np.dot(composition,temp)
    temp = abs(temp-atomic_mass_fwm) 
    atomic_mass_ad = np.dot(composition,temp)

    # for (b.3) column
    column= []
    for atom in elements:
        column.append(mg.Element(atom).group)
        
    column_min = float(min(column))
    column_max = float(max(column))
    column_range = column_max-column_min
    temp = np.array(column)
    column_fwm = np.dot(composition,temp)
    temp = abs(temp-column_fwm) 
    column_ad = np.dot(composition,temp)

    # for (b.4) row
    row= []
    for atom in elements:
        row.append(mg.Element(atom).row)
        
    row_min = float(min(row))
    row_max = float(max(row))
    row_range = row_max-row_min
    temp = np.array(row)
    row_fwm = np.dot(composition,temp)
    temp = abs(temp-row_fwm) 
    row_ad = np.dot(composition,temp)
    
    # for (b.5) atomic radius
    atomic_radius= []
    for atom in elements:
        atomic_radius.append(mg.Element(atom).atomic_radius)
        
    atomic_radius_min = float(min(atomic_radius))
    atomic_radius_max = float(max(atomic_radius))
    atomic_radius_range = atomic_radius_max-atomic_radius_min
    temp = np.array(atomic_radius)
    atomic_radius_fwm = np.dot(composition,temp)
    temp = abs(temp-atomic_radius_fwm) 
    atomic_radius_ad = np.dot(composition,temp)
    
    # for (b.6) electronegativity(X)
    X= []
    for atom in elements:
        X.append(mg.Element(atom).X)
        
    X_min = float(min(X))
    X_max = float(max(X))
    X_range = X_max-X_min
    temp = np.array(X)
    X_fwm = np.dot(composition,temp)
    temp = abs(temp-X_fwm) 
    X_ad = np.dot(composition,temp)    
    
    # for (b.7) valence s electrons
    s_val = []
    for atom in elements:
        str1=mg.Element(atom).electronic_structure
        pos1 = str1.find('s')
        str1 = str1[pos1:]
        pos2 = str1.find('.')
        if (pos1 == -1):
            s_val.append(0)
        else:
            if (pos2 != -1):
                s_val.append(int(str1[1:pos2]))
            else:
                s_val.append(int(str1[1:]))
    s_val_min = float(min(s_val))
    s_val_max = float(max(s_val))
    s_val_range = s_val_max-s_val_min
    temp = np.array(s_val)
    s_val_fwm = np.dot(composition,temp)
    temp = abs(temp-s_val_fwm) 
    s_val_ad = np.dot(composition,temp)    

    # for (b.8) valence p electrons
    p_val = []
    for atom in elements:
        str1=mg.Element(atom).electronic_structure
        pos1 = str1.find('p')
        str1 = str1[pos1:]
        pos2 = str1.find('.')
        if (pos1 == -1):
            p_val.append(0)
        else:
            if (pos2 != -1):
                p_val.append(int(str1[1:pos2]))
            else:
                p_val.append(int(str1[1:]))
    p_val_min = float(min(p_val))
    p_val_max = float(max(p_val))
    p_val_range = p_val_max-p_val_min
    temp = np.array(p_val)
    p_val_fwm = np.dot(composition,temp)
    temp = abs(temp-p_val_fwm) 
    p_val_ad = np.dot(composition,temp)   
    
    # for (b.9) valence p electrons
    d_val = []
    for atom in elements:
        str1=mg.Element(atom).electronic_structure
        pos1 = str1.find('d')
        str1 = str1[pos1:]
        pos2 = str1.find('.')
        if (pos1 == -1):
            d_val.append(0)
        else:
            if (pos2 != -1):
                d_val.append(int(str1[1:pos2]))
            else:
                d_val.append(int(str1[1:]))
    d_val_min = float(min(d_val))
    d_val_max = float(max(d_val))
    d_val_range = d_val_max-d_val_min
    temp = np.array(d_val)
    d_val_fwm = np.dot(composition,temp)
    temp = abs(temp-d_val_fwm) 
    d_val_ad = np.dot(composition,temp)
    
    # for (b.10) valence f electrons
    f_val = []
    for atom in elements:
        str1=mg.Element(atom).electronic_structure
        pos1 = str1.find('f')
        str1 = str1[pos1:]
        pos2 = str1.find('.')
        if (pos1 == -1):
            f_val.append(0)
        else:
            if (pos2 != -1):
                f_val.append(int(str1[1:pos2]))
            else:
                f_val.append(int(str1[1:]))
    f_val_min = float(min(f_val))
    f_val_max = float(max(f_val))
    f_val_range = f_val_max-f_val_min
    temp = np.array(f_val)
    f_val_fwm = np.dot(composition,temp)
    temp = abs(temp-f_val_fwm) 
    f_val_ad = np.dot(composition,temp)
    
    # for (c)
    val = np.array(s_val) + np.array(p_val) + np.array(d_val) + np.array(f_val)
    denominator=np.dot(val,composition)
    temp = np.array(s_val)
    s_val_occ = np.dot(temp,composition)/denominator
    temp = np.array(p_val)
    p_val_occ = np.dot(temp,composition)/denominator
    temp = np.array(d_val)
    d_val_occ = np.dot(temp,composition)/denominator
    temp = np.array(f_val)
    f_val_occ = np.dot(temp,composition)/denominator
    
    # for (d)
    if(num_elements == 1):
        isionic = 0
    else:
        for ii in range(num_elements):
            factor = int(tempcount/len(oxydation[ii]))
            oxydation[ii]=np.array(list(oxydation[ii])*factor)

        for ii in range(1,num_elements):
            oxydation[0] = np.vstack((oxydation[0], oxydation[ii]))
        
#        print(np.sum(oxydation[0],axis=0)) # axis=0 meand sum vertically
#        print(min(abs(np.sum(oxydation[0],axis=0))))

        isionic = 0
        if(min(abs(np.sum(oxydation[0],axis=0))) < 10**-8): isionic = 1
#    print('')
#    print('Is_ionic:', isionic)

    
    # max ionic character:
    IC_list = []
    if(num_elements == 1):
        IC_max = 0
        IC_mean = 0
    else:
        for ii in range(0,num_elements):
            for jj in range(ii+1,num_elements):
            
                e1=str(comp.elements[ii])
                e2=str(comp.elements[jj])
        
                e1=mg.Element(e1)
                e2=mg.Element(e2)

                f1=comp.get_atomic_fraction(e1)
                f2=comp.get_atomic_fraction(e2)
               
                #get the electronegativity attribute:
                i1=e1.X
                i2=e2.X
        
                IC_list.append(1 - np.exp(-(i1-i2)**2/4))
        
        #    print(IC_list)
        IC_max = max(IC_list)
        
        # mean ionic character:
        IC_mean = comp.average_electroneg
    
    return_list=[L0,L2,L3,
                 atomic_number_min,atomic_number_max,atomic_number_range,
                 atomic_number_fwm,atomic_number_ad,
                 atomic_mass_min,atomic_mass_max,atomic_mass_range,
                 atomic_mass_fwm,atomic_mass_ad,
                 column_min,column_max,column_range,
                 column_fwm,column_ad,
                 row_min,row_max,row_range,
                 row_fwm,row_ad,
                 atomic_radius_min,atomic_radius_max,atomic_radius_range,
                 atomic_radius_fwm,atomic_radius_ad,
                 X_min,X_max,X_range,
                 X_fwm,X_ad,
                 s_val_min,s_val_max,s_val_range,
                 s_val_fwm,s_val_ad,
                 p_val_min,p_val_max,p_val_range,
                 p_val_fwm,p_val_ad,
                 d_val_min,d_val_max,d_val_range,
                 d_val_fwm,d_val_ad,
                 f_val_min,f_val_max,f_val_range,
                 f_val_fwm,f_val_ad,
                 s_val_occ,p_val_occ,d_val_occ,f_val_occ,
                 isionic,IC_max,IC_mean
                 ]
    
    return return_list


