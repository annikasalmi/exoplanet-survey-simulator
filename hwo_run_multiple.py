import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pandas as pd

from lifesim.core.hwo_data import HWOData
from ppop_generator import PPop
from tools import PPOP_DIR

# generate files
PPopObj = PPop()

# reduce nuniverses, nstars for testing
## how to initialize a blank dict with pregiven memory size??

for i in range(3):
    t=time.time()
    filename = 'test_runs_' + str(i) # str
    data_path = os.path.join(PPOP_DIR, 'data', filename)
    PPopObj.run_ppop(data_path, ntest=100, nuniverses=1)
    PPopObj.catalog_from_ppop(data_path)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.)  # remove all A stars
    PPopObj.catalog_remove_distance(stype='M', mode='larger', dist=10.)  # remove M stars > 10pc to

    hwo_data = HWOData(PPopObj.catalog)
    hwo_data.determine_detectable()

    df_hab, df_false = hwo_data.organize_data()

    hab_dict_temp = dict(np.asarray(df_hab))
    unhab_dict_temp = dict(np.asarray(df_false))
    print(hab_dict_temp)
    print(unhab_dict_temp)
    if type(hab_dict_temp['F']) != int:
        a=1

    if i == 0:
        total_hab_dict = hab_dict_temp
        total_unhab_dict = unhab_dict_temp
    else:
        for key in hab_dict_temp:
            if type(total_hab_dict[key]) == int:
                list_hab_dict = list([total_hab_dict[key]])
                list_unhab_dict = list([total_unhab_dict[key]])
            else:
                pass
            list_hab_dict.append(hab_dict_temp[key])
            list_unhab_dict.append(unhab_dict_temp[key])
            total_hab_dict[key] = list_hab_dict
            total_unhab_dict[key] = list_unhab_dict

    print(f'total time is {time.time()-t} seconds')


x = np.arange(len(df_size.stype.unique())) 
width=0.4
plt.bar(x-0.2, df_hab.count_overall, width)#, color='cyan') 
plt.bar(x+0.2, df_false.count_overall, width)#, color='orange') 
plt.xticks(x, df_size.stype.unique()) 
plt.xlabel("Stellar Type") 
plt.ylabel("Count") 
plt.title('HWO Detectability')
plt.legend(["Habitable", "Inhabitable"]) 
plt.show() 
a=1