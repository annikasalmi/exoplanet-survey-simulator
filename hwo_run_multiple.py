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
    df = PPopObj.run_ppop(data_path, ntest=100, nuniverses=1)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.)  # remove all A stars
    PPopObj.catalog_remove_distance(stype='M', mode='larger', dist=10.)  # remove M stars > 10pc to

    hwo_data = HWOData(PPopObj.catalog)

    hwo_data.determine_detectable()

    df_hab, df_false = hwo_data.organize_data()

    if i == 0:
        mapping_hab = zip(df_hab.T.columns, df_hab.stype)
        mapping_unhab = zip(df_false.T.columns, df_false.stype)
        df_hab_total = df_hab.T.rename(columns=dict(mapping_hab)).drop('stype').reset_index(drop=True)
        df_false_total = df_false.T.rename(columns=dict(mapping_unhab)).drop('stype').reset_index(drop=True)
    else:
        df_hab_total.loc[len(df_hab_total)] = df_hab.count_overall.values
        df_false_total.loc[len(df_false_total)] = df_false.count_overall.values

    print(f'total time is {time.time()-t} seconds')

df_results = pd.DataFrame(columns=['stypes', 'count_hab', 'error_hab', 'count_unhab', 'error_unhab'])
for i in df_hab_total.columns:
    count = np.mean(df_hab_total[i])
    err = np.std(df_hab_total[i])
    count_unhab = np.mean(df_false_total[i])
    err_unhab = np.std(df_false_total[i])
    df = pd.DataFrame(data={'stypes': [i], 'count_hab': [count], 'error_hab': [err], 
                                        'count_unhab': [count_unhab], 'error_unhab': [err_unhab]})
    df_results = pd.concat([df_results,df], ignore_index=True)
    
stypes=df_results.stypes.values
x = np.arange(len(stypes)) 
width=0.4
plt.bar(x-0.2, df_results.count_hab, yerr=df_results.error_hab, width=width)#, color='cyan') 
plt.bar(x+0.2, df_results.count_unhab, yerr=df_results.error_unhab, width=width)#, color='orange') 
plt.xticks(x, stypes) 
plt.xlabel("Stellar Type") 
plt.ylabel("Count") 
plt.title('HWO Detectability')
plt.legend(["Habitable", "Inhabitable"]) 
plt.show() 
a=1