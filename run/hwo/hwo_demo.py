import requests
import os
import lifesim
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

from lifesim.core.hwo_data import HWOData

# ---------- Set-Up ----------

# create bus
bus = lifesim.Bus()

# setting the options
bus.data.options.set_scenario('baseline')

# set options manually
bus.data.options.set_manual(diameter=4.)
bus.data.options.set_manual(output_path='data_creation/')
bus.data.options.set_manual(output_filename='test_runs_0')

# ---------- Downloading the P-Pop catalog ----------

data = requests.get('https://raw.githubusercontent.com/kammerje/P-pop/main/TestPlanetPopulation.txt')

if os.path.isdir('data_creation'):
    pass
else:
    os.makedirs('data_creation')
if os.path.exists(os.path.join('data_creation','ppop_catalog.txt')):
    pass
else:
    with open(os.path.join('data_creation','ppop_catalog.txt'), 'wb') as file:
        file.write(data.content)

# ---------- Loading the Catalog ----------

bus.data.catalog_from_ppop(input_path='data_creation/ppop_catalog.txt')
bus.data.catalog_remove_distance(stype='A', mode='larger', dist=0.)  # remove all A stars
bus.data.catalog_remove_distance(stype='M', mode='larger', dist=10.)  # remove M stars > 10pc to
# speed up calculation

hwo_data = HWOData(bus.data)
hwo_data.determine_detectable(
    use_exozodi_constraint=True,           # Enable exozodi constraint
    exozodi_scenario='baseline',           # Use baseline exozodi scenario
    use_surface_brightness_criterion=True, # Use new surface brightness criterion
    ignore_exozodi_rejections=False        # Apply exozodi rejections to final detection
)

df_size = hwo_data.catalog.groupby(['stype','habitable']).size().reset_index()
df_size['count_overall'] = df_size[0]
df_size =df_size.drop([0], axis=1)

x = np.arange(len(df_size.stype.unique())) 

df_hab = df_size[df_size.habitable==True].drop(['habitable'],axis=1)
df_false = df_size[df_size.habitable==False].drop(['habitable'],axis=1)
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