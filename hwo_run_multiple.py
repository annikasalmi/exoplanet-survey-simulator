import matplotlib.pyplot as plt
import numpy as np
import time
import os
import pandas as pd
import multiprocessing as mp

from lifesim.core.hwo_data import HWOData
from ppop_generator import PPop
from tools import PPOP_DATA_DIR


def run_single(i):
    '''
    Runs a single instance of the PPop simulation and HWO data analysis.
    '''
    PPopObj = PPop() # i guess we'll reinstantiate each run...

    filename = f'test_runs_{i}'
    data_path = os.path.join(PPOP_DATA_DIR, filename)

    df = PPopObj.run_ppop(data_path, ntest=100, nuniverses=1)
    PPopObj.catalog_from_ppop(data_path, df=df)
    PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.0)
    PPopObj.catalog_remove_distance(stype='M', mode='larger', dist=10.0)

    hwo_data = HWOData(PPopObj.catalog)
    hwo_data.determine_detectable()
    df_hab, df_false = hwo_data.organize_data()

    return {
        'hab_values': df_hab.count_overall.values,
        'hab_stypes': df_hab.stype.tolist(),
        'false_values': df_false.count_overall.values,
        'false_stypes': df_false.stype.tolist()
    }

def main():
    start = time.time()

    indices = list(range(3))
    with mp.Pool(processes=mp.cpu_count()) as pool:
        results = pool.map(run_single, indices)

    # Collect first result for column naming
    df_hab_total = pd.DataFrame([r['hab_values'] for r in results])
    df_hab_total.columns = results[0]['hab_stypes']

    df_false_total = pd.DataFrame([r['false_values'] for r in results])
    df_false_total.columns = results[0]['false_stypes']

    # Build results summary
    df_results = pd.DataFrame(columns=['stypes', 'count_hab', 'error_hab', 'count_unhab', 'error_unhab'])
    for stype in df_hab_total.columns:
        count = np.mean(df_hab_total[stype])
        err = np.std(df_hab_total[stype])
        count_unhab = np.mean(df_false_total[stype])
        err_unhab = np.std(df_false_total[stype])
        df = pd.DataFrame({
            'stypes': [stype],
            'count_hab': [count],
            'error_hab': [err],
            'count_unhab': [count_unhab],
            'error_unhab': [err_unhab]
        })
        df_results = pd.concat([df_results, df], ignore_index=True)

    # Save results
    df_results.to_csv(os.path.join(PPOP_DATA_DIR, 'hwo_results.csv'), index=False)

    # Plotting
    stypes = df_results.stypes.values
    x = np.arange(len(stypes))
    width = 0.4
    plt.bar(x - 0.2, df_results.count_hab, yerr=df_results.error_hab, width=width, label='Habitable')
    plt.bar(x + 0.2, df_results.count_unhab, yerr=df_results.error_unhab, width=width, label='Inhabitable')
    plt.xticks(x, stypes)
    plt.xlabel("Stellar Type")
    plt.ylabel("Count")
    plt.title('HWO Detectability')
    plt.legend()
    plt.tight_layout()
    plt.show()

    print(f"Total time: {time.time() - start:.2f} seconds")

if __name__ == '__main__':
    mp.set_start_method("spawn")  # especially important for macOS/Windows
    main()

# df_results = pd.DataFrame(columns=['stypes', 'count_hab', 'error_hab', 'count_unhab', 'error_unhab'])
# for i in df_hab_total.columns:
#     count = np.mean(df_hab_total[i])
#     err = np.std(df_hab_total[i])
#     count_unhab = np.mean(df_false_total[i])
#     err_unhab = np.std(df_false_total[i])
#     df = pd.DataFrame(data={'stypes': [i], 'count_hab': [count], 'error_hab': [err], 
#                                         'count_unhab': [count_unhab], 'error_unhab': [err_unhab]})
#     df_results = pd.concat([df_results,df], ignore_index=True)

# df_results.to_csv(os.path.join(PPOP_DATA_DIR,'hwo_results.csv'), index=False)
    
# stypes=df_results.stypes.values
# x = np.arange(len(stypes)) 
# width=0.4
# plt.bar(x-0.2, df_results.count_hab, yerr=df_results.error_hab, width=width)#, color='cyan') 
# plt.bar(x+0.2, df_results.count_unhab, yerr=df_results.error_unhab, width=width)#, color='orange') 
# plt.xticks(x, stypes)
# plt.xlabel("Stellar Type") 
# plt.ylabel("Count") 
# plt.title('HWO Detectability')
# plt.legend(["Habitable", "Inhabitable"]) 
# plt.show() 
# a=1



# PPopObj = PPop()
# for i in range(3):
#     t=time.time()
#     filename = 'test_runs_' + str(i) # str
#     data_path = os.path.join(PPOP_DATA_DIR, filename)
#     df = PPopObj.run_ppop(data_path, ntest=100, nuniverses=1)
#     PPopObj.catalog_from_ppop(data_path, df=df)
#     PPopObj.catalog_remove_distance(stype='A', mode='larger', dist=0.)  # remove all A stars
#     PPopObj.catalog_remove_distance(stype='M', mode='larger', dist=10.)  # remove M stars > 10pc to

#     hwo_data = HWOData(PPopObj.catalog)

#     hwo_data.determine_detectable()

#     df_hab, df_false = hwo_data.organize_data()

#     if i == 0:
#         mapping_hab = zip(df_hab.T.columns, df_hab.stype)
#         mapping_unhab = zip(df_false.T.columns, df_false.stype)
#         df_hab_total = df_hab.T.rename(columns=dict(mapping_hab)).drop('stype').reset_index(drop=True)
#         df_false_total = df_false.T.rename(columns=dict(mapping_unhab)).drop('stype').reset_index(drop=True)
#     else:
#         df_hab_total.loc[len(df_hab_total)] = df_hab.count_overall.values
#         df_false_total.loc[len(df_false_total)] = df_false.count_overall.values