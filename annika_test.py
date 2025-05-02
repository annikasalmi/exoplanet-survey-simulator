import pandas as pd
import numpy as np

for i in range(3):
    stype = ['A', 'B', 'C', 'D']
    count_overall = [np.random.randint(0, 100) for _ in range(4)]
    df = pd.DataFrame(data={'stype': stype, 'count_overall': count_overall})
    
    temp_dict = dict(np.asarray(df))

    if i == 0:
        total_dict = temp_dict
    else:
        for key in temp_dict:
            list_dict = list([total_dict[key]])
            total_dict[key]= list_dict.append(temp_dict[key])
            a=1
