# import os
# import sys
# import pandas as pd
# import numpy as np


# file_list = os.listdir('CSV')
# os.mkdir('NPY')

# for files in file_list:
#     file_info = files.split('.')
#     print(file_info[0])
#     dataset = pd.read_csv(f"CSV/{files}")
#     X = dataset.to_numpy()[:, 1:]
#     np.save(f"NPY/{file_info[0]}.npy", X)

import os
import sys
import pandas as pd
import numpy as np


file_list = os.listdir('CSV')
os.mkdir('CSV2')
os.mkdir('NPY2')

for files in file_list:
    file_info = files.split('.')
    print(file_info[0])
    dataset = pd.read_csv(f"CSV/{files}")
    first_column = dataset.columns[0]
    dataset = dataset.drop([first_column], axis=1)
    X = dataset.to_numpy()
    np.save(f"NPY2/{file_info[0]}.npy", X)
    dataset.to_csv(f"CSV2/{file_info[0]}.csv", index=False)
