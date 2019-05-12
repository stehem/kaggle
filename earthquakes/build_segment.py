import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler 


SEGMENT_LENGTH=150000

def build_segment_f(splits, timesteps):
    dfs = []
    for i,segment in enumerate(splits):
        #if i % 100 == 0:
            #print(i)
        path = 'train/%s' % (segment)
        #
        columns = ['acoustic_data','time_to_failure']
        df = pd.read_csv(path, float_precision='round_trip', header=None)
        df.columns = columns
        #
        #last_n=int(SEGMENT_LENGTH/timesteps/4)
        acoustic_data = np.array_split(df['acoustic_data'].values, timesteps)
        #means = np.array([np.mean(sub[-last_n:], dtype=np.float64) for sub in acoustic_data])
        means = np.array([np.mean(sub, dtype=np.float64) for sub in acoustic_data])
        target = df['time_to_failure'].values[-1]
        #
        df2 = pd.DataFrame(data=[means]).T
        df2.columns = ['acoustic_data']
        #df2['rolling_10'] = df['acoustic_data'].rolling(window=10, min_periods=1).mean()
        #df2['rolling_50'] = df['acoustic_data'].rolling(window=50, min_periods=1).mean()
        #df2['rolling_100'] = df['acoustic_data'].rolling(window=100, min_periods=1).mean()
        #df2['min_10'] = df['acoustic_data'].rolling(window=10, min_periods=1).min()
        #df2['max_10'] = df['acoustic_data'].rolling(window=10, min_periods=1).max()
        #df2['std_10'] = df['acoustic_data'].rolling(window=10, min_periods=1).std()
        df2.fillna(df2.mean(),inplace=True)
        columns = df2.columns.values
        df2[columns] = StandardScaler().fit_transform(df2[columns])
        df2['time_to_failure'] = target
        dfs.append(df2)                  
    return dfs



def build_segment_g(splits):
    dfs = []
    for i,segment in enumerate(splits):
        #if i % 100 == 0:
            #print(i)
        path = 'train/%s' % (segment)
        #
        columns = ['acoustic_data','time_to_failure']
        df = pd.read_csv(path, float_precision='round_trip', header=None)
        if len(df) != SEGMENT_LENGTH:
            print(segment)
            continue
        df.columns = columns
        #
        df['acoustic_data'] = pd.to_numeric(df['acoustic_data'], downcast='signed')
        #last_n=int(SEGMENT_LENGTH/timesteps/4)
     
        dfs.append(df['acoustic_data'])                  
    return dfs