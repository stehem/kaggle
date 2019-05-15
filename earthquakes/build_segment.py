import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler 

#https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#autocorrelation
def autocorrelation(x, lag):
    # Slice the relevant subseries based on the lag
    y1 = x[:(len(x)-lag)]
    y2 = x[lag:]
    # Subtract the mean of the whole series x
    x_mean = np.mean(x)
    # The result is sometimes referred to as "covariation"
    sum_product = np.sum((y1 - x_mean) * (y2 - x_mean))
    # Return the normalized unbiased covariance
    v = np.var(x)
    if np.isclose(v, 0):
        return np.NaN
    else:
        return sum_product / ((len(x) - lag) * v)
    
#https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#cid_ce
def cid_ce(x, normalize):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if normalize:
        s = np.std(x)
        if s!=0:
            x = (x - np.mean(x))/s
        else:
            return 0.0

    x = np.diff(x)
    return np.sqrt(np.dot(x, x))

#https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#kurtosis
def kurtosis(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.kurtosis(x)


#https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#percentage_of_reoccurring_datapoints_to_all_datapoints
def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    if len(x) == 0:
        return np.nan

    unique, counts = np.unique(x, return_counts=True)

    if counts.shape[0] == 0:
        return 0

    return np.sum(counts > 1) / float(counts.shape[0])

#https://tsfresh.readthedocs.io/en/latest/_modules/tsfresh/feature_extraction/feature_calculators.html#ratio_beyond_r_sigma
def ratio_beyond_r_sigma(x, r):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.sum(np.abs(x - np.mean(x)) > r * np.std(x))/x.size

def ratio_value_number_to_time_series_length(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if x.size == 0:
        return np.nan

    return np.unique(x).size / x.size


def skewness(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)
    return pd.Series.skew(x)


def build_training_sample(df: pd.DataFrame, number_of_groups: int) -> pd.DataFrame:
    acoustic_data = np.array_split(df['acoustic_data'].astype("float").values, number_of_groups)

    data = np.array([(
        np.mean(sub),
        np.max(sub),
        np.min(sub),
        np.std(sub),
        np.absolute(np.max(sub) - np.min(sub)),
        np.percentile(sub, 25),
        np.percentile(sub, 50),
        np.percentile(sub, 75),
        np.sum(sub),
        len(set(sub)),
        len(sub[sub > 0]),
        len(sub[sub < 0]),
        np.sum(np.abs(sub)),
        #len(sub[sub > np.mean(sub)]),
        #len(sub[sub < np.mean(sub)]),
        np.argmax(sub),
        np.argmin(sub),
        #np.median(sub),
        np.dot(sub,sub),
        np.sum(np.abs(np.ediff1d(sub))),
        autocorrelation(sub,10),
        cid_ce(sub,True),
        kurtosis(sub),
        np.mean(np.ediff1d(sub)),
        percentage_of_reoccurring_datapoints_to_all_datapoints(sub),
        ratio_beyond_r_sigma(sub,2),
        ratio_value_number_to_time_series_length(sub),
        skewness(sub)
    ) for sub in acoustic_data])

    df2 = pd.DataFrame(data)
    df2.columns = ["mean", "max", "min", "std", "abs", "q25", "q50", "q75",\
                  "sum","uniq",
                   "pos","negs", 
                   "ssum", 
                   #"gtmean", "ltmean",\
                  "imax", "imin", 
                   #"median",
                   "abs_nrg",
                   "abs_sum_chg",
                   "autocorr_10",
                   "cid_ce",
                   "kurtosis",
                   "mean_chg",
                   "reocurring_pct",
                   "r_sigma",
                   "ratio_to_length",
                   "skewness"
                  ]


    columns = df2.columns.values
    df2[columns] = StandardScaler().fit_transform(df2[columns])
    df2['rolling_10'] = df2['mean'].rolling(window=10, min_periods=1).mean()
    df2['rolling_25'] = df2['mean'].rolling(window=25, min_periods=1).mean()
    df2['lag_10'] = df2['mean'].shift(10)
    df2['lag_25'] = df2['mean'].shift(25)

    df2.fillna(df2.mean(),inplace=True)
    
    return df2

    


def build_segment_f(splits, number_of_groups,test=False, augment=False):
    dfs = []
    splits = sorted(splits)
    
    for i,segment in enumerate(splits):
        if i % 100 == 0:
            print(i)
        root = 'train'
        columns = ['acoustic_data','time_to_failure']
        header=None
        if test:
            root = 'test'
            columns = ['acoustic_data']
            header=0
                     
        path = '%s/%s' % (root,segment)
        #
        
        df = pd.read_csv(path, float_precision='round_trip', header=header)
        df.columns = columns
        #
        df2 = build_training_sample(df, number_of_groups)
        
        if test:
            df2["seg_id"] = segment
        else:
            df2['time_to_failure'] = df['time_to_failure'].values[-1]
            
        dfs.append(df2)
        
        ##
        if augment and not test and i > 0:
        
            previous_split = splits[i-1]
            previous_path = '%s/%s' % (root,previous_split)

            previous_df = pd.read_csv(previous_path, float_precision='round_trip', header=header)
            previous_df.columns = columns

            df_ = pd.DataFrame(previous_df[75000:150001].values)
            df_.columns = columns
            df_ = df_.append(df[0:75000]).reset_index()

            df3 = build_training_sample(df_, number_of_groups)

            if test:
                df3["seg_id"] = segment
            else:
                df3['time_to_failure'] = df_['time_to_failure'].values[-1]

            dfs.append(df3)
             
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