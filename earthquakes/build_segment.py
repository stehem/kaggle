import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler,MinMaxScaler 
import itertools


def _roll(a, shift):
    if not isinstance(a, np.ndarray):
        a = np.asarray(a)
    idx = shift % len(a)
    return np.concatenate([a[-idx:], a[:-idx]])

def _get_length_sequences_where(x):
    if len(x) == 0:
        return [0]
    else:
        res = [len(list(group)) for value, group in itertools.groupby(x) if value == 1]
        return res if len(res) > 0 else [0]

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

def longest_strike_below_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x <= np.mean(x))) if x.size > 0 else 0


def longest_strike_above_mean(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.max(_get_length_sequences_where(x >= np.mean(x))) if x.size > 0 else 0


def count_above_mean(x):
    m = np.mean(x)
    return np.where(x > m)[0].size


def count_below_mean(x):
    m = np.mean(x)
    return np.where(x < m)[0].size


def last_location_of_maximum(x):
    x = np.asarray(x)
    return 1.0 - np.argmax(x[::-1]) / len(x) if len(x) > 0 else np.NaN


def first_location_of_maximum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmax(x) / len(x) if len(x) > 0 else np.NaN


def last_location_of_minimum(x):
    x = np.asarray(x)
    return 1.0 - np.argmin(x[::-1]) / len(x) if len(x) > 0 else np.NaN


def first_location_of_minimum(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    return np.argmin(x) / len(x) if len(x) > 0 else np.NaN


def percentage_of_reoccurring_datapoints_to_all_datapoints(x):
    if len(x) == 0:
        return np.nan

    unique, counts = np.unique(x, return_counts=True)

    if counts.shape[0] == 0:
        return 0

    return np.sum(counts > 1) / float(counts.shape[0])



def percentage_of_reoccurring_values_to_all_values(x):
    if not isinstance(x, pd.Series):
        x = pd.Series(x)

    if x.size == 0:
        return np.nan

    value_counts = x.value_counts()
    reoccuring_values = value_counts[value_counts > 1].sum()

    if np.isnan(reoccuring_values):
        return 0

    return reoccuring_values / x.size



def sum_of_reoccurring_values(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    counts[counts > 1] = 1
    return np.sum(counts * unique)



def sum_of_reoccurring_data_points(x):
    unique, counts = np.unique(x, return_counts=True)
    counts[counts < 2] = 0
    return np.sum(counts * unique)


def ratio_value_number_to_time_series_length(x):
    if not isinstance(x, (np.ndarray, pd.Series)):
        x = np.asarray(x)
    if x.size == 0:
        return np.nan
    return np.unique(x).size / x.size

def number_peaks(x, n):
    """
    Calculates the number of peaks of at least support n in the time series x. A peak of support n is defined as a
    subsequence of x where a value occurs, which is bigger than its n neighbours to the left and to the right.

    Hence in the sequence

    >>> x = [3, 0, 0, 4, 0, 0, 13]

    4 is a peak of support 1 and 2 because in the subsequences

    >>> [0, 4, 0]
    >>> [0, 0, 4, 0, 0]

    4 is still the highest value. Here, 4 is not a peak of support 3 because 13 is the 3th neighbour to the right of 4
    and its bigger than 4.

    :param x: the time series to calculate the feature of
    :type x: numpy.ndarray
    :param n: the support of the peak
    :type n: int
    :return: the value of this feature
    :return type: float
    """
    x_reduced = x[n:-n]

    res = None
    for i in range(1, n + 1):
        result_first = (x_reduced > _roll(x, i)[n:-n])

        if res is None:
            res = result_first
        else:
            res &= result_first

        res &= (x_reduced > _roll(x, -i)[n:-n])
    return np.sum(res)

def sample_entropy(x):
    x = np.array(x)

    sample_length = 1 # number of sequential points of the time series
    tolerance = 0.2 * np.std(x) # 0.2 is a common value for r - why?

    n = len(x)
    prev = np.zeros(n)
    curr = np.zeros(n)
    A = np.zeros((1, 1))  # number of matches for m = [1,...,template_length - 1]
    B = np.zeros((1, 1))  # number of matches for m = [1,...,template_length]

    for i in range(n - 1):
        nj = n - i - 1
        ts1 = x[i]
        for jj in range(nj):
            j = jj + i + 1
            if abs(x[j] - ts1) < tolerance:  # distance between two vectors
                curr[jj] = prev[jj] + 1
                temp_ts_length = min(sample_length, curr[jj])
                for m in range(int(temp_ts_length)):
                    A[m] += 1
                    if j < n - 1:
                        B[m] += 1
            else:
                curr[jj] = 0
        for j in range(nj):
            prev[j] = curr[j]

    N = n * (n - 1) / 2
    B = np.vstack(([N], B[0]))

    # sample entropy = -1 * (log (A/B))
    similarity_ratio = A / B
    se = -1 * np.log(similarity_ratio)
    se = np.reshape(se, -1)
    return se[0]


def build_training_sample(df: pd.DataFrame, number_of_groups: int, scale: bool) -> pd.DataFrame:
    features = ['acoustic_data','roll_diff_1']
    all_dfs = []

    for feature in features:
      
        acoustic_data = np.array_split(df[feature].astype("float").values, number_of_groups)

        data = np.array([(
            np.median(sub),
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
            skewness(sub),
            #longest_strike_below_mean(sub),
            #longest_strike_above_mean(sub),
            #last_location_of_maximum(sub),
            #first_location_of_maximum(sub),
            #last_location_of_minimum(sub),
            #first_location_of_minimum(sub),
            percentage_of_reoccurring_datapoints_to_all_datapoints(sub),
            percentage_of_reoccurring_values_to_all_values(sub),
            sum_of_reoccurring_values(sub),
            sum_of_reoccurring_data_points(sub),
            #ratio_value_number_to_time_series_length(sub),
            #number_peaks(sub,1000),
            #sample_entropy(sub),
            np.mean(sub[0:25000]),
            np.mean(sub[-25000:]),
            np.absolute(np.max(sub[-25000:]) - np.min(sub[0:25000])),
            len(sub[0:25000][sub[0:25000] > 0]),
            len(sub[0:25000][sub[0:25000] < 0]),
            len(sub[-25000:][sub[-25000:] > 0]),
            len(sub[-25000:][sub[-25000:] < 0]),
        ) for sub in acoustic_data])

        dff = pd.DataFrame(data)

        dff.columns = [f"{feature}_median", f"{feature}_mean", f"{feature}_max", f"{feature}_min", f"{feature}_std", 
                       f"{feature}_abs", f"{feature}_q25", f"{feature}_q50", f"{feature}_q75",\
                      f"{feature}_sum",f"{feature}_uniq",
                       f"{feature}_pos",f"{feature}_negs", 
                       f"{feature}_ssum", 
                       #"gtmean", "ltmean",\
                      f"{feature}_imax", f"{feature}_imin", 
                       #"median",
                       f"{feature}_abs_nrg",
                       f"{feature}_abs_sum_chg",
                       f"{feature}_autocorr_10",
                       f"{feature}_cid_ce",
                       f"{feature}_kurtosis",
                       f"{feature}_mean_chg",
                       f"{feature}_reocurring_pct",
                       f"{feature}_r_sigma",
                       f"{feature}_ratio_to_length",
                       f"{feature}_skewness",
                       #"strike_below",
                       #"strike_above",
                       #"last_loc_max",
                       #"first_loc_max",
                       #"last_loc_min",
                       #"first_loc_min",
                       f"{feature}_perc_reocurr_dp",
                       f"{feature}_perc_reocurr_all",
                       f"{feature}_sum_reoccurr_val",
                       f"{feature}_sum_reoccurr_dp",
                       #"ratio_value_number",
                       #"peaks",
                       f"{feature}_mean_head",
                       f"{feature}_mean_tail",
                       f"{feature}_abs_diff_head_tail",
                       f"{feature}_pos_head",
                       f"{feature}_neg_head",
                       f"{feature}_pos_tail",
                       f"{feature}_neg_tail",
                       #"entropy"
                      ]

        all_dfs.append(dff)
    
    df2 = pd.concat(all_dfs, axis=1)
    columns = df2.columns.values
    if scale:
        df2[columns] = StandardScaler().fit_transform(df2[columns])
    #df2[columns] = MinMaxScaler(feature_range =(-1, 1)).fit_transform(df2[columns])
    #df2['rolling_10'] = df2['mean'].rolling(window=10, min_periods=1).mean()
    #df2['rolling_25'] = df2['mean'].rolling(window=25, min_periods=1).mean()
    #df2['lag_10'] = df2['mean'].shift(10)
    #df2['lag_25'] = df2['mean'].shift(25)

    #df2.fillna(df2.mean(),inplace=True)
    
    return df2


def add_noise(df, pct):
    mu = df['acoustic_data'].mean()
    sigma = df['acoustic_data'].std()

    indices = np.random.choice(df.index.values, int(len(df)*pct))
    df.loc[indices, 'acoustic_data'] = np.random.normal(mu, sigma, len(indices)) 
    return df


def build_segment_f(splits, number_of_groups,test=False, augment=False, scale=True, noise=0.5, smart_augment=False):
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
        df['roll_1000'] = df['acoustic_data'].rolling(1000,min_periods=1000).mean()
        df['roll_1000'].fillna(df['roll_1000'][1000],inplace=True)
        df['shifted_1'] = df['roll_1000'].shift(1)
        df['shifted_1'].fillna(df['roll_1000'][1000],inplace=True)
        df["roll_diff_1"] = df['shifted_1'] - df['roll_1000']
        #df['shifted_1000'] = df['roll_1000'].shift(1000)
        #df['shifted_1000'].fillna(df['roll_1000'][1000],inplace=True)
        #df["roll_diff_1000"] = df['shifted_1000'] - df['roll_1000']
        #
        df2 = build_training_sample(df, number_of_groups, scale)
        
        if test:
            df2["seg_id"] = segment
        else:
            df2['time_to_failure'] = df['time_to_failure'].values[-1]
            df2['augmented'] = False
            #df2['segment'] = segment
            
        dfs.append(df2)
        
        ##
        if augment and not test and i > 0:
        
            previous_split = splits[i-1]
            previous_path = '%s/%s' % (root,previous_split)

            previous_df = pd.read_csv(previous_path, float_precision='round_trip', header=header)
            previous_df.columns = columns

            previous_df = add_noise(previous_df, noise)
            previous_df['roll_1000'] = previous_df['acoustic_data'].rolling(1000,min_periods=1000).mean()
            previous_df['roll_1000'].fillna(previous_df['roll_1000'][1000],inplace=True)
            previous_df['shifted_1'] = previous_df['roll_1000'].shift(1)
            previous_df['shifted_1'].fillna(previous_df['roll_1000'][1000],inplace=True)
            previous_df["roll_diff_1"] = previous_df['shifted_1'] - previous_df['roll_1000']

            df_ = pd.DataFrame(previous_df[75000:150001].values)
            df_.columns = previous_df.columns
            noise_df = add_noise(df, noise)
            noise_df['roll_1000'] = noise_df['acoustic_data'].rolling(1000,min_periods=1000).mean()
            noise_df['roll_1000'].fillna(noise_df['roll_1000'][1000],inplace=True)
            noise_df['shifted_1'] = noise_df['roll_1000'].shift(1)
            noise_df['shifted_1'].fillna(noise_df['roll_1000'][1000],inplace=True)
            noise_df["roll_diff_1"] = noise_df['shifted_1'] - noise_df['roll_1000']
            df_ = df_.append(noise_df[0:75000]).reset_index()

            df3 = build_training_sample(df_, number_of_groups, scale)
            df3['time_to_failure'] = df_['time_to_failure'].values[-1]
            df3['augmented'] = True
            #df3['segment'] = segment


            dfs.append(df3)
            
        if smart_augment and not test and i > 0:
        
            df_ttf = df['time_to_failure'].values[-1]
            if df_ttf < 10:
                continue
                         
            
            if df_ttf < 12:
                 for i in range(0,1):
                        noisy = add_noise(df, noise)
                        noisy['roll_1000'] = noisy['acoustic_data'].rolling(1000,min_periods=1000).mean()
                        noisy['roll_1000'].fillna(noisy['roll_1000'][1000],inplace=True)
                        noisy['shifted_1'] = noisy['roll_1000'].shift(1)
                        noisy['shifted_1'].fillna(noisy['roll_1000'][1000],inplace=True)
                        noisy["roll_diff_1"] = noisy['shifted_1'] - noisy['roll_1000']
                        noisy = build_training_sample(noisy, number_of_groups, scale)
                        noisy['time_to_failure'] = df['time_to_failure'].values[-1]
                        noisy['augmented'] = True
                        dfs.append(noisy)
                        
            elif df_ttf < 14:
                 for i in range(0,4):
                        noisy = add_noise(df, noise)
                        noisy['roll_1000'] = noisy['acoustic_data'].rolling(1000,min_periods=1000).mean()
                        noisy['roll_1000'].fillna(noisy['roll_1000'][1000],inplace=True)
                        noisy['shifted_1'] = noisy['roll_1000'].shift(1)
                        noisy['shifted_1'].fillna(noisy['roll_1000'][1000],inplace=True)
                        noisy["roll_diff_1"] = noisy['shifted_1'] - noisy['roll_1000']
                        noisy = build_training_sample(noisy, number_of_groups, scale)
                        noisy['time_to_failure'] = df['time_to_failure'].values[-1]
                        noisy['augmented'] = True
                        dfs.append(noisy)
            else:
                 for i in range(0,10):
                        noisy = add_noise(df, noise)
                        noisy['roll_1000'] = noisy['acoustic_data'].rolling(1000,min_periods=1000).mean()
                        noisy['roll_1000'].fillna(noisy['roll_1000'][1000],inplace=True)
                        noisy['shifted_1'] = noisy['roll_1000'].shift(1)
                        noisy['shifted_1'].fillna(noisy['roll_1000'][1000],inplace=True)
                        noisy["roll_diff_1"] = noisy['shifted_1'] - noisy['roll_1000']
                        noisy = build_training_sample(noisy, number_of_groups, scale)
                        noisy['time_to_failure'] = df['time_to_failure'].values[-1]
                        noisy['augmented'] = True
                        dfs.append(noisy)
             
    return dfs