import numpy as np

def build_test_lr_f(interval, test, training_test, features):
    print(interval)
    test_lstm_data = []
    for index, row in test[interval[0]:interval[1]].iterrows():
        #if index % 10000 == 0:
        #print(index)
        line = training_test[(training_test['shop_id'] == row['shop_id']) &\
                             (training_test['item_id'] == row['item_id'])].sort_values(by=['date_block_num'])[features]
        avgs = []
        for feature in features:
            feature_avg = np.average(line[feature].values, weights=[1,2,4,8,16,32,64], axis=0)
            avgs.append(feature_avg)
       # if len(line) != 10:
            #print(len(line), row['shop_id'], row['item_id'])
        #print(line)
        test_lstm_data.append(avgs)
        
    return {interval: test_lstm_data}