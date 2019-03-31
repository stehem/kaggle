
def build_test_f(interval, test, training_test, features, window_size):
    print(interval)
    test_lstm_data = []
    for index, row in test[interval[0]:interval[1]].iterrows():
        #if index % 10000 == 0:
        #print(index)
        line = training_test[(training_test['shop_id'] == row['shop_id']) & (training_test['item_id'] == row['item_id'])]\
                .sort_values(by=['date_block_num'])[features].values
       # if len(line) != 10:
            #print(len(line), row['shop_id'], row['item_id'])
        #print(line)
        test_lstm_data.append(line[1:window_size+1])
        
    return {interval: test_lstm_data}