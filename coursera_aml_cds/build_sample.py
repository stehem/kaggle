import numpy as np

def build_sample_f(window, training,features):
    #groups
    lstm_data = []
    lstm_y = []
    upper_limit = len(window) - 1 


    print(window)
    a = training[training['date_block_num'].isin(window)][['shop_item_cnt_block'] + ['date_block_num','item_id', 'shop_id'] + features]\
                    .sort_values(by=["date_block_num"]).groupby(["item_id", "shop_id"])

    for name, group in a:
        #print(group.values)
        steps = []
        ys = []
        #print("group.values",group.values)
        for step in group.values:
            #print(step)
            #print("step", len(step))
            #step is np.array 
            #step[4:] is np.array print(type(step[4:]))
            steps.append(step[4:])
            #print(step[9])
            #print(step[0])
            ys.append(step[0])
        #remove last
        #print(type(steps[0:8]))
        #jan to sept, y = oct
        lstm_data.append(np.array(steps[0:upper_limit]))
        #remove first
        #print(ys)
        #preditct for setptember
        lstm_y.append(ys[-1])

    return lstm_data, np.array([np.array([y]) for y in lstm_y])