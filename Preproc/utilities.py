'''these utilities takes the raw data as input, and applies data preprocessing so that it could be used in the model
This is a time series data. A sliding window approach is used to create the input. The window length is set to be 20. Therefore, each training example will be 20 time steps...
...of the variables defining a market stock. In this model, market stocks form the nodes and the price and other indicator time series form the node features.'''

import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta


def rawData_collate(raw_csv_name, company_name):  #csv name with full directory path
    df1 = pd.read_csv('/content/drive/MyDrive/Stock_data-FMP_API/sentiment_data_142Stocks_5yr.csv')
    df = pd.read_csv('/content/drive/MyDrive/Stock_data-FMP_API/' + raw_csv_name)
    #creating the numpy matrix in the shape n_timesteps * n_stocks * n_variables to be input to the dataset creator for spektral
    temp_lst = []
    df_date = []
    for date_df in df['Date']:
        df_date.append(str(date_df)[0:10])
    df1_date = []
    for date_df1 in df1['Date']:
        df1_date.append(str(date_df1)[0:10])

    df = df.drop('Date', 1)
    df1 = df1.drop('Date', 1)

    df['date1'] = pd.Series(df_date)
    df1['date1'] = pd.Series(df1_date)
    for prep_nam in company_name:
        sent = []
        for dt in df_date:
            if dt in df1_date:
                indx = df1[df1['date1'] == dt].index.values[0]
                sent.append(df1[prep_nam][indx])
            else:
                k_d = 0
                sent.append(k_d)
                continue
        sent = pd.Series(sent)
        df['Sentiment'] = sent
        feat_arr = df[['Adj Close', 'Volume', 'Sentiment']].to_numpy()
        temp_nparray = np.reshape(df['company_name'].to_numpy(), (df['company_name'].to_numpy().shape[0],1))
        prep_index = np.where(temp_nparray[:,0] == prep_nam)[0]
        arr = feat_arr[prep_index,:]
        temp_lst.append(np.reshape(arr,(arr.shape[0],1,feat_arr.shape[1])))
    
    i_cncat = 0
    while i_cncat < len(temp_lst) - 1:
        if i_cncat == 0:
            data_arr = np.concatenate((temp_lst[i_cncat],temp_lst[i_cncat+1]), axis = 1)
            i_cncat += 2
        else:
            data_arr = np.concatenate((data_arr, temp_lst[i_cncat]), axis = 1)
            i_cncat += 1

    #creating the feature : LogR
    logR_arr = np.log(np.divide(data_arr[1:data_arr.shape[0], :, 0], data_arr[0:(data_arr.shape[0]-1), :, 0]))
    logR_arr = np.reshape(logR_arr, (logR_arr.shape[0], logR_arr.shape[1], 1))

    rawData_arr = np.concatenate((logR_arr, np.reshape(data_arr[1:data_arr.shape[0], :, 1:3], (logR_arr.shape[0], logR_arr.shape[1], 2))), axis = 2)

    print('The return vector contains data in the format: n_timesteps * n_stocks * n_variables \n')
    return rawData_arr




def sliding_window_data(rawData, window_size):
    s_arr = []
    label_arr = []
    cnt = 0
    for i in range(window_size, rawData.shape[0]):
        temp_d = rawData[(i-window_size):i, :, :]
        l1 = np.where(rawData[i, :, 3] >= 0, 1, 0)
        l2 = np.where(rawData[i, :, 3] < 0, 0, 0)
        label = np.squeeze((l1 + l2))
        s_arr.append(temp_d)
        label_arr.append(label)
        cnt += 1

    s_arr = np.array(s_arr)
    label_arr = np.reshape(np.array(label_arr), (cnt, rawData.shape[1]))
    return s_arr, label_arr



def normalize(x, y, stats=None, train=False, std_norm = False):
    if std_norm:
        if train:
            avg_x = np.mean(x, axis=0, keepdims=True)
            std_x = np.std(x, axis=0, keepdims=True)
            avg_y = np.mean(y, axis=0, keepdims=True)
            std_y = np.std(y, axis=0, keepdims=True)
            x -= avg_x
            x /= std_x
            #y -= avg_y
            #y /= std_y
            return x, y, [[avg_x, std_x], [avg_y, std_y]]
        else:
            x -= stats[0][0]
            x /= stats[0][1]
            #y -= stats[1][1]
            #y /= stats[1][0]
            return x, y
    else:
        if train:
            max_x = np.amax(x, axis=0, keepdims=True)
            min_x = np.amin(x, axis=0, keepdims=True)
            max_y = np.amax(y, axis=0, keepdims=True)
            min_y = np.amin(y, axis=0, keepdims=True)
            x -= min_x
            x /= (max_x - min_x)
            #y -= min_y
            #y /= (max_y - min_y)
            return x, y, [[max_x, min_x], [max_y, min_y]]
        else:
            x -= stats[0][1]
            x /= (stats[0][0] - stats[0][1])
            #y -= stats[1][1]
            #y /= (stats[1][0] - stats[1][1])
            return x, y


def inv_tran(y, stats):
    y *= stats[1][1]
    y += stats[1][0]
    return y


def data_split(s_arr, label_arr, split):
    #train:val:test split
    data = s_arr
    label = label_arr
    train_split = int(split[0]*data.shape[0])
    val_split = int((split[1]+split[0])*data.shape[0])
    test_split = data.shape[0]
    t_xtrain, t_xval, t_xtest = data[0:train_split], data[train_split:val_split], data[val_split:test_split]
    t_ytrain, t_yval, t_ytest = label[0:train_split], label[train_split:val_split], label[val_split:test_split]
    t_xtrain, t_ytrain, stats= normalize(t_xtrain, t_ytrain, train=True)
    t_xval, t_yval = normalize(t_xval, t_yval, stats=stats, train=False)
    t_xtest, t_ytest = normalize(t_xtest, t_ytest, stats=stats, train=False)
    return [t_xtrain, t_xval, t_xtest], [t_ytrain, t_yval, t_ytest], stats

def MyDataset(x, y):
    out = [x, y]
    return out
