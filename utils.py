import numpy as np
import time


def _transform_indices(indices, wind_indices):
    new_indices = [0]*len(indices)
    for i,ind in enumerate(indices):
        new_indices[i] = ind-sum(i<ind for i in wind_indices)
    return new_indices

def _transform_wind(data, indices):
    
    for i in indices:
        sin_wind = np.sin(np.deg2rad(data[:,i]))
        cos_wind = np.cos(np.deg2rad(data[:,i]))
        data = np.c_[data,sin_wind,cos_wind]
    for i in indices:
        data = np.delete(data, i, 1)
    return data

    

def _train_test_split(data, num_test_dates, history_len):
    
    train = data[:-num_test_dates]
    test = data[-num_test_dates-history_len:]
    return train, test

def _preprocess(data, history_len, horizon):
    
    k = history_len
    l = horizon
    
    input_data = np.zeros((data.shape[0]-k-l+1,k,data.shape[1]))
    output_data = np.zeros((data.shape[0]-k-l+1,l,data.shape[1]))
    
    for m in range(data.shape[1]):
        for i in range(k):
            input_data[:,i,m] = data[i:-1*(k+l-i)+1,m]
        for i in range(l):
            output_data[:,i,m] = data[i+k:data.shape[0]-1*(l-i)+1,m]
         
    return input_data, output_data

def rmse(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    
    mse = np.average((y_true - y_pred) ** 2)
    
    return np.sqrt(mse)

def corr(y_true, y_pred):

    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)


    ymean_true = y_true.mean()
    ymean_pred = y_pred.mean()

    ym_true = y_true - ymean_true
    ym_pred = y_pred - ymean_pred

    normy_true = np.linalg.norm(ym_true)
    normy_pred = np.linalg.norm(ym_pred)

    r = np.dot(ym_true/normy_true, ym_pred/normy_pred)
    
    return r