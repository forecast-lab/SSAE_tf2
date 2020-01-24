import argparse

parser = argparse.ArgumentParser(description='Precipitation Forecasting with LSTM Autoencoders')
parser.add_argument('--data', type=str, required=True,
                    help='location of the data file')
parser.add_argument('--test_data', type=str,
                    help='location of the data file to make predictions in the prediction mode')
parser.add_argument('--model', type=str, default='SSAE',
                    help='Name of the model: options are "SSAE", "S2S1" or "S2S2"')
parser.add_argument('--mode', type=str, default='training',
                    help='Choose "training" to train the model and "prediction" to use the pretrained model for predictions.')
parser.add_argument('--season_window', type=int, default=70,
                    help='The window size of the seasonal component')
parser.add_argument('--window', type=int, default=2,
                    help='The window size of the short component')
parser.add_argument('--horizon', type=int, default=3,
                    help='The forecast horizon')
parser.add_argument('--test_days', type=int, default=365*3,
                    help='The number of consecutive days in the test set')
parser.add_argument('--prep_index', type=int, default=6,
                    help='The (pre-transformed) column index of the precipitation')
parser.add_argument('--wind_indices', nargs='+', type=int, default=[],
                    help='The (pre-transformed) column indices of the wind directions')
parser.add_argument('--winsor_indices', nargs='+', type=int, default=[],
                    help='The (pre-transformed) indices of the features to be winsorized')
parser.add_argument('--season_indices', nargs='+', type=int, default=[],
                    help='The (pre-transformed) column indices of the input features of the seasonal component')
parser.add_argument('--pool_size', type=int, default=41,
                    help='The window size of the average pooling layer')
parser.add_argument('--strides', type=int, default=14,
                    help='The stride of the average pooling layer')
parser.add_argument('--hidden', type=int, default=120,
                    help='The dimensionality of the hidden states of the short encoder')
parser.add_argument('--season_hidden', type=int, default=100,
                    help='The dimensionality of the hidden states of the seasonal encoder')
parser.add_argument('--enc_act', type=str, default='tanh',
                   help='The activation function of the short encoder')
parser.add_argument('--dec_act', type=str, default='relu',
                   help='The activation function of the short decoder')
parser.add_argument('--season_enc_act', type=str, default='tanh',
                   help='The activation function of the seasonal encoder')
parser.add_argument('--season_dec_act', type=str, default='tanh',
                   help='The activation function of the seasonal decoder')
parser.add_argument('--integrate_method', type=str, default='multiply',
                   help='The way to combine the short and seasonal component: options are "multiply", "add" or "linear"')
parser.add_argument('--epochs', type=int, default=75,
                    help='The number of training epochs')
parser.add_argument('--batch_size', type=int, default=256, metavar='N',
                    help='batch size')
parser.add_argument('--loss', type=float, default=0,
                    help='loss function: 0 : mean-squared error, q : q-th quantile regression')

parser.add_argument('--lr', type=float, default=0.0008,
                   help='The learning rate')
parser.add_argument('--decay_rate', type=float, default=0.955,
                   help='The exponential decay rate of the learning rate schedule')
parser.add_argument('--decay_steps', type=int, default=20,
                   help='The number of steps to reach the specified decay rate')
parser.add_argument('--save_predictions', action='store_true', default=False,
                   help='Save the predictions (starting from day (season_)window+1) to a *.csv file')
parser.add_argument('--save', type=str,  default='model/pvd.h5',
                    help='path to save the final model (only in training mode')
parser.add_argument('--load', type=str,  default='model/pvd.h5',
                    help='path to load the model (only in prediction and evaluation mode')


args = parser.parse_args()


import numpy as np
import time

from keras.models import Model
from keras.layers import Input
from keras_radam import RAdam

from models import S2S1, S2S2, SSAE
from losses import q_loss
from callbacks import LearningRateExponentialDecay
from utils import _transform_indices, _transform_wind
from utils import _train_test_split, _preprocess
from utils import rmse, corr


def create_model(args, feature_num):
    if args.model == 'SSAE':
        
        #Change the index of seasonal features to match the transformed data
        wind_index = feature_num-2*len(args.wind_indices)
        season_indices = []
        for i in args.season_indices:
            if i in args.wind_indices:
                season_indices.extend([wind_index , wind_index+1])
                wind_index += 2
            else:
                season_indices.append(i)
       
        input_node = Input(shape=(args.season_window, feature_num))
        output = SSAE(short_history=args.window,
                         horizon=args.horizon,
                         seasonal_features= season_indices,
                         pool_size = args.pool_size,
                         strides = args.strides,
                         hidden = args.hidden,
                         seasonal_hidden = args.season_hidden,
                         encoder_activation = args.enc_act,
                         decoder_activation = args.dec_act)(input_node)
    elif args.model == 'S2S1':
        input_node = Input(shape=(args.window, feature_num))
        output = S2S1(horizon=args.horizon,
                         hidden = args.hidden)(input_node)
    elif args.model == 'S2S2':
        input_node = Input(shape=(args.window, feature_num))
        output = S2S2(horizon=args.horizon,
                         hidden = args.hidden,
                         encoder_activation = args.enc_act,
                         decoder_activation = args.dec_act)(input_node)
    else:
        raise ValueError("The model must be one of 'SSAE', 'S2S1' or 'S2S2'.")

    model = Model(inputs=input_node,outputs=output)
    return model

def evaluate(model, X_test, y_test):
    y_pred = model.predict(X_test)
    test_rmse = rmse(y_pred[:,-1],y_test[:,-1]) 
    test_corr = corr(y_pred[:,-1],y_test[:,-1])
    return test_rmse, test_corr, y_pred
    




#Load the data
data = np.genfromtxt(args.data, delimiter=',',skip_header=1)

#Transform wind directions
data = _transform_wind(data, args.wind_indices)

if args.model == 'SSAE':
    window = args.season_window
else:
    window = args.window


train,test = _train_test_split(data, args.test_days, window)


#Change indices to match with the transformed data
prep_index = args.prep_index-sum(i<args.prep_index for i in args.wind_indices)
winsor_indices = _transform_indices(args.winsor_indices, args.wind_indices)

#Winsorization
uppers = dict()
for i in winsor_indices:
    upper = np.percentile(train[:,i],99.5)
    data[:,i] = np.clip(data[:,i],a_min = data[:,i].min(), a_max=upper)
    uppers[i] = upper 
uplim = data[:,prep_index].max()

#Normalization
train_min = train.min(axis=0)
train_max = train.max(axis=0)
data = (data-train_min)/(train_max-train_min)


#Moving window transformation
train,test = _train_test_split(data, args.test_days, window)
train_in, train_out = _preprocess(train, window, args.horizon)
y_train_out = train_out[:,:,prep_index]
test_in, test_out = _preprocess(test, window, args.horizon)
y_test_out = test_out[:,:,prep_index]

feature_num = train_in.shape[2]
model = create_model(args, feature_num)

lrd = LearningRateExponentialDecay(init_learning_rate=args.lr, 
                                   decay_rate=args.decay_rate, 
                                   decay_steps=args.decay_steps)
ra_adam = RAdam(lr=args.lr, beta_1=0.9, beta_2=0.99, epsilon=None, 
                          decay=0.0, amsgrad=False, warmup_proportion=0.1)

if args.loss == 0:
    loss_func = 'mse'
elif args.loss > 0 and args.loss <= 1:
    loss_func = q_loss(args.loss)
else:
    raise ValueError("The value of loss must be between 0 and 1.")
model.compile(loss=loss_func, optimizer=ra_adam)


if args.mode == 'training':
    # At any point you can hit Ctrl + C to break out of training early.
    try:
#         print('Begin training (press Ctrl + C to interrupt)...');
        model.fit(train_in,y_train_out, 
              epochs=args.epochs, 
              batch_size=args.batch_size, 
              callbacks=[lrd], verbose = 0)
        model.save_weights(args.save)
#         print('model saved as',str(args.save),'. Evaluating the model on the test set...')
        test_rmse, test_corr, y_pred = evaluate(model, test_in, y_test_out)
        print (uplim*test_rmse, test_corr)
        if args.save_predictions:
            timestr = time.strftime("%Y%m%d-%H%M%S")
            np.savetxt(args.data[:-4]+timestr+'.csv', uplim*y_pred, delimiter = ',')
            

    except KeyboardInterrupt:
        print('-' * 89)
        print('Exiting from training early')

elif args.mode == 'prediction':
    
    if args.test_data == None:
        raise ValueError("Please provide the path to your test data in the --test_data argument.")
    
    #Load the test data
    test_data = np.genfromtxt(args.test_data, delimiter=',',skip_header=1)

    #Transform wind directions
    test_data = _transform_wind(test_data, args.wind_indices)
    for i in uppers:
        test_data[:,i] = np.clip(data[:,i],a_min = test_data[:,i].min(), a_max=uppers[i])
    
    test_data = (test_data - train_min)/(train_max - train_min)
    
    test_data = np.array([test_data[-window:,:]])
    model.load_weights(args.load)
    print('Making predictions...')
    y_pred = model.predict(test_data)
    print(uplim*y_pred[0])


elif args.mode == 'evaluation':
    # Load the saved model.
    model.load_weights(args.load)
    print('Evaluating the model on the test set...')
    test_rmse, test_corr, y_pred = evaluate(model, test_in, y_test_out)
    print ("test rmse {:5.4f} | test corr {:5.4f}".format(uplim*test_rmse, test_corr))
    if args.save_predictions:
        timestr = time.strftime("%Y%m%d-%H%M%S")
        np.savetxt(args.data[:-4]+timestr+'.csv', uplim*y_pred, delimiter = ',')
        
else:
    raise ValueError("The mode must be one of 'training', 'prediction' or 'evaluation'.")
    