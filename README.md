# Weather forecast with LSTM autoencoders

### Description

This is the python code accompanying the paper 

> Ponnoprat, D. (2021). Short-term daily precipitation forecasting with seasonally-integrated autoencoder. Applied Soft Computing, 102, 107083. https://doi.org/10.1016/j.asoc.2021.107083
	
The code comes with the shell scripts to train and evaluate the autoencoder models (S2S1, S2S2 and SSAE) on the datasets used in the paper.

### Requirements
Following libraries are required to run the scripts:

* Python 3.6 or higher
* Tensorflow 2

### Usage

#### Model

To build a Seasonally-integrated Autoencoder (SSAE) model, place `models.py` in working directory and write the following script:

	from tensorflow.keras.layers import Input
	from models import SSAE

	input_node = Input(shape=(window_size, num_features))
	output = SSAE(short_history,			#Look-back window of short component 
			horizon,			#Forecast horizon.
 			seasonal_features,		#Indices of the variables to be used for seasonal component. 
 			pool_size,			#Window size of average pooling.
 			strides,			#Step size of moving window in average pooling.
 			hidden,				#Number of hidden units in the short component. 
			seasonal_hidden			#Number of hidden units in the seasonal components.
			)(input_node)

	model = Model(inputs=input_node,outputs=output)
	
The input must have shape `(num_rows, window_size, num_features)` and the output must have shape `(num_rows, window_size)`.
	
#### Experiments

To train SSAE using the provided data, simply run the shell script:

	./Providence.sh

The weights of the trained model will be saved in the `model` folder, and the forecast values will be saved in `path/to/data_XXXXXX-xxxxxx.csv`. 

#### Changing arguments

You can modify the arguments in the script files. For example, `--model` and `--horizon` let you specify the model and the forecast horizon, respectively. The descriptions of all available options can be accessed via the command:

	python3 main.py -h

#### Running the script in different modes

The default argument `training` mode let you train and evaluate the model. In addition, there are `prediction` mode that allows you to forecast from past data and `evaluation` mode that allows you to load the pretrained weights and evaluate the model on the test set. We have prepared the scripts for these modes, namely `Providence_predict.sh` and `Providence_eval.sh`, as well as the pretrained weights for all three models in the `model` folder. To use these two modes, you need to specify the location of the pretrained weights using `--load` option. For example, the weights of SSAE that makes forecast over the next three days on Providence dataset are stored in `pvd_ssae_3.h5`

	--load model/pvd_ssae_3.h5

You also need to specify the test data.

	--test_data Data/data_name.csv

Note that these scripts use the training data as the test data.

To run in these modes, you need to change some parameters:
* For all models, the parameter of `--horizon` must match the forecast horizon of the pretrained weights.
* For S2S-1 model, you have to change the parameter of `--hidden` to 100.

#### Changing the loss function

You can also change the loss function from the default mean-squared error to the quantile loss by changing the parameter in `loss`. For example,

	--loss q 

to train with the q-th quantile loss (`0<q<1`). 

#### Changing seasonality integration method

In the paper, the seasonality is integrated into the model via multiplication. For comparison purpose, two additional methods are also provided: use `--integrate_method add` or `--integrate_method linear` to integrate via addition and linear combination, respectively.


The scripts are tested with Python version 3.6, Keras version 2.2.4 and Tensorflow version 1.12.0.


