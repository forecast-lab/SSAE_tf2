# Weather forecast with LSTM autoencoders

### Description

This is the python code accompanying the paper *Short-term Daily Precipitation Forecasting with Seasonally-Integrated Long Short-Term Memory Autoencoder* for reproducibility. The code comes with the shell scripts to train and evaluate the autoencoder models (S2S1, S2S2 and SSAE) on the datasets used in the paper.

### Requirements
Following libraries are required to run the scripts:

* Keras 2.2 or higher
* Tensorflow 1

Additionally, we use RAdam for stable learning-rate schedules. RAdam can be installed via pip installer.

	pip install keras-rectified-adam

### Usage


#### Basic

The code provides To train the SSAE model using the settings from the papers, simply run the shell script:

	./Providence.sh

The weights of the trained model will be saved in the `model` folder with the name specified in the `--save` option. 

#### Running the script in different modes

The default argument `--mode training` let you train and evaluate the model. In addition, there are `--mode prediction` to make a forecast from a single stream of past data and `--mode evaluation` to load the pretrained weights and evaluate the model on the test set. To use these two modes, you need to specify the location of the pretrained weights

	--load model/pretrained_weights.h5

and the test data

	--test_data Data/data_name.csv

An example of a shell script that runs in prediction and evaluation mode is provided in `Providence_predict.sh` and `Providence_eval.sh`, respectively. To run in these modes, you need to change some parameters:
* For all models, the parameter of `--horizon` must match the forecast horizon of the loaded weights.
* For S2S-1 model, you have to change the parameter of `--hidden` to 100.

#### Saving the predictions

The `training` and `evaluation` mode also allows you to save the predictions by adding the following option to the script:

	--save_predictions filename.csv

#### Changing the loss function

You can also change the loss function from the default mean-squared error to the quantile loss by changing the parameter in `loss`. For example,

	--loss q 

to train with the q-th quantile loss (`0<q<1`). 

#### Changing seasonal integration method

In the paper, the seasonality is integrated into the model via multiplication. For comparison purpose, two additional methods are also provided: use `--integrate_method add` or `--integrate_method linear` to integrate via addition and linear combination, respectively.

Lastly, the descriptions of all options can be accessed via the command:

	./Providence.sh -h


The scripts are tested with Python version 3.6, Keras version 2.2.4 and Tensorflow version 1.12.0


