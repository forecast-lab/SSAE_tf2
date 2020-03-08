from keras.models import Model
from keras.layers import Layer, Input, Lambda, Reshape, Concatenate, LSTM
from keras.layers import Dense, AveragePooling1D, Add, Multiply, RepeatVector


class Autoencoder(Model):
    def __init__(self, horizon, 
                 hidden,
                 decoder_activation = 'tanh',
                 encoder_activation = 'tanh',
                 pass_states = False,
                 share_decoder_weights = False,
                 dense_after_encoder = False,
                 initializer = 'glorot_normal',
                 name = 'autoencoder',
                 **kwargs):
        self.horizon = horizon
        self.hidden = hidden
        self.enc_act = encoder_activation
        self.dec_act = decoder_activation
        self.dense_after_encoder = dense_after_encoder
        self.pass_states = pass_states
        self.share_decoder_weights = share_decoder_weights
        self.init = initializer
        super(Autoencoder, self).__init__(name=name, **kwargs)
        self.encoder = LSTM(self.hidden,
                            return_sequences=False,
                            return_state=True,
                            kernel_initializer=self.init,
                            activation=self.enc_act)
        if self.dense_after_encoder:
            self.hidden_dense = Dense(self.hidden,
                                      kernel_initializer=self.init,
                                      activation='relu')

        if self.share_decoder_weights:
            self.repeatvector = RepeatVector(self.horizon)
            self.decoder = LSTM(self.hidden,
                                return_sequences=True,
                                return_state=False,
                                kernel_initializer=self.init,
                                activation=self.dec_act)
        else:    
            self.reshape_1 = Reshape((1,self.hidden))
            self.decoder_layers = [0]*self.horizon
            for h in range(self.horizon):   
                self.decoder_layer = LSTM(self.hidden,
                                            return_sequences=True,
                                            return_state=True,
                                            kernel_initializer=self.init,
                                            activation=self.dec_act)
                self.decoder_layers[h] = self.decoder_layer
        self.concatenate = Concatenate(axis=1)

    def call(self, inputs):
        h_T, state_h, state_c = self.encoder(inputs)
        encoder_states = [state_h, state_c]
        
        if self.dense_after_encoder:
            h_T = self.hidden_dense(h_T)
        
        if self.pass_states:
            decoder_states = encoder_states
        else:
            decoder_states = None
            
        if self.share_decoder_weights:
            y_H = self.repeatvector(h_T)
            y = self.decoder(y_H, decoder_states)
        else:
            h_T = self.reshape_1(h_T)
            y_H = [0]*self.horizon
            for h in range(self.horizon):
                y_h, state_h, state_c = self.decoder_layers[h](h_T, decoder_states)
                y_H[h] = y_h
                decoder_states = [state_h, state_c]
            if self.horizon == 1:
                y = y_H[0]
            else:
                y = self.concatenate(y_H)
        return y
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.horizon, self.hidden)

    
    
class S2S1(Model):
    """ S2S-1
    # Arguments
        horizon: int, >=1. Forecast horizon.
        hidden: int, >=1. Number of hidden units in the short component. 
        encoder_activation: string. Activation function in the encoder part of the short component.
        decoder_activation: string. Activation function in the decoder part of the short component.
    """

    def __init__(self, horizon, 
                 hidden,
                 decoder_activation = 'tanh',
                 encoder_activation = 'tanh',
                 initializer = 'glorot_normal',
                 name = 'S2S1',
                 **kwargs):
        
        self.horizon = horizon
        self.hidden = hidden
        self.enc_act = encoder_activation
        self.dec_act = decoder_activation
        self.init = initializer
        super(S2S1, self).__init__(name=name, **kwargs)
        
        self.autoencoder = Autoencoder(horizon = self.horizon,
                                       hidden = self.hidden,
                                       decoder_activation = self.dec_act,
                                       encoder_activation = self.enc_act,
                                       pass_states = False,
                                       share_decoder_weights = True,
                                       dense_after_encoder = True,
                                       name = 'autoencoder')        
        self.dense = Dense(1,kernel_initializer=self.init,use_bias=False)
        self.reshape = Reshape((self.horizon,))


    def call(self, inputs):
        y = self.autoencoder(inputs)
        output = self.dense(y)
        output = self.reshape(output)
        return output
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.horizon,)
    
    
class S2S2(Model):
    """ S2S-2
    # Arguments
        horizon: int, >=1. Forecast horizon.
        hidden: int, >=1. Number of hidden units in the short component. 
        encoder_activation: string. Activation function in the encoder part of the short component.
        decoder_activation: string. Activation function in the decoder part of the short component.
    """

    def __init__(self, horizon, 
                 hidden,
                 decoder_activation,
                 encoder_activation = 'tanh',
                 initializer = 'glorot_normal',
                 name = 'S2S2',
                 **kwargs):
        
        self.horizon = horizon
        self.hidden = hidden
        self.enc_act = encoder_activation
        self.dec_act = decoder_activation
        self.init = initializer
        super(S2S2, self).__init__(name=name, **kwargs)
        
        self.autoencoder = Autoencoder(horizon = self.horizon,
                                       hidden = self.hidden,
                                       decoder_activation = self.dec_act,
                                       encoder_activation = self.enc_act,
                                       pass_states = False,
                                       share_decoder_weights = False,
                                       dense_after_encoder = False,
                                       name = 'S2S2')        
        self.dense = Dense(1,kernel_initializer=self.init,use_bias=False)
        self.reshape = Reshape((self.horizon,))

    def call(self, inputs):
        y = self.autoencoder(inputs)
        output = self.dense(y)
        output = self.reshape(output)
        return output
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.horizon,)
    
class SSAE(Model):
    """ Seasonally-integrated autoencoder
    # Arguments
        short_history: int, >=1. Look-back window of short component
        horizon: int, >=1. Forecast horizon.
        seasonal_features: list. Indices of the variables to be used for seasonal component. 
        pool_size: int, >=1. Window size of average pooling.
        strides: int, >=1. Step size of moving window in average pooling.
        hidden: int, >=1. Number of hidden units in the short component. 
        seasonal_hidden: int, >=1. Number of hidden units in the seasonal components.
        encoder_activation: string. Activation function in the encoder part of the short component.
        seasonal_encoder_activation: string. Activation function in the encoder part of the seasonal component.
        decoder_activation: string. Activation function in the decoder part of the short component.
        seasonal_decoder_activation: string. Activation function in the decoder part of the seasonal component.
        integrate_method: string. Type of decomposition into seasonal and short components: 'add' for Additive, 'multiply' for multiplicative and 'linear' for a linear combination. 
    """

    def __init__(self, short_history,
                 horizon,
                 seasonal_features,
                 pool_size,
                 strides,
                 hidden,
                 seasonal_hidden,
                 encoder_activation = 'tanh',
                 seasonal_encoder_activation = 'tanh',
                 decoder_activation = 'tanh',
                 seasonal_decoder_activation = 'tanh',
                 integrate_method = 'multiply',
                 initializer = 'glorot_normal',
                 name = 'SSAE',
                 **kwargs):
        self.short_history = short_history
        self.horizon = horizon
        self.seasonal_features = seasonal_features
        self.pool_size = pool_size,
        self.strides = strides,
        self.hidden = hidden
        self.enc_act = encoder_activation
        self.dec_act = decoder_activation
        self.s_enc_act = seasonal_encoder_activation
        self.s_dec_act = seasonal_decoder_activation
        self.integrate_method = integrate_method
        self.init = initializer
        super(SSAE, self).__init__(name=name, **kwargs)
        
        self.short_lambda = Lambda(lambda a: a[:,-1*self.short_history:,:])
        self.autoencoder = Autoencoder(horizon = self.horizon,
                                       hidden = self.hidden,
                                       decoder_activation = self.dec_act,
                                       encoder_activation = self.enc_act,
                                       pass_states = False,
                                       share_decoder_weights = False,
                                       dense_after_encoder = False,
                                       name = 'autoencoder')        
        self.dense = Dense(1,kernel_initializer=self.init,use_bias=False)
        
        self.feature_selector = [0]*len(self.seasonal_features)
        
        for i,f in enumerate(self.seasonal_features):
            self.lambda_feature = Lambda(lambda a: a[:,:,f:f+1])
            self.feature_selector[i] = self.lambda_feature
            
        self.average_pooling = AveragePooling1D(pool_size = self.pool_size, 
                                                strides = self.strides)
        if len(seasonal_features) > 1:
            self.concatenate = Concatenate(axis=2)
        
        self.seasonal_autoencoder = Autoencoder(horizon = self.horizon,
                                                hidden = self.hidden,
                                                decoder_activation = self.s_dec_act,
                                                encoder_activation = self.s_enc_act, 
                                                pass_states = True,
                                                share_decoder_weights = False,
                                                dense_after_encoder = False,
                                                name = 'autoencoder') 
        self.seasonal_dense = Dense(1,kernel_initializer=self.init,use_bias=False)
        if self.integrate_method == 'multiply':
            self.multiply = Multiply()
        elif self.integrate_method == 'add':
            self.add = Add()
        elif self.integrate_method == 'linear':
            self.concatecate_y = Concatenate()
            self.dense_y = Dense(1,kernel_initializer=self.init,use_bias=False)
        else:
            raise ValueError("integrate_method must be one of 'multiply', 'add' or 'linear'.")
            
        self.reshape = Reshape((self.horizon,))
        

    def call(self, inputs):
        inputs_short = self.short_lambda(inputs)
        y = self.autoencoder(inputs_short)
        y = self.dense(y)
        
        if len(self.seasonal_features) == 1:
            z = self.feature_selector[0](inputs)
            z = self.average_pooling(z)
        else:
            pooled_features = [0]*len(self.seasonal_features)
            for i in range(len(self.feature_selector)):
                z = self.feature_selector[i](inputs)
                z = self.average_pooling(z)
                pooled_features[i] = z
            z = self.concatenate(pooled_features)
        y_s = self.seasonal_autoencoder(z)
        y_s = self.seasonal_dense(y_s)
        
        if self.integrate_method == 'multiply':
            output = self.multiply([y,y_s])
        elif self.integrate_method == 'add':
            output = self.add([y,y_s])
        elif self.integrate_method == 'linear':
            y_ys = self.concatecate_y([y,y_s])
            output = self.dense_y(y_ys)
        
        output = self.reshape(output)
        
        return output
    
    def compute_output_shape(self,input_shape):
        return (input_shape[0], self.horizon,)
