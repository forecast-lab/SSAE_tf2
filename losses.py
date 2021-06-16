import tensorflow.keras.backend as K

def q_loss(q):
    #Compute q-th quantile loss.
    
    def loss_func(y,f):
        e = (y-f)
        e_a = K.abs(e)
        return K.mean(K.maximum(q*e, (q-1)*e), axis=-1)
    
    return loss_func