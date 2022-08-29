import keras.backend as bk

def dropout(input):
    training = bk.learning_phase()
    if training is 1 or training is True:
        input *= bk.cast(bk.random_uniform(bk.shape(input), minval=0, maxval=2, dtype='int32'), dtype='float32')
        input /= 0.5
    return input