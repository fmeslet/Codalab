#################
# IMPORTATION
#################


# Modelisation
import tensorflow as tf
import keras

# Numeric algebra
import numpy as np

# Keras import
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from keras.layers import Input
from keras.models import Model, Sequential
from keras.optimizers import Adam
from keras.layers import Activation
from keras.layers.convolutional_recurrent import ConvLSTM2D

# Others
import gc
import os


###################
# SET PARAMETERS
###################


MAIN_DIR = "/home/mp/mesletf/scratch/CERFACS/wave/"
RESULT = MAIN_DIR + "result2/"
DATA = MAIN_DIR + "data/"


###################
# LOAD DATA
###################


def load_data():
    """Load data.

    Returns:
        nd.array -- Data shape in
        (batch, timesteps, channel, 200, 200)
        in np.float32
    """

    # Note : I convert data in float32 and into shape
    # (batch, timestep, channel, 200, 200).
    # I split data into chunk to facilitate network
    # transfer and convert to float32 in order
    # to avoid RessourceExaustError
    X = np.concatenate((np.load(DATA + 'X_1.npy'),
                        np.load(DATA + 'X_3.npy'),
                        np.load(DATA + 'X_4.npy'),
                        np.load(DATA + 'X_5.npy'),
                        np.load(DATA + 'X_2.npy')), axis=0)
    return X


def scale(X):
    """Normalize data.

    Arguments:
        X {nd.array} -- Matrices which need to be shaped.

    Returns:
        tuple -- Matrices, standard deviation and mean
    """
    std = X.std()
    mean = X.mean()
    X = (X - mean) / (std)
    return X, std, mean


###################
# DATA PREPARATION
###################
# Note : a better way would be to use an iterator for learning by batch


def train_val_test_split(X, y, val_size=10, test_size=10,
                         input_size=1, output_size=0):
    """Split the data into train/validation/test set.

    Arguments:
        X {nd.array} -- Input matrices
        y {nd.array} -- Target matrices

    Keyword Arguments:
        val_size {int} -- Percentage of validation data (default: {10})
        test_size {int} -- Percentage of test data (default: {10})
        input_size {int} -- Input size (default: {1})
        output_size {int} -- Output size (default: {0})

    Returns:
        tuple -- X_train, y_train, X_val, y_val
    """
    # Convert percentage into value
    val_size = int((val_size*0.01)*X.shape[0])
    test_size = int((test_size*0.01)*X.shape[0])

    limit_train = X.shape[0] - val_size - test_size - output_size + 1
    limit_val = limit_train + val_size
    limit_test = limit_val + test_size

    # TRAINING SET
    X_train = []
    y_train = []
    for i in range(input_size, limit_train):
        X_train.append(X[i-input_size:i, :])
        y_train.append(y[i:i+output_size, :])
    X_train, y_train = np.array(X_train), np.array(y_train)

    # VALIDATION SET
    X_val = []
    y_val = []
    for i in range(limit_train, limit_val):
        X_val.append(X[i-input_size:i, :])
        y_val.append(y[i:i+output_size, :])
    X_val, y_val = np.array(X_val), np.array(y_val)

    return X_train, y_train, X_val, y_val


def reshape(data, output_size=4):
    """Get the different matrices reshaped.

    Arguments:
        data {nd.array} -- data we want to extract training,
        validation and test set reshaped

    Keyword Arguments:
        output {int} -- output size (default: {4})
    """

    val_size = int((20*0.01)*data.shape[0])
    limit_train = data.shape[0] - val_size
    limit_val = data.shape[0]

    X_train = np.empty((0, 4, 1, 200, 200), np.float32)
    y_train = np.empty((0, output_size, 1, 200, 200), np.float32)

    X_val = np.empty((0, 4, 1, 200, 200), np.float32)
    y_val = np.empty((0, output_size, 1, 200, 200), np.float32)

    for i in range(limit_train):
        X, y = data[0], data[0]
        X_train_new, y_train_new, _, _ = train_val_test_split(
            X, y, val_size=0, test_size=0,
            input_size=4, output_size=output_size)
        X_train = np.concatenate((X_train, X_train_new), axis=0)
        y_train = np.concatenate((y_train, y_train_new), axis=0)

        # Reduce memory
        data = np.delete(data, 0, 0)
        del X_train_new, y_train_new
        gc.collect()

    # VAL_SIZE
    for i in range(limit_train, limit_val):
        X, y = data[0], data[0]
        X_val_new, y_val_new, _, _ = train_val_test_split(
            X, y, val_size=0, test_size=0,
            input_size=4, output_size=output_size)
        X_val = np.concatenate((X_val, X_val_new), axis=0)
        y_val = np.concatenate((y_val, y_val_new), axis=0)

        # Reduce memory
        data = np.delete(data, 0, 0)
        del X_val_new, y_val_new
        gc.collect()

    return X_train, y_train, X_val, y_val


###################
# KERAS MODEL
###################


def build_model():
    """Get the Kera model.

    Returns:
        keras.Model -- Keras model
    """

    keras.backend.clear_session()

    n_units = 45

    # Note : you can't use a tab ConvLSTM2DCell
    # so you can't use for loop in order to define layers

    # Training ENCODER
    encoder_inputs = Input(shape=(None, 1, 200, 200))

    encoder_convlstm_1 = ConvLSTM2D(filters=n_units,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    data_format="channels_first",
                                    return_sequences=True,
                                    return_state=True)
    encoder_convlstm_1_outputs, state_h_1, state_c_1 = encoder_convlstm_1(
        encoder_inputs)

    # Tricks to avoid AssertionError
    encoder_states_1 = encoder_convlstm_1.get_initial_state(encoder_inputs)

    encoder_convlstm_2 = ConvLSTM2D(filters=n_units,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    data_format="channels_first",
                                    return_sequences=True,
                                    return_state=True)
    encoder_convlstm_2_outputs, state_h_2, state_c_2 = encoder_convlstm_2(
        encoder_inputs)

    # Tricks to avoid AssertionError
    encoder_states_2 = encoder_convlstm_2.get_initial_state(encoder_inputs)

    encoder_convlstm_3 = ConvLSTM2D(filters=1,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    data_format="channels_first",
                                    return_state=True)
    encoder_convlstm_3_outputs, state_h_3, state_c_3 = encoder_convlstm_3(
        encoder_inputs)

    # Tricks to avoid AssertionError
    encoder_states_3 = encoder_convlstm_3.get_initial_state(encoder_inputs)

    encoder_states = [state_h_1, state_c_1, state_h_2,
                      state_c_2, state_h_3, state_c_3]

    # Tricks to avoid shape error
    encoder_states_1[0] = tf.identity(encoder_states_1[0], name="test")
    encoder_states_1[1] = tf.identity(encoder_states_1[1], name="test")

    encoder_states_2[0] = tf.identity(encoder_states_2[0], name="test")
    encoder_states_2[1] = tf.identity(encoder_states_2[1], name="test")

    encoder_states_3[0] = tf.identity(encoder_states_3[0], name="test")
    encoder_states_3[1] = tf.identity(encoder_states_3[1], name="test")

    # Training DECODER
    decoder_inputs = Input(shape=(None, 1, 200, 200))

    decoder_convlstm_1 = ConvLSTM2D(filters=n_units,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    data_format="channels_first",
                                    return_state=True,
                                    return_sequences=True)
    decoder_convlstm_1_outputs, _, _ = decoder_convlstm_1(
        decoder_inputs, initial_state=encoder_states_1)

    decoder_convlstm_2 = ConvLSTM2D(filters=n_units,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    data_format="channels_first",
                                    return_state=True,
                                    return_sequences=True)
    decoder_convlstm_2_outputs, _, _ = decoder_convlstm_2(
        decoder_convlstm_1_outputs, initial_state=encoder_states_2)

    decoder_convlstm_3 = ConvLSTM2D(filters=1,
                                    kernel_size=(3, 3),
                                    padding='same',
                                    data_format="channels_first",
                                    return_state=True,
                                    return_sequences=True)
    decoder_outputs, _, _ = decoder_convlstm_3(
        decoder_convlstm_2_outputs, initial_state=encoder_states_3)

    model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

    # Define INFERENCE ENCODER
    encoder_model = Model(encoder_inputs, encoder_states)

    # Define INFERENCE DECODER
    decoder_state_input_h_1 = Input(shape=(n_units, 200, 200))
    decoder_state_input_c_1 = Input(shape=(n_units, 200, 200))

    decoder_state_input_h_2 = Input(shape=(n_units, 200, 200))
    decoder_state_input_c_2 = Input(shape=(n_units, 200, 200))

    decoder_state_input_h_3 = Input(shape=(1, 200, 200))
    decoder_state_input_c_3 = Input(shape=(1, 200, 200))

    decoder_states_inputs = [decoder_state_input_h_1, decoder_state_input_c_1,
                             decoder_state_input_h_2, decoder_state_input_c_2,
                             decoder_state_input_h_3, decoder_state_input_c_3]

    # Tricks to avoid AssertionError
    decoder_state_h_1 = tf.identity(decoder_state_input_h_1,
                                    name="convolution")
    decoder_state_c_1 = tf.identity(decoder_state_input_c_1,
                                    name="convolution")

    # Tricks to avoid AssertionError
    decoder_state_h_2 = tf.identity(decoder_state_input_h_2,
                                    name="convolution")
    decoder_state_c_2 = tf.identity(decoder_state_input_c_2,
                                    name="convolution")

    # Tricks to avoid AssertionError
    decoder_state_h_3 = tf.identity(decoder_state_input_h_3,
                                    name="convolution")
    decoder_state_c_3 = tf.identity(decoder_state_input_c_3,
                                    name="convolution")

    decoder_states = [decoder_state_h_1, decoder_state_c_1,
                      decoder_state_h_2, decoder_state_c_2,
                      decoder_state_h_3, decoder_state_c_3]

    decoder_convlstm_1_outputs, state_h_1, state_c_1 = decoder_convlstm_1(
        decoder_inputs, initial_state=decoder_states[0:2])
    decoder_convlstm_2_outputs, state_h_2, state_c_2 = decoder_convlstm_2(
        decoder_convlstm_1_outputs, initial_state=decoder_states[2:4])
    decoder_outputs, state_h_3, state_c_3 = decoder_convlstm_3(
        decoder_convlstm_2_outputs, initial_state=decoder_states[4:6])

    decoder_states = [state_h_1, state_c_1, state_h_2,
                      state_c_2, state_h_3, state_c_3]

    decoder_model = Model([decoder_inputs] + decoder_states_inputs,
                          [decoder_outputs] + decoder_states)

    # return all models
    return model, encoder_model, decoder_model


###################
# LEARNING
###################


def train(X_train, X_val, y_train, y_val):
    """Create the Keras models and train it.

    Arguments:
        X_train {nd.array} -- Input training matrice
        X_val {nd.array} -- Input validation matrice
        y_train {nd.array} -- Target training matrice
        y_val {nd.array} -- Target validation matrice

    Returns:
        Tuple -- history of learning, model trained,
        inference encoder, inference decoders
    """
    epochs = 7

    cbs = [ReduceLROnPlateau(monitor='loss', factor=0.5,
                             patience=1, min_lr=1e-6, verbose=0),
           EarlyStopping(monitor='val_loss', min_delta=1e-7,
                         patience=5, verbose=1, mode='min',
                         restore_best_weights=True)]

    model, infenc, infdec = build_model()

    model.compile(optimizer=keras.optimizers.Adam(1e-3),
                  loss='mean_squared_error', metrics=['mse'])

    model.summary()

    X_train_bis = np.pad(y_train, ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)),
                         mode='constant')[:, :-1]
    X_val_bis = np.pad(y_val, ((0, 0), (1, 0), (0, 0), (0, 0), (0, 0)),
                       mode='constant')[:, :-1]

    history = model.fit([X_train, X_train_bis], y_train,
                        validation_data=([X_val, X_val_bis], y_val),
                        epochs=epochs,
                        batch_size=8,
                        shuffle=True,
                        callbacks=cbs)

    del X_train_bis, X_val_bis
    gc.collect()

    return history, model, infenc, infdec


###################
# PREDICT
###################


def predict_sequence(infenc, infdec, source, n_steps):
    """Predict sequences using inference encoder and decoder.

    Arguments:
        infenc {keras.Model} -- Inference encoder mode
        infdec {keras.Model} -- Inference decoder model
        source {nd.array} -- data to predict
        n_steps {int} -- Number of steps to predict

    Returns:
        nd.array -- Predictions
    """
    # encode
    state = infenc.predict(source)
    # start of sequence input
    target_seq = np.zeros((1, 1, 1, 200, 200))
    # collect predictions
    output = np.empty((1, 0, 1, 200, 200), np.float16)

    for t in range(n_steps):
        # predict next char
        yhat, h_1, c_1, h_2, c_2, h_3, c_3 = infdec.predict(
            [target_seq] + state)
        # store prediction
        output = np.concatenate((output, yhat), axis=1)
        # update state
        state = [h_1, c_1, h_2, c_2, h_3, c_3]
        # update target sequence
        target_seq = yhat

    return output.reshape((1, n_steps, 1, 200, 200))


###################
# MAIN
###################


if __name__ == '__main__':
    # GET DATA
    gc.collect()
    data = load_data()
    data, std, mean = scale(data)

    # GET MATRICES
    X_train, y_train, X_val, y_val = reshape(data, output=10)
    # Reduce memory
    del data

    # GET MODEL
    history, model, infenc, infdec = train(X_train, X_val, y_train, y_val)

    # Reduce memory (can be remove)
    del X_train, y_train, X_val, y_val
    gc.collect()

    gc.collect()
    data = load_data()
    data, std, mean = scale(data)

    # CHECK PERFORMANCE
    result = (predict_sequence(infenc, infdec,
                               data[0:1, 0:4], n_steps=100) * std) + mean
    np.save(RESULT+"y_pred_decoder", result)
    np.save(RESULT+"y_true_decoder", data[0:1])

    # PREDICT
    acoustic_opposed = (
        np.load(DATA+"acoustic_opposed_inputs.npy") - mean) / std
    acoustic = (np.load(DATA+"acoustic_inputs.npy") - mean) / std

    acoustic_pred_decoder = np.empty((0, 100, 1, 200, 200), np.float32)

    # PREDICT ACCOUSTIC OPPOSED
    acoustic_opposed_pred_decoder = predict_sequence(
        infenc, infdec, acoustic_opposed[0:1, 0:4], n_steps=100)
    np.save(RESULT+"acoustic_opposed_pred_decoder",
            acoustic_opposed_pred_decoder)

    # PREDICT ACCOUSTIC
    for i in range(acoustic.shape[0]):
        acoustic_pred_decoder_new = predict_sequence(
            infenc, infdec, acoustic[i:i+1, 0:4], n_steps=100)
        acoustic_pred_decoder = np.concatenate(
            (acoustic_pred_decoder, acoustic_pred_decoder_new), axis=0)

    acoustic_pred_decoder = (acoustic_pred_decoder * std) + mean
    np.save(RESULT+"acoustic_pred_decoder", acoustic_pred_decoder)
