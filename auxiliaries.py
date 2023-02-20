import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from keras import layers
import tensorflow_probability as tfp
import matplotlib.pyplot as plt
import types

# Set the seed for replicable results
seed = 0
np.random.seed(seed)
tf.random.set_seed(seed)

def func_lags(data,rhs_names,lags):

    # Create an storage Data Frame
    data_rhs = data[rhs_names].copy()

    # Which variables will go into your Xtrain/Xtest?
    rhs_colnames_store = []

    for ii in lags:
        
        # Create the new Column-Names
        rhs_colnames = ["L" + str(ii) + "_" + cc for cc in rhs_names]

        # Collect the variable names
        rhs_colnames_store.extend(rhs_colnames)
        
        # Lag the entire RHS
        data_rhs[rhs_colnames] = data[rhs_names].shift(ii)

    return data_rhs, rhs_colnames_store



def func_scale(data,scaler):
    
    if scaler == 0:
        # ---------------------------- Scale: [0;1]
        # Mean & SD of 'data'
        data_max = np.array(data.max()).reshape((1,data.shape[1]))
        data_min = np.array(data.min()).reshape((1,data.shape[1]))
        # ---- Normalize 'data'
        data_sc = (data - data_min) / (data_max - data_min)

        # ---------- Define numerator and denominator of the scaler
        my_scale = {'numerator': data_min, 'denominator': data_max - data_min} 

    elif scaler == 1:
        # ---------------------------- Scale: mean = 0; sd = 1

        # Mean & SD of 'data'
        data_mean = np.array(data.mean()).reshape((1,data.shape[1]))
        data_sd = np.array(data.std()).reshape((1,data.shape[1]))
        # ---- Normalize 'data'
        data_sc = (data - data_mean) / data_sd

        # ---------- Define numerator and denominator of the scaler
        my_scale = {'numerator': data_mean, 'denominator': data_sd} 
    
    elif scaler == 3:
        # ---------------------------- Scale: mean = 0; sd = 1

        # Mean & SD of 'data'
        data_mean = 0
        data_sd = 1
        # ---- Normalize 'data'
        data_sc = (data - data_mean) / data_sd

        # ---------- Define numerator and denominator of the scaler
        my_scale = {'numerator': data_mean, 'denominator': data_sd} 


    return data_sc, my_scale



def func_scale_invert(data, my_scale):

    inv_data = data * my_scale['denominator'] + my_scale['numerator']

    return inv_data



def func_param_init(nn_hyparams,n_X):

    layers_n = nn_hyparams['number_of_layers']
    nodes_n = nn_hyparams['number_of_nodes']

    params = {}
    for ii in range(0,layers_n):

        if ii == 0:
            params['W' + str(ii)] = np.random.randn(nodes_n[ii],n_X)*0.01
        else:
            params['W' + str(ii)] = np.random.randn(nodes_n[ii],nodes_n[ii-1])*0.01

        params['b' + str(ii)] = np.zeros((nodes_n[ii],1))

    return params



def func_build_DataSet(data,NBER,X_var,Y_var,lags,p_train,p_dev,scaler):

    # ---- First: Some Housekeeping, e.g. delete columns with NAs
    # Are there any "intermediate" NA's?
    nans = data.columns[data.isna().any()]
    for ii in nans:
        while ii in X_var: X_var.remove(ii)  


    # ---- Create lags
    X_data, rhs_names = func_lags(data,X_var,lags)
    X_data['Date'] = data['Date']
    X_data.drop(X_var,axis=1,inplace=True)

    if Y_var == ['USREC']:
        Y_data = NBER[list(['Date'] + Y_var)]
    else:   
        Y_data = data[list(['Date'] + Y_var)]

    X_data = X_data.loc[max(lags):,:]
    X_data.reset_index(drop=True,inplace=True)
    Y_data = Y_data.loc[max(lags):,:]
    Y_data.reset_index(drop=True,inplace=True)

    TT = Y_data.shape[0]

    # ====================== Create: Training-, Dev-, and Training-Set ================================ #
    h_train = range(0,int(p_train*TT))
    h_dev = range(max(h_train)+1, int((p_train+p_dev)*TT))
    h_test = range(max(h_dev)+1,TT)

    if Y_var != ['USREC']:
        data_train = pd.DataFrame(np.concatenate((Y_data.loc[h_train,Y_var],X_data.loc[h_train,rhs_names]), axis=1),
                                    columns=Y_var + rhs_names)

        data_dev = pd.DataFrame(np.concatenate((Y_data.loc[h_dev,Y_var],X_data.loc[h_dev,rhs_names]), axis=1),
                                    columns=Y_var + rhs_names)

        data_test = pd.DataFrame(np.concatenate((Y_data.loc[h_test,Y_var],X_data.loc[h_test,rhs_names]), axis=1),
                                    columns=Y_var + rhs_names)


    # ================================ Scale your Data ================================ #
    if Y_var == ['USREC']:
        data_train_sc, my_scale = func_scale(X_data.loc[h_train,rhs_names],scaler)
        data_dev_sc = (X_data.loc[h_dev,rhs_names] - my_scale['numerator']) / my_scale['denominator']
        data_test_sc = (X_data.loc[h_test,rhs_names] - my_scale['numerator']) / my_scale['denominator']

        data_train_sc = pd.DataFrame(np.concatenate((Y_data.loc[h_train,Y_var],data_train_sc), axis=1),
                                    columns=Y_var + rhs_names)
        data_dev_sc = pd.DataFrame(np.concatenate((Y_data.loc[h_dev,Y_var],data_dev_sc), axis=1),
                                    columns=Y_var + rhs_names)

        data_test_sc = pd.DataFrame(np.concatenate((Y_data.loc[h_test,Y_var],data_test_sc), axis=1),
                                    columns=Y_var + rhs_names)

    else:
        data_train_sc, my_scale = func_scale(data_train,scaler)
        data_dev_sc = (data_dev - my_scale['numerator']) / my_scale['denominator']
        data_test_sc = (data_test - my_scale['numerator']) / my_scale['denominator']
        my_scale['numerator'] = my_scale['numerator'][:,1:]
        my_scale['denominator'] = my_scale['denominator'][:,1:]

    # Training Set
    X_train = np.array(data_train_sc[rhs_names])
    Y_train = np.array(data_train_sc[Y_var])

    # Validation Set
    X_dev = np.array(data_dev_sc[rhs_names])
    Y_dev = np.array(data_dev_sc[Y_var])

    # Test Set
    X_test = np.array(data_test_sc[rhs_names])
    Y_test = np.array(data_test_sc[Y_var])


    # Collect everyting in an output dictionary
    dict_out = {'X_train': X_train,
                'Y_train': Y_train,
                'X_dev': X_dev,
                'Y_dev': Y_dev,
                'X_test': X_test,
                'Y_test': Y_test,
                'Y_data': Y_data,
                'my_scale': my_scale,
                'h_train': h_train,
                'h_dev': h_dev,
                'h_test': h_test,
                'rhs_vars': rhs_names}


    return dict_out

def func_buildNN_Sequential(X,hyparams):

    # ------- Unpack ------- #
    layer_n = hyparams['number_of_layers']
    layer_ty = hyparams['type_of_layers']
    nodes_n = hyparams['number_of_nodes']
    dropout_r = hyparams['dropout_rate']
    activ_f = hyparams['activation_function']
    ret_seq = hyparams['return_sequences']
    reg_val = hyparams['reg_val']
    penalty = hyparams['penalty']
    rec_drop = hyparams['recurrent_dropout'] 

    # ------- Define the Regularizer ------ #
    if penalty == 'L1':
        my_reg = keras.regularizers.l1(reg_val)
    elif penalty == 'L2':
        my_reg = keras.regularizers.l2(reg_val)                  


# ------- Build the Model ------- #

    model = keras.Sequential()

    # ---- Input
    if 'LSTM' in layer_ty:
        model.add(keras.Input(shape=(np.shape(X)[1],np.shape(X)[2])))
    else:
        model.add(keras.Input(shape=(np.shape(X)[1],)))
    
    # ---- Hidden Layers
    if layer_n > 1:
        if layer_ty[0] == 'LSTM':
            model.add(layers.LSTM(units=nodes_n[0], 
                            return_sequences = ret_seq[0], 
                            #dropout=dropout_r[0]),
                            recurrent_dropout=rec_drop[0],
                            activation=activ_f[0],
                            kernel_regularizer=my_reg))
            model.add(layers.Dropout(rate=dropout_r[0],seed=123))
        elif layer_ty[0] == 'Dense':
            model.add(layers.Dense(units=nodes_n[0], activation=activ_f[0],
                                    kernel_regularizer=my_reg))
            model.add(layers.Dropout(rate=dropout_r[0],seed=123))

    if layer_n > 2:    
        for ll in range(2,layer_n-1):
            if layer_ty[ll] == 'LSTM':
                model.add(layers.LSTM(units=nodes_n[ll],
                                return_sequences = ret_seq[ll], 
                                #dropout=dropout_r[ll]),
                                recurrent_dropout=rec_drop[ll],
                                activation=activ_f[ll],
                                kernel_regularizer=my_reg))
                model.add(layers.Dropout(rate=dropout_r[ll],seed=123))
            elif layer_ty[ll] == 'Dense':
                model.add(layers.Dense(units=nodes_n[ll], activation=activ_f[ll],
                                        kernel_regularizer=my_reg))
                model.add(layers.Dropout(rate=dropout_r[ll],seed=123))

    # ---- Output Layer
    if layer_ty[layer_n-1] == 'LSTM':
        model.add(layers.LSTM(units=nodes_n[layer_n-1],
                        return_sequences = ret_seq[layer_n-1], 
                        recurrent_dropout=rec_drop[layer_n-1],
                        #dropout=dropout_r[layer_n-1]),
                        activation=activ_f[layer_n-1],
                        kernel_regularizer=my_reg))
    elif layer_ty[layer_n-1] == 'Dense':
        model.add(layers.Dense(units=nodes_n[layer_n-1], activation=activ_f[layer_n-1]))


    # ---- Final Stage
    if hyparams['optimizer'] == 'Adam':
        optim = keras.optimizers.Adam(learning_rate=hyparams['learning_rate'],
                                      beta_1 = hyparams['Adam_b1'],
                                      beta_2 = hyparams['Adam_b2'],
                                     )
    elif hyparams['optimizer'] == 'RMSprop':
        optim = keras.optimizers.RMSprop(learning_rate=hyparams['learning_rate'],
                                         rho = hyparams['RMSprop_rho'],
                                         momentum = hyparams['momentum']
                                        )
    elif hyparams['optimizer'] == 'SGD':
        optim = keras.optimizers.SGD(learning_rate=hyparams['learning_rate'],
                                    momentum = hyparams['momentum'])

    if hyparams['loss_function'] == 'bin_foc_loss':
        model.compile(loss=functf_binary_focal_loss(alpha=hyparams['alpha'],gamma=hyparams['gamma']),
                      optimizer=optim,
                      metrics=hyparams['metrics'])
    else: 
        model.compile(loss=hyparams['loss_function'],optimizer=optim,
                      metrics=hyparams['metrics'])

    return model




def func_buildNN(X,hyparams):

    # ------- Unpack ------- #
    layer_n = hyparams['number_of_layers']
    layer_ty = hyparams['type_of_layers']
    nodes_n = hyparams['number_of_nodes']
    dropout_r = hyparams['dropout_rate']
    activ_f = hyparams['activation_function']
    ret_seq = hyparams['return_sequences']                   
    reg_val = hyparams['reg_val'] 
    penalty = hyparams['penalty']   
    rec_drop = hyparams['recurrent_dropout']

    # ------- Define the Regularizer ------ #
    if penalty == 'L1':
        my_reg = keras.regularizers.l1(reg_val)
    elif penalty == 'L2':
        my_reg = keras.regularizers.l2(reg_val)

    # ------- Build the Model ------- #

    # ---- Input
    if 'LSTM' in layer_ty:
        input = keras.Input(shape=(np.shape(X)[1],np.shape(X)[2]))
    else:
        input = keras.Input(shape=(np.shape(X)[1],))
        
    # ---- Hidden Layers
    if layer_n > 1:
        if layer_ty[0] == 'LSTM':
            hidden = layers.LSTM(units=nodes_n[0], input_shape=(input.shape[1],input.shape[2]), 
                                return_sequences = ret_seq[0], 
                                #dropout=dropout_r[0]),
                                recurrent_dropout=rec_drop[0],
                                activation=activ_f[0],
                                kernel_regularizer=my_reg)(input)
            hidden = layers.Dropout(rate=dropout_r[0],seed=123)(hidden)
        elif layer_ty[0] == 'Dense':
            hidden = layers.Dense(units=nodes_n[0], activation=activ_f[0],
                                    kernel_regularizer=my_reg)(input)
            hidden = layers.Dropout(rate=dropout_r[0],seed=123)(hidden)

    if layer_n > 2:
        for ll in range(1,layer_n-1):
            if layer_ty[ll] == 'LSTM':
                if len(hidden.shape) < 3:
                    hidden = layers.Flatten()(hidden)
                    hidden = layers.Reshape([1,hidden.shape[1]])(hidden)
                hidden = layers.LSTM(units=nodes_n[ll], input_shape=(input.shape[1],input.shape[2]), 
                                    return_sequences = ret_seq[ll], 
                                    recurrent_dropout=rec_drop[ll],
                                    #dropout=dropout_r[ll]),
                                    activation=activ_f[ll],
                                    kernel_regularizer=my_reg)(hidden)
                hidden = layers.Dropout(rate=dropout_r[ll],seed=123)(hidden)
            elif layer_ty[ll] == 'Dense':
                hidden = layers.Dense(units=nodes_n[ll], activation=activ_f[ll],
                                        kernel_regularizer=my_reg)(hidden)
                hidden = layers.Dropout(rate=dropout_r[ll],seed=123)(hidden)
        
    # ---- Output Layer
    if layer_ty[layer_n-1] == 'LSTM':
        if len(hidden.shape) < 3:
                    hidden = layers.Flatten()(hidden)
                    hidden = layers.Reshape([1,hidden.shape[1]])(hidden)
        output = layers.LSTM(units=nodes_n[layer_n-1], input_shape=(hidden.shape[1],hidden.shape[2]), 
                            return_sequences = ret_seq[layer_n-1], 
                            recurrent_dropout=rec_drop[layer_n-1],
                            #dropout=dropout_r[layer_n-1]),
                            activation=activ_f[layer_n-1],
                            kernel_regularizer=my_reg)(hidden)
    elif layer_ty[layer_n-1] == 'Dense':
        output = layers.Dense(units=nodes_n[layer_n-1], activation=activ_f[layer_n-1],
                                kernel_regularizer=my_reg)(hidden)
    

    # ---- Set the model and compile
    model = keras.Model(inputs=input, outputs=output)

    if hyparams['optimizer'] == 'Adam':
        optim = keras.optimizers.Adam(learning_rate=hyparams['learning_rate'],
                                      beta_1 = hyparams['Adam_b1'],
                                      beta_2 = hyparams['Adam_b2'],
                                     )
    elif hyparams['optimizer'] == 'RMSprop':
        optim = keras.optimizers.RMSprop(learning_rate=hyparams['learning_rate'],
                                         rho = hyparams['RMSprop_rho'],
                                         momentum = hyparams['momentum']
                                        )
    elif hyparams['optimizer'] == 'SGD':
        optim = keras.optimizers.SGD(learning_rate=hyparams['learning_rate'],
                                    momentum = hyparams['momentum'])

    if hyparams['loss_function'] == 'bin_foc_loss':
        model.compile(loss=functf_binary_focal_loss(alpha=hyparams['alpha'],gamma=hyparams['gamma']),
                      optimizer=optim,
                      metrics=hyparams['metrics'])
    else: 
        model.compile(loss=hyparams['loss_function'],optimizer=optim,
                      metrics=hyparams['metrics'])

    return model

def func_buildLogit(hyparams):

    a = hyparams['alpha']
    g = hyparams['gamma']
    reg_l1 = hyparams['reg_l1']   

    if hyparams['loss_function'] == 'bin_foc_loss':
        model = keras.Sequential()
        model.add(layers.Dense(units=1, activation='sigmoid',
                                kernel_regularizer=keras.regularizers.l1(reg_l1)))
        if hyparams['optimizer'] == 'Adam':
            optim = keras.optimizers.Adam(learning_rate=hyparams['learning_rate'],
                                        beta_1 = hyparams['Adam_b1'],
                                        beta_2 = hyparams['Adam_b2'],
                                        )
        elif hyparams['optimizer'] == 'RMSprop':
            optim = keras.optimizers.RMSprop(learning_rate=hyparams['learning_rate'],
                                            rho = hyparams['RMSprop_rho'],
                                            momentum = hyparams['momentum']
                                            )
        elif hyparams['optimizer'] == 'SGD':
            optim = keras.optimizers.SGD(learning_rate=hyparams['learning_rate'],
                                        momentum = hyparams['momentum'])
        model.compile(loss=functf_binary_focal_loss(alpha=a,gamma=g),optimizer=optim,
                    metrics=hyparams['metrics'])
    else:
        model = keras.Sequential()
        model.add(layers.Dense(units=1, activation='sigmoid',
                                kernel_regularizer=keras.regularizers.l1(reg_l1)))

        if hyparams['optimizer'] == 'Adam':
            optim = keras.optimizers.Adam(learning_rate=hyparams['learning_rate'],
                                        beta_1 = hyparams['Adam_b1'],
                                        beta_2 = hyparams['Adam_b2'],
                                        )
        elif hyparams['optimizer'] == 'RMSprop':
            optim = keras.optimizers.RMSprop(learning_rate=hyparams['learning_rate'],
                                            rho = hyparams['RMSprop_rho'],
                                            momentum = hyparams['momentum']
                                            )
        elif hyparams['optimizer'] == 'SGD':
            optim = keras.optimizers.SGD(learning_rate=hyparams['learning_rate'],
                                        momentum = hyparams['momentum'])

        model.compile(loss=hyparams['loss_function'],optimizer=optim,
                    metrics=hyparams['metrics'])

    return model


class functf_binary_focal_loss(keras.losses.Loss):

    def __init__(self, alpha, gamma, name='bin_foc_loss'):
        super().__init__(name=name)
        self.alpha = alpha
        self.gamma = tf.convert_to_tensor(gamma, dtype=tf.float32)

    def call(self, y_true, y_pred):
        p_t = y_true * y_pred +  tf.subtract(1.0,y_true) * tf.subtract(1.0,y_pred)
        alpha_t = tf.multiply(y_true,self.alpha) + tf.subtract(1.0,y_true) * tf.subtract(1.0,self.alpha)
        return tf.reduce_mean(- alpha_t * tf.math.pow(tf.convert_to_tensor(tf.subtract(1.0,p_t), dtype=tf.float32),self.gamma) * tf.math.log(tf.convert_to_tensor(p_t, dtype=tf.float32)))


def func_binary_focal_loss(y_pred,y_true,hyparams):

    a = hyparams['alpha']
    g = hyparams['gamma']

    p_t = y_true * y_pred + (1 - y_true) * (1 - y_pred)
    alpha_t = y_true * a + (1 - y_true) * (1 - a)

    return tf.math.reduce_mean(- alpha_t * (1 - p_t)**g * np.log(np.maximum(p_t,1e-5)))


def func_BlockBS(X,Y,block_size):

    # ---- Draw index
    idx = np.random.choice(range(0,X.shape[0]-block_size), int(X.shape[0] / block_size))

    # --- Build the index
    idx_BS = []
    for ii in range(0,len(idx)):
        idx_BS = idx_BS + list(range(idx[ii],idx[ii]+block_size))

    # --- If the length of the index is shorter than 'X', fill the missing obs
    if len(idx_BS) < np.shape(X)[0]:
        idx_miss = np.random.choice(range(0,X.shape[0]-block_size),1)
        obs_miss = np.shape(X)[0] - len(idx_BS)
        idx_BS = idx_BS + list(range(idx_miss[0],idx_miss[0]+obs_miss))

    # ---- Draw whole rows
    X_BS = X[idx_BS,:]
    Y_BS = Y[idx_BS,:]

    # In Case you want Out-Of-Bag
    #idx_oob = [i for i in range(0,X.shape[0]) if i not in idx]
    #X_oob = X[idx_oob,:]
    #Y_oob = Y[idx_oob,:]

    return X_BS, Y_BS

def func_neg_loglik(y_true,y_pred):
    return -y_pred.log_prob(y_true)


def func_plot_USREC(Y_data,pred_train,pred_dev,pred_test):

    timesteps = pd.to_datetime(Y_data.Date.values)

    # ---- Helping arrays
    nan_train = np.empty(len(pred_train))
    nan_train.fill(np.nan)
    nan_dev = np.empty(len(pred_dev))
    nan_dev.fill(np.nan)
    nan_test = np.empty(len(pred_test))
    nan_test.fill(np.nan)

    # ---- Train-,Dev-,Test-Predictions
    plot_train = np.concatenate((pred_train.flatten(),nan_dev,nan_test))
    if len(pred_dev) == 0:
        plot_dev = np.concatenate((nan_train,nan_test))
    else:
        plot_dev = np.concatenate((nan_train,pred_dev.flatten(),nan_test))
    plot_test = np.concatenate((nan_train,nan_dev,pred_test.flatten(),))

    # ---- Plotting
    plt.figure(figsize=(30,15))
    fig, ax = plt.subplots()

    # ---- Shade the NBER Periods
    ax.fill_between(pd.to_datetime(Y_data.Date.values), 0,1, 
                    where=Y_data['USREC'], 
                    transform=ax.get_xaxis_transform(),
                    color='gray')

    #plt.plot(timesteps,Y_test,color='red',label='Data: Observations')
    plt.plot(timesteps,plot_train,color='green',label='Model: Train-Set-Prediction')
    plt.plot(timesteps,plot_dev,color='yellow',label='Model: Validation-Set-Prediction')
    plt.plot(timesteps,plot_test,color='red',label='Model: Test-Set-Prediction')

    # ---- Some cosmetics
    plt.xlim([timesteps[0],timesteps[len(timesteps)-1]])
    plt.ylabel('Probability of Recession')

    # ---- Show the plot
    plt.show()

    # ---- Also Export the dataframe
    plot_df = pd.DataFrame(data={'Date':timesteps,'pred_train':plot_train,'pred_dev':plot_dev,'pred_test':plot_test})


    return fig, plot_df





def func_plotBS_USREC(Y_data,pred_train,pred_dev,pred_test):


    # ---- Extract Median-, High-, Low- Bounds from Dictionaries
    pred_train_median = pred_train['median']
    pred_train_low = pred_train['low']
    pred_train_up = pred_train['up']

    pred_dev_median = pred_dev['median']
    pred_dev_low = pred_dev['low']
    pred_dev_up = pred_dev['up']

    pred_test_median = pred_test['median']
    pred_test_low = pred_test['low']
    pred_test_up = pred_test['up']


    # --- Define the x-Axis
    timesteps = pd.to_datetime(Y_data.Date.values)

    # ---- Helping arrays
    nan_train = np.empty(len(pred_train_median))
    nan_train.fill(np.nan)
    nan_dev = np.empty(len(pred_dev_median))
    nan_dev.fill(np.nan)
    nan_test = np.empty(len(pred_test_median))
    nan_test.fill(np.nan)

    # ---- Training-Set
    plot_train_median = np.concatenate((pred_train_median.flatten(),nan_dev,nan_test))
    plot_train_low = np.concatenate((pred_train_low.flatten(),nan_dev,nan_test))
    plot_train_up = np.concatenate((pred_train_up.flatten(),nan_dev,nan_test))

    # ---- Validation-Set
    if len(pred_dev_median) == 0:
        plot_dev_median = np.concatenate((nan_train,nan_test))
        plot_dev_low = np.concatenate((nan_train,nan_test))
        plot_dev_up = np.concatenate((nan_train,nan_test))
    else:
        plot_dev_median = np.concatenate((nan_train,pred_dev_median.flatten(),nan_test))
        plot_dev_low = np.concatenate((nan_train,pred_dev_low.flatten(),nan_test))
        plot_dev_up = np.concatenate((nan_train,pred_dev_up.flatten(),nan_test))

    # ---- Test-Set
    plot_test_median = np.concatenate((nan_train,nan_dev,pred_test_median.flatten(),))
    plot_test_low = np.concatenate((nan_train,nan_dev,pred_test_low.flatten(),))
    plot_test_up = np.concatenate((nan_train,nan_dev,pred_test_up.flatten(),))

    # ---- Plotting
    plt.figure(figsize=(30,15))
    fig, ax = plt.subplots()

    # ---- Shade the NBER Periods
    ax.fill_between(pd.to_datetime(Y_data.Date.values), 0,1, 
                    where=Y_data['USREC'], 
                    transform=ax.get_xaxis_transform(),
                    color='gray')

    # --- Plot the median predictions
    #plt.plot(timesteps,Y_test,color='red',label='Data: Observations')
    plt.plot(timesteps,plot_train_median,color='green',label='Model: Train-Set-Prediction')
    plt.plot(timesteps,plot_dev_median,color='yellow',label='Model: Validation-Set-Prediction')
    plt.plot(timesteps,plot_test_median,color='red',label='Model: Test-Set-Prediction')


    # --- Plot the bands
    plt.fill_between(pd.to_datetime(Y_data.Date.values), 
                plot_train_low.flatten(),plot_train_up.flatten(), 
                alpha=0.5, 
                color='green')
    plt.fill_between(pd.to_datetime(Y_data.Date.values), 
                plot_dev_low.flatten(),plot_dev_up.flatten(), 
                alpha=0.5, 
                color='yellow')
    plt.fill_between(pd.to_datetime(Y_data.Date.values), 
                plot_test_low.flatten(),plot_test_up.flatten(), 
                alpha=0.5, 
                color='red')


    # ---- Some cosmetics
    plt.xlim([timesteps[0],timesteps[len(timesteps)-1]])
    plt.ylabel('Probability of Recession')

    # ---- Show the plot
    plt.show()

    # ---- Also Export the dataframe
    plot_df = pd.DataFrame(data={'Date':timesteps,
                                 'pred_train_median':plot_train_median,
                                 'pred_train_low':plot_train_low,
                                 'pred_train_up':plot_train_up,
                                 'pred_dev_median':plot_dev_median,
                                 'pred_dev_low':plot_dev_low,
                                 'pred_dev_up':plot_dev_up,
                                 'pred_test_median':plot_test_median,
                                 'pred_test_low':plot_test_low,
                                 'pred_test_up':plot_test_up,})


    return fig, plot_df




def func_eval_train(history):

        eval_train = pd.DataFrame(history.history)
        eval_train['epoch'] = history.epoch

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Abs Error [MPG]')
        plt.plot(eval_train['epoch'], eval_train['mae'],
                label='Train Error')
        plt.plot(eval_train['epoch'], eval_train['val_mae'],
                label = 'Val Error')
        plt.ylim([0,0.5])
        plt.legend()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Mean Square Error [$MPG^2$]')
        plt.plot(eval_train['epoch'], eval_train['mse'],
                label='Train Error')
        plt.plot(eval_train['epoch'], eval_train['val_mse'],
                label = 'Val Error')
        plt.ylim([0,0.25])
        plt.legend()
        plt.show()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.plot(eval_train['epoch'], eval_train['loss'],
                label='Training Set')
        plt.plot(eval_train['epoch'], eval_train['val_loss'],
                label = 'Val Set')
        plt.ylim([0,1])
        plt.legend()
        plt.show()

        plt.figure()
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.plot(eval_train['epoch'], eval_train['acc'],
                label='Training Set')
        plt.plot(eval_train['epoch'], eval_train['val_acc'],
                label = 'Val Set')
        plt.ylim([0,1.025])
        plt.legend()
        plt.show()


def func_buildComponentNN(nn_comps, X_train_C):

    # --- Unpack the ingredients of the Component-NN
    groups_C = nn_comps['groups']
    merge_C = nn_comps['merge']
    layers_C = nn_comps['layers']
    nodes_C = nn_comps['nodes']
    activation_C = nn_comps['activation']
    dropout_C = nn_comps['dropout']
    reg_l1_C = nn_comps['reg_l1']
    nn_hyparams_general = nn_comps['nn_hyparams']

    # --- I) Block 1: Build Individual Components
    C_B1 = {}
    Concat_C_B1 = {}
    Final_C_B1 = {}
    input_dict = {}
    #input_dict = locals()

    # --- Run over the 'merge_C' indices: 
    # --- --- first, create network for those components with "Position 2"-index of "1"
    # --- --- second, merge the output of the above with the components with a "Position 2"-index of "0"
    for mm in range(len(merge_C)):

            C_B1[mm] = []

            # --- Run over the indices stored in "Position 2"
            for pp in range(len(merge_C[mm][1])):

                    # --- What is the position in 'group_C', i.e. "Position 1" in 'merge_C'?
                    gg = merge_C[mm][0][pp]

                    # --- Define the Input
                    mm_gg_input = tf.keras.layers.Input(shape=(np.shape(X_train_C[gg])[1],), name='C' + str(mm) + '_' + str(gg))
                    #input_dict[mm+pp] = mm_gg_input
                    input_dict[f"input_{mm}_{gg}"] = mm_gg_input
                    #exec(f"input_{mm}_{gg} = mm_gg_input",globals(),input_dict)

            # --- If merge_C[mm][1][pp] == 1, set the Network for those Components,
            # --- --- i.e. if "Position 2" == 1, because you will transform this component first before combining it with other components
                    if merge_C[mm][1][pp] == 1:
                            gg_hidden = layers.Dense(units = nodes_C[gg][0],
                                            activation = activation_C[gg][0],
                                            kernel_regularizer =  keras.regularizers.l1(reg_l1_C[gg][0])
                                            )(mm_gg_input)
                            gg_hidden = layers.Dropout(dropout_C[gg][0])(gg_hidden)

                            # --- Run over Further Layers, if there are
                            if len(layers_C[gg]) > 1:

                                    for ll in range(1,len(layers_C[gg])):

                                            gg_hidden = layers.Dense(units = nodes_C[gg][ll],
                                                                    activation = activation_C[gg][ll],
                                                                    kernel_regularizer =  keras.regularizers.l1(reg_l1_C[gg][ll])
                                                                    )(gg_hidden)
                                            gg_hidden = layers.Dropout(dropout_C[gg][ll])(gg_hidden)

                            # --- Store the created Component in 'C_B1'
                            C_B1[mm].append(gg_hidden)

                    elif merge_C[mm][1][pp] == 0:
                            C_B1[mm].append(mm_gg_input)


    # --- Combine each Set of Components in 'C_B1'
    for cb1 in range(len(C_B1)):
            Concat_C_B1[cb1] = C_B1[cb1][0]

            if len(C_B1[cb1]) > 1: 

                    for ll in range(1,len(C_B1[cb1])):

                            Concat_C_B1[cb1] = layers.concatenate([Concat_C_B1[cb1], C_B1[cb1][ll]])


    # --- For Each Concatenated Set in 'Concat_C_B1': 
    # --- --- Create the Network Structure, according to the <structure> specified by "Position 2" == 0

    for mm in range(len(merge_C)):

            if 0 in merge_C[mm][1]:

                    Final_C_B1[mm] = []

                    idx_struc = merge_C[mm][1].index(0)

                    gg_hidden = layers.Dense(units = nodes_C[idx_struc][0],
                                            activation = activation_C[idx_struc][0],
                                            kernel_regularizer =  keras.regularizers.l1(reg_l1_C[idx_struc][0])
                                            )(Concat_C_B1[mm])
                    gg_hidden = layers.Dropout(dropout_C[idx_struc][0])(gg_hidden)

                    # --- Run over Further Layers, if there are
                    if len(layers_C[idx_struc]) > 1:

                            for ll in range(1,len(layers_C[idx_struc])):

                                    gg_hidden = layers.Dense(units = nodes_C[idx_struc][ll],
                                                            activation = activation_C[idx_struc][ll],
                                                            kernel_regularizer =  keras.regularizers.l1(reg_l1_C[idx_struc][ll])
                                                            )(gg_hidden)
                                    gg_hidden = layers.Dropout(dropout_C[idx_struc][ll])(gg_hidden)

                    # --- Store the created Component in 'C_B1'
                    Final_C_B1[mm] = gg_hidden

            else:
                Final_C_B1[mm] =  Concat_C_B1[mm]
                

                               
    # ----------------------------------------------------------------------------------------- #
    # --- II) Block 2: If there are several Components in 'Final_C_B1', combine them all

    if len(Final_C_B1) > 1: 

            C_B2 = Final_C_B1[0]

            for ll in range(1,len(Final_C_B1)):

                    C_B2 = layers.concatenate([C_B2, Final_C_B1[ll]])

            # --- Block 2: Create Final Layer

            #C_B2_output = layers.Dense(units = int(len(C_B1)), activation='sigmoid',
            #                        kernel_regularizer =  keras.regularizers.l1(0.01)
            #                        )(C_B2)

            #C_B2_output = layers.Dropout(0.2)(C_B2_output)


            C_B2_output = layers.Dense(units = nn_hyparams_general['final_layer_nodes'], 
                                       activation=nn_hyparams_general['final_layer_activation'],
                                       kernel_regularizer =  keras.regularizers.l1(0.0)
                                      )(C_B2)

    else:

            C_B2_output = layers.Dense(units = nn_hyparams_general['final_layer_nodes'], 
                                       activation=nn_hyparams_general['final_layer_activation'],
                                       kernel_regularizer =  keras.regularizers.l1(0.0)
                                      )(Final_C_B1[0])


    # ----------------------------------------------------------------------------------------- #
    # --- III) Set the Model
    #components_input = [eval("input_"+str(mm)+"_"+str(gg)) for mm in range(len(merge_C)) for gg in groups_C.keys()]
    components_input = [input_dict["input_"+str(mm)+"_"+str(gg)] for mm in range(len(merge_C)) for gg in groups_C.keys()]

    model = keras.Model(
                        inputs=[components_input],
                        outputs=[C_B2_output],
                    )

    # ----------------------------------------------------------------------------------------- #
    # --- IV) Compile the Model
    if nn_hyparams_general['loss_function'] == 'binary_crossentropy':

        my_loss = keras.losses.BinaryCrossentropy(from_logits=False)

    elif nn_hyparams_general['loss_function'] == 'bin_foc_crossentropy':

        my_loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=nn_hyparams_general['gamma'],
                                                         from_logits=False, 
                                                         label_smoothing=nn_hyparams_general['alpha'])


    model.compile(
                #optimizer=keras.optimizers.RMSprop(0.0001),
                optimizer=keras.optimizers.Adam(nn_hyparams_general['learning_rate']),
                loss=[  
                        my_loss
                        
                        #keras.losses.BinaryCrossentropy(from_logits=False),
                        #keras.losses.CategoricalCrossentropy(from_logits=True),
                        #functf_binary_focal_loss(alpha=0.25,gamma=2)
                        #tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=False, 
                        #                                        label_smoothing=0.0),
                        #tfa.losses.SigmoidFocalCrossEntropy(reduction=tf.keras.losses.Reduction.AUTO),
                    ],
                metrics=nn_hyparams_general['metrics']
                #loss_weights=[1.0, 0.2],
                )

    return model



def func_build_YSNet(X_train_comps,nn_comps,with_LSTM):


    # --- Unpack the ingredients of the YS-Net

    X_train_C0 = X_train_comps[0]
    X_train_C1 = X_train_comps[1]

    layers_C = nn_comps['layers']
    nodes_C = nn_comps['nodes']
    activation_C = nn_comps['activation']
    dropout_C = nn_comps['dropout']
    reg_l1_C = nn_comps['reg_l1']
    nn_hyparams_general = nn_comps['nn_hyparams']


    # ----------------------------- Component 0 ----------------------------- #

    # --- Input Layer
    if with_LSTM == True:
            C0_input = tf.keras.Input(shape = (np.shape(X_train_C0)[1],np.shape(X_train_C0)[2]), 
                                    name='C0')
    else:
            C0_input = tf.keras.Input(shape = (np.shape(X_train_C0)[1],), name='C0')

    # --- Hidden Layers
    if layers_C[0][0] == 'Dense':
                    C0_hidden = layers.Dense(units=nodes_C[0][0], 
                                            activation=activation_C[0][0],
                                            kernel_regularizer=keras.regularizers.l1(reg_l1_C[0][0]),
                                             name="C0_hidden_0")(C0_input)
                    C0_hidden = layers.Dropout(dropout_C[0][0],name="C0_dropout_0")(C0_hidden)

    if len(layers_C[0]) > 1:
            for ll in range(1,len(layers_C[0])-1):

                    if layers_C[0][ll] == 'Dense':
                            C0_hidden = layers.Dense(units=nodes_C[0][ll], 
                                                    activation=activation_C[0][ll],
                                                    kernel_regularizer=keras.regularizers.l1(reg_l1_C[0][ll]),
                                                     name="C0_hidden_"+str(ll))(C0_hidden)
                            C0_hidden = layers.Dropout(dropout_C[0][ll],name="C0_dropout_"+str(ll))(C0_hidden)
            
    # --- Output Layer
    if layers_C[0][len(layers_C[0])-1] == 'Dense':
            C0_output = layers.Dense(units=nodes_C[0][len(layers_C[0])-1], 
                                    activation=activation_C[0][len(layers_C[0])-1], 
                                    kernel_regularizer=keras.regularizers.l1(reg_l1_C[0][len(layers_C[0])-1]),
                                    name="C0_output")(C0_hidden)
            if with_LSTM == True and len(C0_output.shape) > 2:
                    C0_output = layers.Reshape((C0_output.shape[2],))(C0_output)

    C0_model = tf.keras.Model(inputs=C0_input, outputs=C0_output)


    # ----------------------------- Component 1 ----------------------------- #

    # --- Input Layer
    if with_LSTM == True:
            C1_input = tf.keras.Input(shape = (np.shape(X_train_C1)[1],np.shape(X_train_C1)[2]), 
                                    name='C1')
    else:
            C1_input = tf.keras.Input(shape = (np.shape(X_train_C1)[1],), name='C1')

    # --- Hidden Layers
    if layers_C[1][0] == 'Dense':
                    C1_hidden = layers.Dense(units=nodes_C[1][0], 
                                            activation=activation_C[1][0],
                                            kernel_regularizer=keras.regularizers.l1(reg_l1_C[1][0]),
                                             name="C1_hidden_0")(C1_input)
                    C1_hidden = layers.Dropout(dropout_C[1][0],name="C1_dropout_0")(C1_hidden)

    if len(layers_C[1]) > 1:
            for ll in range(1,len(layers_C[1])-1):

                    if layers_C[1][ll] == 'Dense':
                            C1_hidden = layers.Dense(units=nodes_C[1][ll], 
                                                    activation=activation_C[1][ll],
                                                    kernel_regularizer=keras.regularizers.l1(reg_l1_C[1][ll]),
                                                     name="C1_hidden_"+str(ll))(C1_hidden)
                            C1_hidden = layers.Dropout(dropout_C[1][ll],name="C1_dropout_"+str(ll))(C1_hidden)
            
    # --- Output Layer
    if layers_C[1][len(layers_C[1])-1] == 'Dense':
            C1_output = layers.Dense(units=nodes_C[1][len(layers_C[1])-1], 
                                    activation=activation_C[1][len(layers_C[1])-1], 
                                    kernel_regularizer=keras.regularizers.l1(reg_l1_C[1][len(layers_C[1])-1]),
                                    name="C1_output")(C1_hidden)
            if with_LSTM == True and len(C1_output.shape) > 2:
                    C1_output = layers.Reshape((C1_output.shape[2],))(C1_output)


    C1_model = tf.keras.Model(inputs=C1_input, outputs=C1_output)

    # ----------------------------------------------------------------------------------------- #
    # --- II) 3) Combine Component 0 & Component 1

    class Final_Layer(keras.layers.Layer):
            def __init__(self, units=1,name="YSNet"):
                    super(Final_Layer, self).__init__(name=name)
                    b_init = tf.zeros_initializer()
                    self.b = tf.Variable(initial_value=b_init(shape=(units,), 
                                        dtype="float32"), trainable=False)

            def call(self, input_params):
                    features = input_params[0]
                    weights = input_params[1]
                    a_Final_Layer = tf.nn.sigmoid(tf.reduce_sum(tf.keras.layers.Multiply()([features,weights]) + self.b,axis=1,keepdims=True))
                    #a_Final_Layer = tf.nn.sigmoid(tf.reduce_sum(tf.keras.layers.Multiply()([features,weights]),axis=1,keepdims=True))

                    return a_Final_Layer

    output = Final_Layer()

    #C0_model_pred = tf.Variable(C0_model.predict(X_train_C0), trainable=False)
    #C1_model_pred = tf.Variable(C1_model.predict(X_train_C1), trainable=False)
    #final_output = output(features=C0_model_pred, weights=C1_model_pred)
    # tf.reduce_sum(tf.keras.layers.Multiply()([C0_model_pred,C1_model_pred]),axis=1,keepdims=True)
    # tf.nn.sigmoid(tf.reduce_sum(tf.keras.layers.Multiply()([C0_model_pred,C1_model_pred]),axis=1,keepdims=True))


    C0_model_out = C0_model.get_layer("C0_output").output
    C1_model_out = C1_model.get_layer("C1_output").output
    final_output = output([C0_model_out,C1_model_out])

    
    # ----------------------------------------------------------------------------------------- #
    # --- Set the Model
    model = keras.Model(
                        inputs=[C0_input, C1_input],
                        outputs=[final_output],
                    )

    # --- Define the Loss Function
    if nn_hyparams_general['loss_function'] == 'binary_crossentropy':

            my_loss = tf.keras.losses.BinaryCrossentropy(from_logits=False)

    elif nn_hyparams_general['loss_function'] == 'bin_foc_loss':

            my_loss = functf_binary_focal_loss(alpha=nn_hyparams_general['alpha'],
                                            gamma=nn_hyparams_general['gamma'])

    elif nn_hyparams_general['loss_function'] == 'binary_focalcrossentropy':

            my_loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=nn_hyparams_general['gamma'], 
                                                            from_logits=False, 
                                                            label_smoothing=nn_hyparams_general['alpha'])


    # --- Compile the Model
    model.compile(
                #optimizer=keras.optimizers.RMSprop(0.0001),
                optimizer=keras.optimizers.Adam(nn_hyparams_general['learning_rate']),
                loss=[my_loss],
                metrics=nn_hyparams_general['metrics']
                )

    out = {'model':model,'C0_model':C0_model,'C1_model':C1_model}

    return out


def transform(column, transforms):
    transformation = transforms[column.name]
    # For quarterly data like GDP, we will compute
    # annualized percent changes
    mult = 4 if column.index.freqstr[0] == 'Q' else 1
    
    # 1 => No transformation
    if transformation == 1:
        pass
    # 2 => First difference
    elif transformation == 2:
        column = column.diff()
    # 3 => Second difference
    elif transformation == 3:
        column = column.diff().diff()
    # 4 => Log
    elif transformation == 4:
        column = np.log(column)
    # 5 => Log first difference, multiplied by 100
    #      (i.e. approximate percent change)
    #      with optional multiplier for annualization
    elif transformation == 5:
        column = np.log(column).diff() * 100 * mult
    # 6 => Log second difference, multiplied by 100
    #      with optional multiplier for annualization
    elif transformation == 6:
        column = np.log(column).diff().diff() * 100 * mult
    # 7 => Exact percent change, multiplied by 100
    #      with optional annualization
    elif transformation == 7:
        column = ((column / column.shift(1))**mult - 1.0) * 100
        
    return column


def remove_outliers(dta):
    # Compute the mean and interquartile range
    mean = dta.mean()
    iqr = dta.quantile([0.25, 0.75]).diff().T.iloc[:, 1]
    
    # Replace entries that are more than 10 times the IQR
    # away from the mean with NaN (denotes a missing entry)
    mask = np.abs(dta) > mean + 10 * iqr
    treated = dta.copy()
    treated[mask] = np.nan

    return treated


def load_fredmd_data(filename):
    #base_url = 'https://s3.amazonaws.com/files.fred.stlouisfed.org/fred-md'
    
    # - FRED-MD --------------------------------------------------------------
    # 1. Download data
    #orig_m = (pd.read_csv(f'{base_url}/monthly/{vintage}.csv')
                #.dropna(how='all'))
    orig_m = (pd.read_csv(f'{filename}.csv').dropna(how='all'))
    
    # 2. Extract transformation information
    transform_m = orig_m.iloc[0, 1:]
    orig_m = orig_m.iloc[1:]

    # 3. Extract the date as an index
    orig_m.index = pd.PeriodIndex(orig_m.sasdate.tolist(), freq='M')
    orig_m.drop('sasdate', axis=1, inplace=True)

    # 4. Apply the transformations
    dta_m = orig_m.apply(transform, axis=0,
                         transforms=transform_m)

    # 5. Remove outliers (but not in 2020)
    dta_m.loc[:'2019-12'] = remove_outliers(dta_m.loc[:'2019-12'])
    
    return types.SimpleNamespace(
        orig_m=orig_m, #orig_q=orig_q,
        dta_m=dta_m, transform_m=transform_m)

    # In case you one day want to play with quarterly data:
        #dta_q=dta_q, transform_q=transform_q, factors_q=factors_q)

def func_Future_lags(data,rhs_names,lags):

    # Create an storage Data Frame
    data_rhs = pd.DataFrame(data=data.loc[(data.shape[0]-min(lags)):,'Date'],
                            columns=['Date']).reset_index(drop=True)

    # --- 'Date' shall display the date for which the forecast is made:
    # --- ---> add 'min(lags)' to each 'Date'
    dates_future = pd.to_datetime(data_rhs['Date']) + pd.offsets.DateOffset(months=min(lags))
    date_string = [str(dates_future[ii])[:7] for ii in range(len(dates_future))]
    data_rhs.Date = date_string


    # Which variables will go into your Xtrain/Xtest?
    rhs_colnames_store = []

    for ii in lags:
        
        # Create the new Column-Names
        rhs_colnames = ["L" + str(ii) + "_" + cc for cc in rhs_names]

        # Collect the variable names
        rhs_colnames_store.extend(rhs_colnames)
        
        # Collect the very last 'ii'-observations of predictors 'rhs_names'
        idx_start = data.shape[0] - ii
        data_rhs.loc[:,rhs_colnames] = 0
        data_rhs.loc[:,rhs_colnames] = np.array(data.loc[idx_start:(idx_start+min(lags)-1),rhs_names])

    return data_rhs, rhs_colnames_store

def func_build_DataSet_Future(data,X_var,Y_var,lags,my_scale):

    # ---- First: Some Housekeeping, e.g. delete columns with NAs
    # Are there any "intermediate" NA's?
    nans = data.columns[data.isna().any()]
    for ii in nans:
        while ii in X_var: X_var.remove(ii)  

    # ======================= Create the Predictor-Set for the Future ========================= #

    # --- Get data:
    X_data_future, rhs_names = func_Future_lags(data,X_var,lags)

    # --- Scale data:
    if Y_var == ['USREC']:
        data_future_sc = (X_data_future.loc[:,rhs_names] - my_scale['numerator']) / my_scale['denominator']

        data_future_sc = pd.DataFrame(np.concatenate((np.zeros((min(lags),1)),data_future_sc), axis=1),
                                    columns=Y_var + rhs_names)

    else:
        data_future_sc = (X_data_future[rhs_names] - my_scale['numerator']) / my_scale['denominator']

    X_future = np.array(data_future_sc[rhs_names])

    # Collect everyting in an output dictionary
    dict_out = {
                'X_future': X_future,
                'X_data_future': X_data_future,
               }


    return dict_out


def func_plotBS_USREC_Future(Y_data,pred_train,pred_dev,pred_test,pred_future):


    # ---- Extract Median-, High-, Low- Bounds from Dictionaries
    pred_train_median = pred_train['median']
    pred_train_low = pred_train['low']
    pred_train_up = pred_train['up']

    pred_dev_median = pred_dev['median']
    pred_dev_low = pred_dev['low']
    pred_dev_up = pred_dev['up']

    pred_test_median = pred_test['median']
    pred_test_low = pred_test['low']
    pred_test_up = pred_test['up']

    pred_future_median = pred_future['median']
    pred_future_low = pred_future['low']
    pred_future_up = pred_future['up']


    # --- Define the x-Axis
    timesteps = pd.to_datetime(Y_data.Date.values)

    # ---- Helping arrays
    nan_train = np.empty(len(pred_train_median))
    nan_train.fill(np.nan)
    nan_dev = np.empty(len(pred_dev_median))
    nan_dev.fill(np.nan)
    nan_test = np.empty(len(pred_test_median))
    nan_test.fill(np.nan)
    nan_future = np.empty(len(pred_future_median))
    nan_future.fill(np.nan)

    # ---- Training-Set
    plot_train_median = np.concatenate((pred_train_median.flatten(),nan_dev,nan_test,nan_future))
    plot_train_low = np.concatenate((pred_train_low.flatten(),nan_dev,nan_test,nan_future))
    plot_train_up = np.concatenate((pred_train_up.flatten(),nan_dev,nan_test,nan_future))

    # ---- Validation-Set
    if len(pred_dev_median) == 0:
        plot_dev_median = np.concatenate((nan_train,nan_test,nan_future))
        plot_dev_low = np.concatenate((nan_train,nan_test,nan_future))
        plot_dev_up = np.concatenate((nan_train,nan_test,nan_future))
    else:
        plot_dev_median = np.concatenate((nan_train,pred_dev_median.flatten(),nan_test,nan_future))
        plot_dev_low = np.concatenate((nan_train,pred_dev_low.flatten(),nan_test,nan_future))
        plot_dev_up = np.concatenate((nan_train,pred_dev_up.flatten(),nan_test,nan_future))

    # ---- Test-Set
    plot_test_median = np.concatenate((nan_train,nan_dev,pred_test_median.flatten(),nan_future))
    plot_test_low = np.concatenate((nan_train,nan_dev,pred_test_low.flatten(),nan_future))
    plot_test_up = np.concatenate((nan_train,nan_dev,pred_test_up.flatten(),nan_future))

    # ---- Future-Set
    plot_future_median = np.concatenate((nan_train,nan_dev,nan_test,pred_future_median.flatten()))
    plot_future_low = np.concatenate((nan_train,nan_dev,nan_test,pred_future_low.flatten()))
    plot_future_up = np.concatenate((nan_train,nan_dev,nan_test,pred_future_up.flatten()))


    # ---- Plotting
    plt.figure(figsize=(30,15))
    fig, ax = plt.subplots()

    # ---- Shade the NBER Periods
    ax.fill_between(pd.to_datetime(Y_data.Date.values), 0,1, 
                    where=Y_data['USREC'], 
                    transform=ax.get_xaxis_transform(),
                    color='gray')

    # --- Plot the median predictions
    #plt.plot(timesteps,Y_test,color='red',label='Data: Observations')
    plt.plot(timesteps,plot_train_median,color='green',label='Model: Train-Set-Prediction')
    plt.plot(timesteps,plot_dev_median,color='yellow',label='Model: Validation-Set-Prediction')
    plt.plot(timesteps,plot_test_median,color='red',label='Model: Test-Set-Prediction')
    plt.plot(timesteps,plot_future_median,color='blue',label='Model: Future-Outlook')


    # --- Plot the bands
    plt.fill_between(pd.to_datetime(Y_data.Date.values), 
                plot_train_low.flatten(),plot_train_up.flatten(), 
                alpha=0.5, 
                color='green')
    plt.fill_between(pd.to_datetime(Y_data.Date.values), 
                plot_dev_low.flatten(),plot_dev_up.flatten(), 
                alpha=0.5, 
                color='yellow')
    plt.fill_between(pd.to_datetime(Y_data.Date.values), 
                plot_test_low.flatten(),plot_test_up.flatten(), 
                alpha=0.5, 
                color='red')
    plt.fill_between(pd.to_datetime(Y_data.Date.values), 
                plot_future_low.flatten(),plot_future_up.flatten(), 
                alpha=0.5, 
                color='blue')


    # ---- Some cosmetics
    plt.xlim([timesteps[0],timesteps[len(timesteps)-1]])
    plt.ylabel('Probability of Recession')

    # ---- Show the plot
    plt.show()

    # ---- Also Export the dataframe
    plot_df = pd.DataFrame(data={'Date':timesteps,
                                 'pred_train_median':plot_train_median,
                                 'pred_train_low':plot_train_low,
                                 'pred_train_up':plot_train_up,
                                 'pred_dev_median':plot_dev_median,
                                 'pred_dev_low':plot_dev_low,
                                 'pred_dev_up':plot_dev_up,
                                 'pred_test_median':plot_test_median,
                                 'pred_test_low':plot_test_low,
                                 'pred_test_up':plot_test_up,
                                 'pred_future_median':plot_future_median,
                                 'pred_future_low':plot_future_low,
                                 'pred_future_up':plot_future_up})


    return fig, plot_df