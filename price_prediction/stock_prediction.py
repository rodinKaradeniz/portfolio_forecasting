import numpy as np
import random
import pandas as pd 
from pylab import mpl, plt
plt.style.use('seaborn')
mpl.rcParams['font.family'] = 'serif'

from lstm import LSTM
from pandas import datetime
import math, time
import itertools
import datetime
from operator import itemgetter
from sklearn.metrics import mean_squared_error
from sklearn.preprocessing import MinMaxScaler
from math import sqrt
import torch
import torch.nn as nn
from torch.autograd import Variable

# Load data
def load_data(ticker_df, look_back):
    data_raw = ticker_df.values # convert to numpy array
    data = []
    
    # create all possible sequences of length look_back
    for index in range(len(data_raw) - look_back): 
        data.append(data_raw[index: index + look_back])
    
    data = np.array(data)
    test_set_size = int(np.round(0.2 * data.shape[0]))
    train_set_size = data.shape[0] - (test_set_size)
    
    x_train = data[:train_set_size,:-1,:]
    x_train = torch.from_numpy(x_train).type(torch.Tensor)

    y_train = data[:train_set_size,-1,:]
    y_train = torch.from_numpy(y_train).type(torch.Tensor)
    
    x_test = data[train_set_size:,:-1]
    x_test = torch.from_numpy(x_test).type(torch.Tensor)

    y_test = data[train_set_size:,-1,:]
    y_test = torch.from_numpy(y_test).type(torch.Tensor)
    
    return x_train, y_train, x_test, y_test


# Training
def train(model, x_train, y_train, loss_fn, optimiser, look_back, num_epochs=100):
    hist = np.zeros(num_epochs)

    # Number of steps to unroll
    seq_dim =look_back-1  

    for t in range(num_epochs):
        # Initialise hidden state
        # Don't do this if you want your LSTM to be stateful
        # model.hidden = model.init_hidden()

        # Forward pass
        y_train_pred = model(x_train)

        loss = loss_fn(y_train_pred, y_train)
        if t % 10 == 0 and t !=0:
            print("Epoch ", t, "MSE: ", loss.item())

        hist[t] = loss.item()

        # Zero out gradient, else they will accumulate between epochs
        optimiser.zero_grad()

        # Backward pass
        loss.backward()

        # Update parameters
        optimiser.step()

    return y_train_pred

if __name__ == "__main__":
    df = pd.read_csv(f'./history_data/AAPL.csv', index_col='Date', parse_dates=True, usecols=['Date', 'Close'])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    df['Close'] = scaler.fit_transform(df['Close'].values.reshape(-1,1))

    # Sequence length
    look_back = 60

    # Get data
    x_train, y_train, x_test, y_test = load_data(df, look_back)

    # Prepare model
    input_dim = 1
    hidden_dim = 32
    num_layers = 2 
    output_dim = 1
    model = LSTM(input_dim=input_dim, hidden_dim=hidden_dim, output_dim=output_dim, num_layers=num_layers)
    loss_fn = torch.nn.MSELoss()
    optimiser = torch.optim.Adam(model.parameters(), lr=0.01)

    # Training
    y_train_pred = train(model, x_train, y_train, loss_fn, optimiser, look_back, num_epochs=100)

    # make predictions
    y_test_pred = model(x_test)

    # invert predictions
    y_train_pred = scaler.inverse_transform(y_train_pred.detach().numpy())
    y_train = scaler.inverse_transform(y_train.detach().numpy())
    y_test_pred = scaler.inverse_transform(y_test_pred.detach().numpy())
    y_test = scaler.inverse_transform(y_test.detach().numpy())

    # calculate root mean squared error
    trainScore = math.sqrt(mean_squared_error(y_train[:,0], y_train_pred[:,0]))
    print('Train Score: %.2f RMSE' % (trainScore))
    testScore = math.sqrt(mean_squared_error(y_test[:,0], y_test_pred[:,0]))
    print('Test Score: %.2f RMSE' % (testScore))