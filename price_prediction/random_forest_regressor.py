import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import matplotlib.pyplot as plt

# Collect and clean data
ticker = "AAPL"
df = pd.read_csv(f'./data/{ticker}.csv')

# # Visualization
# df.plot(x='Date', y='Close')
# plt.show()

# Create the model
model = RandomForestRegressor()

# Train the model
X = df[['Open', 'High', 'Low', 'Volume']] # Features
X = X[:int(len(df)-1)] # Excluding the last row
y = df['Close']
y = y[:int(len(df)-1)]
model.fit(X, y)

# Test the model
predictions = model.predict(X)
print('The model score is: ', model.score(X, y))

# Make predictions
new_data = df[['Open', 'High', 'Low', 'Volume']].tail(1)
prediction = model.predict(new_data)
print('The model predicts the last row/day to be:', prediction)
print('Actual Value:', df['Close'].tail(1))