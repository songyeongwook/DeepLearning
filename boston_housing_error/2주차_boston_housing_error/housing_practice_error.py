
import pandas as pd
file_path = "Boston_Housing.csv"

HomeDataDirectory = 'Data'

Data_file_path = HomeDataDirectory + file_path

df_data = pd.read_csv("pr_week_02_boston_housing_error/2주차_boston_housing_error/Boston_Housing.csv")

print(df_data.shape)

#%%
import numpy as np

bh_data = np.array(df_data.values, dtype=np.float32)

x_train = np.array(bh_data[:400,:12], dtype=np.float32)
y_train = np.array(bh_data[:400,12], dtype=np.float32).reshape(-1,1)

x_test = np.array(bh_data[400:,:12], dtype=np.float32)
y_test = np.array(bh_data[400:,12], dtype=np.float32).reshape(-1,1)

print(x_train.shape, y_train.shape, x_test.shape, y_test.shape)


# #%%
from sklearn.linear_model import LinearRegression

lr = LinearRegression() 

lr.fit(x_train, y_train) 

y_pred = lr.predict(x_test)


import matplotlib.pyplot as plt
plt.close("all")
plt.title("실제 값 vs 예측 값")
plt.plot(y_pred, 'b', label='실제 값') 
plt.plot(y_test, 'r', label='예측 값') 
plt.show()
