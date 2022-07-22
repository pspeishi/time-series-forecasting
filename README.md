# **Time Series Forecasting**

---

This project aims to:

* Utilise machine learning models to generate forecasts for univariate time series data
* Provide forecasts for different data frequencies (e.g. 1h interval) and forecasting horizones (e.g. next 3 days)
* Provide an automated pipeline for time series forecasting, from data preprocessing to hyperparameter tuning and selecting the best model to generate forecasts

---

## User Guide
Python version: 3.9.7

To install all packages used:
```
pip install -r requirements.txt
```

### Importing the modules
```
from utils.preprocessor import Preprocessor
from forecaster import AutomatedForecasting
```

### Preprocessor
* Converts univariate time series dataframe into a time series dataframe of the specified frequency
* Input: pandas.DataFrame indexed by DateTime (row: datetime, col: observations)
* `fit(X)`: Finds the median profile and outlier profile of X
* `transform(X)`: Processes the dataframe and removes outliers using fitted median profile and outlier profile
* Output: pandas.DataFrame indexed by DateTime (row: datetime, col: observations)
```
pp = Preprocessor(resample_freq='1h', remove_outlier=True)
df_pp = pp.fit(df).transform(df)
```
Parameters:
* `resample_freq`: Resample freqency of time series. `str, default='30min'`
* `remove_outlier`: If True, outliers will be removed. `bool, default=True`



### AutomatedForecasting
* Conducts Automated Forecasting which finds the best combination of model and hyperparameters.
* Input: pandas.DataFrame from Preprocessor
* `fit(X)`: Conducts model and hyperparameter search
```
engine = AutomatedForecasting(freq='1h', horizon='3d', models=['Linear', 'LGB', 'Prophet'], iteration=100, method='bayesian', early_stopping_steps=10)
engine.fit(df_pp)
```
Parameters:
* `freq`: Frequency of time series. `str, default='30min'`
* `horizon`: Forecast horizon. `str, default='1d'`
* `models`: Models included in the search space. `list`
* `iteration`: Number of iterations for the search. `int, default=50`
* `method`: Hyperparameter search method. `{'bayesian', 'randomized_search'}, default='bayesian'`
* `early_stopping_steps`: Number of trials with no improvement in best loss before bayesian optimization stops. `int, default=10`

Attributes:
* `best_model`: Model with the best perfomance and fitted with the input time series


### Available Models:
* Linear Model `'Linear'`
* Prophet Model `'Prophet'`
* LGB Model `'LGB'`
* XGB Model `'XGB'`
* Sarimax Model `'Sarimax'`
* LSTM Model `'LSTM'`
* Hybrid Model `'Hybrid'`


### Workflow
#### 1. Fit preprocessor with training data and Transform training data
The preprocessor is initiated and fitted with training data. The data is then transformed. In this step, the data is resampled to the specified resampling frequency and outliers are removed. 
```
pp = Preprocessor(resample_freq='1h', remove_outlier=True)
df_train_pp = pp.fit(df_train).transform(df_train)
```

#### 2. Train model with preprocessed training data
Using the automated forecaster, hyperparameter tuning is conducted and the best model is fitted with the training data.
```
from forecaster import AutomatedForecasting
engine = AutomatedForecasting(freq='1h', horizon='3d', models=['Linear', 'LGB', 'Prophet'], iteration=100, method='bayesian', early_stopping_steps=10)
engine.fit(df_train_pp)
```

#### 3. Preprocess new data
New data consisting of 7 days of data is transformed using the fitted preprocessor from step 1.
```
df_new_pp = pp.transform(df_new)
```

#### 4. Produce forecast using preprocessed new data
The fitted `engine` takes the processed new data as input and returns the forecast values for the forecast horizon `horizon` and frequency `freq` specified in step 2.
```
df_fcst = engine.predict(df_new_pp)
```

#### 5. Obtain top 5 features
The best model is retrieved from the automated forecaster as an attribute via `engine.best_model`. 
```
best_model = engine.best_model
```
