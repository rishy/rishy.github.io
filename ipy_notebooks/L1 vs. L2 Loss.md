

    import numpy as np
    import pandas as pd
    from sklearn.cross_validation import train_test_split
    from sklearn.ensemble import GradientBoostingRegressor
    from statsmodels.tools.eval_measures import rmse
    import matplotlib.pylab as plt


    # Make pylab inline and set the theme to 'ggplot'
    plt.style.use('ggplot')
    %pylab inline

    Populating the interactive namespace from numpy and matplotlib


    WARNING: pylab import has clobbered these variables: ['plt', 'mod']
    `%pylab --no-import-all` prevents importing * from pylab and numpy



    # Read Boston Housing Data
    data = pd.read_csv('Datasets/Housing.csv')


    # Create a data frame with all the independent features
    data_indep = data.drop('medv', axis = 1)
    
    # Create a target vector(vector of dependent variable, i.e. 'medv')
    data_dep = data['medv']
    
    # Split data into training and test sets
    train_X, test_X, train_y, test_y = train_test_split(data_indep, data_dep,
                                                        test_size = 0.20,
                                                        random_state = 42)

### Regression without any Outliers:


    # Now let's fit a GradientBoostingRegressor with a L1(Least Absolute Deviation) loss function
    # Set a random seed so that we can reproduce the results
    np.random.seed(32767)
    
    # A GradientBoostingRegressor with L1(Least Absolute Deviation) as the loss function
    mod = GradientBoostingRegressor(loss='lad')
    
    fit = mod.fit(train_X, train_y)
    predict = fit.predict(test_X)
    
    # Root Mean Squared Error
    print "MSE -> %f" % rmse(predict, test_y)

    MSE -> 3.440147



    # Suppress printing numpy array in scientific notation
    np.set_printoptions(suppress=True)
    
    error = predict - test_y
    
    # Print squared errors in all test samples 
    np.around(error ** 2, decimals = 2)




    array([   0.03,    0.09,   13.16,    0.08,    0.  ,    1.49,    0.52,
              0.31,    2.2 ,   11.76,    0.39,    0.  ,    0.71,    0.01,
              6.73,   62.99,    2.71,    0.08,   13.18,    3.96,    2.2 ,
             13.17,    0.18,    0.63,    3.91,    4.46,    2.2 ,    1.9 ,
              2.26,    6.1 ,    7.01,    0.24,  111.68,    0.44,   31.79,
              4.04,    0.06,    0.01,   11.08,    0.06,    6.94,    5.05,
             16.06,   12.71,    0.04,    0.  ,    4.9 ,    0.21,   10.28,
             10.61,    5.29,    0.04,    1.43,    6.06,    3.39,    0.11,
              7.18,   20.04,    1.45,    0.09,    4.82,    2.39,    2.19,
              0.05,    0.81,    0.63,    4.83,    4.14,    4.  ,    0.75,
              0.91,    0.65,    0.42,    1.17,    0.63,   12.26,    0.41,
              3.02,    0.89,   51.75,    0.16,   18.5 ,    0.36,    1.43,
              3.31,    8.56,    1.62,    2.8 ,    0.73,    1.59,    0.75,
              2.85,    0.55,    4.29,   48.63,    3.88,  476.98,   59.54,
             12.18,   22.68,    2.86,    0.45])




    # A GradientBoostingRegressor with L2(Least Squares) as the loss function
    mod = GradientBoostingRegressor(loss='ls')
    
    fit = mod.fit(train_X, train_y)
    predict = fit.predict(test_X)
    
    # Root Mean Squared Error
    print "MSE -> %f" % rmse(predict, test_y)

    MSE -> 2.542019



    error = predict - test_y
    
    # Print squared errors in all test samples 
    np.around(error ** 2, decimals = 2)




    array([  0.02,   1.51,  17.29,   1.49,   2.5 ,   5.38,   0.12,   0.03,
             1.03,  18.  ,   2.44,   1.11,   5.32,   0.31,   1.68,  16.66,
             1.54,   1.86,  24.29,   3.47,   1.  ,  14.71,   0.03,   2.27,
             0.61,   2.82,   3.62,   2.67,   5.87,   9.84,  11.36,   0.11,
            39.7 ,   2.7 ,  21.12,   6.05,   2.57,   0.09,  13.48,   0.56,
             1.96,   4.68,  32.54,  11.9 ,   0.  ,   0.24,   6.82,   0.  ,
             3.35,  15.8 ,   1.78,   0.07,   2.35,   2.28,  18.14,   0.04,
             3.54,  14.06,   3.48,   0.17,   0.57,   0.92,   0.7 ,   1.01,
             0.33,   9.48,   1.98,   1.3 ,   7.62,   7.32,   1.62,   1.61,
             0.06,   0.58,   6.42,   1.81,   0.03,   8.83,   0.81,  34.72,
             0.51,  15.03,   0.94,   0.73,   0.89,   4.12,   0.92,   1.91,
             0.39,   0.2 ,   0.7 ,   0.16,   0.1 ,   3.2 ,  25.71,  12.62,
            84.07,  30.38,   5.81,   7.08,   3.31,   2.21])



As apparent from RMSE errors of L1 and L2 loss functions, Least Squares(L2)
outperform L1, when there are no outliers in the data.

### Regression with Outliers:


    # Some statistics about the Housing Data
    data.describe()




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>black</th>
      <th>lstat</th>
      <th>medv</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
      <td>506.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>3.613524</td>
      <td>11.363636</td>
      <td>11.136779</td>
      <td>0.069170</td>
      <td>0.554695</td>
      <td>6.284634</td>
      <td>68.574901</td>
      <td>3.795043</td>
      <td>9.549407</td>
      <td>408.237154</td>
      <td>18.455534</td>
      <td>356.674032</td>
      <td>12.653063</td>
      <td>22.532806</td>
    </tr>
    <tr>
      <th>std</th>
      <td>8.601545</td>
      <td>23.322453</td>
      <td>6.860353</td>
      <td>0.253994</td>
      <td>0.115878</td>
      <td>0.702617</td>
      <td>28.148861</td>
      <td>2.105710</td>
      <td>8.707259</td>
      <td>168.537116</td>
      <td>2.164946</td>
      <td>91.294864</td>
      <td>7.141062</td>
      <td>9.197104</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.006320</td>
      <td>0.000000</td>
      <td>0.460000</td>
      <td>0.000000</td>
      <td>0.385000</td>
      <td>3.561000</td>
      <td>2.900000</td>
      <td>1.129600</td>
      <td>1.000000</td>
      <td>187.000000</td>
      <td>12.600000</td>
      <td>0.320000</td>
      <td>1.730000</td>
      <td>5.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.082045</td>
      <td>0.000000</td>
      <td>5.190000</td>
      <td>0.000000</td>
      <td>0.449000</td>
      <td>5.885500</td>
      <td>45.025000</td>
      <td>2.100175</td>
      <td>4.000000</td>
      <td>279.000000</td>
      <td>17.400000</td>
      <td>375.377500</td>
      <td>6.950000</td>
      <td>17.025000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.256510</td>
      <td>0.000000</td>
      <td>9.690000</td>
      <td>0.000000</td>
      <td>0.538000</td>
      <td>6.208500</td>
      <td>77.500000</td>
      <td>3.207450</td>
      <td>5.000000</td>
      <td>330.000000</td>
      <td>19.050000</td>
      <td>391.440000</td>
      <td>11.360000</td>
      <td>21.200000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>3.677082</td>
      <td>12.500000</td>
      <td>18.100000</td>
      <td>0.000000</td>
      <td>0.624000</td>
      <td>6.623500</td>
      <td>94.075000</td>
      <td>5.188425</td>
      <td>24.000000</td>
      <td>666.000000</td>
      <td>20.200000</td>
      <td>396.225000</td>
      <td>16.955000</td>
      <td>25.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.976200</td>
      <td>100.000000</td>
      <td>27.740000</td>
      <td>1.000000</td>
      <td>0.871000</td>
      <td>8.780000</td>
      <td>100.000000</td>
      <td>12.126500</td>
      <td>24.000000</td>
      <td>711.000000</td>
      <td>22.000000</td>
      <td>396.900000</td>
      <td>37.970000</td>
      <td>50.000000</td>
    </tr>
  </tbody>
</table>
</div>



After looking at the minimum and maximum values of 'medv' column, we can see
that the range of values in 'medv' is [5, 50].<br/>
Let's add a few Outliers in this Dataset, so that we can see some significant
differences with <b>L1</b> and <b>L2</b> loss functions.


    # Get upper and lower bounds[min, max] of all the features
    stats = data.describe()
    extremes = stats.loc[['min', 'max'],:].drop('medv', axis = 1)
    extremes




<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>crim</th>
      <th>zn</th>
      <th>indus</th>
      <th>chas</th>
      <th>nox</th>
      <th>rm</th>
      <th>age</th>
      <th>dis</th>
      <th>rad</th>
      <th>tax</th>
      <th>ptratio</th>
      <th>black</th>
      <th>lstat</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>min</th>
      <td>0.00632</td>
      <td>0</td>
      <td>0.46</td>
      <td>0</td>
      <td>0.385</td>
      <td>3.561</td>
      <td>2.9</td>
      <td>1.1296</td>
      <td>1</td>
      <td>187</td>
      <td>12.6</td>
      <td>0.32</td>
      <td>1.73</td>
    </tr>
    <tr>
      <th>max</th>
      <td>88.97620</td>
      <td>100</td>
      <td>27.74</td>
      <td>1</td>
      <td>0.871</td>
      <td>8.780</td>
      <td>100.0</td>
      <td>12.1265</td>
      <td>24</td>
      <td>711</td>
      <td>22.0</td>
      <td>396.90</td>
      <td>37.97</td>
    </tr>
  </tbody>
</table>
</div>



Now, we are going to generate 5 random samples, such that their values lies in
the [min, max] range of respective features.


    # Set a random seed
    np.random.seed(1234)
    
    # Create 5 random values 
    rands = np.random.rand(5, 1)
    rands




    array([[ 0.19151945],
           [ 0.62210877],
           [ 0.43772774],
           [ 0.78535858],
           [ 0.77997581]])




    # Get the 'min' and 'max' rows as numpy array
    min_array = np.array(extremes.loc[['min'], :])
    max_array = np.array(extremes.loc[['max'], :])
    
    # Find the difference(range) of 'max' and 'min'
    range = max_array - min_array
    range




    array([[  88.96988,  100.     ,   27.28   ,    1.     ,    0.486  ,
               5.219  ,   97.1    ,   10.9969 ,   23.     ,  524.     ,
               9.4    ,  396.58   ,   36.24   ]])




    # Generate 5 samples with 'rands' value
    outliers_X = (rands * range) + min_array
    outliers_X




    array([[  17.04578252,   19.15194504,    5.68465061,    0.19151945,
               0.47807845,    4.56054001,   21.49653863,    3.23572024,
               5.40494736,  287.356192  ,   14.40028283,   76.27278363,
               8.67066488],
           [  55.35526271,   62.2108771 ,   17.43112727,    0.62210877,
               0.68734486,    6.80778568,   63.30676167,    7.97086794,
              15.30850173,  512.98499602,   18.44782245,  247.03589642,
              24.27522186],
           [  38.95090441,   43.7727739 ,   12.40121272,    0.43772774,
               0.59773568,    5.84550107,   45.40336346,    5.94324817,
              11.067738  ,  416.36933524,   16.71464075,  173.91406674,
              17.59325326],
           [  69.87957895,   78.53585837,   21.88458216,    0.78535858,
               0.76668427,    7.65978645,   79.15831848,    9.76610981,
              19.06324743,  598.52789787,   19.98237069,  311.77750713,
              30.19139507],
           [  69.40067405,   77.99758081,   21.73774005,    0.77997581,
               0.76406824,    7.63169374,   78.63565097,    9.70691596,
              18.93944359,  595.70732345,   19.9317726 ,  309.64280598,
              29.99632329]])




    # We will also create some hard coded outliers for 'medv', i.e. our target
    medv_outliers = np.array([0, 0, 600, 700, 600])


    # Let's have a look at the data types of all the columns
    # so that we can create our new Dataset compatible with the original one
    data_indep.dtypes




    crim       float64
    zn         float64
    indus      float64
    chas         int64
    nox        float64
    rm         float64
    age        float64
    dis        float64
    rad          int64
    tax          int64
    ptratio    float64
    black      float64
    lstat      float64
    dtype: object




    # Change the type of 'chas', 'rad' and 'tax' to rounded of Integers
    outliers_X[:, [3, 8, 9]] = np.int64(np.round(outliers_X[:, [3, 8, 9]]))


    # Finally concatenate our existing 'train_X' and 'train_y' with these outliers
    train_X = np.append(train_X, outliers_X, axis = 0)
    train_y = np.append(train_y, medv_outliers, axis = 0)


    # Plot a histogram of 'medv' in train_y
    fig = plt.figure(figsize=(13,7))
    plt.hist(train_y, bins=50, range = (-10, 800))
    fig.suptitle('medv Count', fontsize = 20)
    plt.xlabel('medv', fontsize = 16)
    plt.ylabel('count', fontsize = 16)




    <matplotlib.text.Text at 0x7f906589cb10>




![png](L1%20vs.%20L2%20Loss_files/L1%20vs.%20L2%20Loss_22_1.png)


You can see there are some clear outliers at 600, 700 and even one or two 'medv'
values are 0.<br/>
Since, our outliers are in place now, we will once again fit the
GradientBoostingRegressor with L1 and L2 Loss function to see the contrast in
their performances with outliers.


    # So let's fit a GradientBoostingRegressor with a L1(Least Absolute Deviation) loss function
    np.random.seed(9876)
    
    # A GradientBoostingRegressor with L1(Least Absolute Deviation) as the loss function
    mod = GradientBoostingRegressor(loss='lad')
    
    fit = mod.fit(train_X, train_y)
    predict = fit.predict(test_X)
    
    # Root Mean Squared Error
    print "MSE -> %f" % rmse(predict, test_y)

    MSE -> 7.055568



    error = predict - test_y
    
    # Print squared errors in all test samples 
    np.around(error ** 2, decimals = 2)




    array([    0.04,     0.69,    17.13,     0.01,     0.62,     0.01,
               1.88,     0.17,     0.54,    16.12,     2.4 ,     0.23,
               0.04,     0.  ,     1.62,    83.27,     1.58,     0.23,
              17.81,     1.3 ,     2.77,    12.79,     0.07,     0.45,
               1.59,     7.27,     1.3 ,     2.45,     5.99,     7.88,
               9.02,     0.49,  3396.2 ,     0.01,    28.32,    12.75,
               0.45,     0.  ,    15.67,     0.  ,     4.2 ,     3.72,
              30.43,     5.71,     0.01,     0.18,     7.47,     0.09,
               1.28,    12.06,     2.49,     0.13,     6.22,     2.76,
               2.29,     0.02,     4.91,    16.74,     1.94,     0.27,
               0.25,   363.92,     2.42,     0.63,     0.58,     4.43,
               4.28,     3.15,     1.17,     0.97,     0.73,     1.43,
               0.53,     0.49,     0.81,    16.5 ,     0.07,    14.49,
               1.84,    38.94,     0.17,    16.95,     0.45,     1.42,
               1.97,     6.45,     1.08,     3.42,     0.36,     0.46,
               1.1 ,     3.07,     0.12,     2.18,    46.74,     1.48,
             652.32,    66.97,     9.32,    46.2 ,     3.05,     0.66])




    # A GradientBoostingRegressor with L2(Least Squares) as the loss function
    mod = GradientBoostingRegressor(loss='ls')
    
    fit = mod.fit(train_X, train_y)
    predict = fit.predict(test_X)
    
    # Root Mean Squared Error
    print "MSE -> %f" % rmse(predict, test_y)

    MSE -> 9.806251



    error = predict - test_y
    
    # Print squared errors in all test samples 
    np.around(error ** 2, decimals = 2)




    array([    0.  ,     6.18,    11.  ,     0.36,     0.45,     1.62,
               0.28,     0.02,     0.59,    22.96,     2.24,     0.06,
               0.98,     0.02,     0.64,     6.52,     1.72,     0.04,
               7.4 ,     3.76,     0.76,    15.28,     0.38,     1.95,
               3.7 ,     9.  ,     4.28,     6.69,     5.83,    10.83,
              17.79,     0.57,   121.92,     2.3 ,    15.77,     4.94,
               0.02,     0.6 ,    13.59,     0.03,    11.32,     1.51,
              16.45,     6.85,     0.07,     0.21,     6.32,     0.12,
               3.41,    12.18,     7.96,     2.12,     2.95,     9.69,
              17.15,     0.4 ,     2.43,     4.46,     1.58,     1.59,
               2.45,     0.66,     2.56,     3.33,     0.43,    13.59,
               6.76,     1.15,     2.87,     3.3 ,     4.7 ,     0.01,
               1.3 ,     0.02,     0.86,  8597.78,     0.07,     1.7 ,
               0.  ,    33.7 ,     0.71,    10.75,     0.03,     0.63,
               1.55,    13.66,     0.89,     2.61,     0.27,     0.05,
               1.12,     0.08,     0.  ,     3.76,    63.48,    10.78,
             475.58,   111.95,    12.11,     5.93,     1.53,     1.99])



With outliers in the dataset, a L2(Loss function) tries to adjust the
coefficients features according to these outliers on the expense of other
features, since the squared-error is going to be huge for these outliers(for
error > 1). On the other hand L1(Least absolute deviation) is quite robust to
outliers.<br/>
As a result, L2 loss function may result in huge deviations in some of the
samples which results in reduced accuracy.
