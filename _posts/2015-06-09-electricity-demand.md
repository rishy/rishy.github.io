---
layout:     post
title:      Electricity Demand Analysis and Appliance Detection
date:       2015-06-10 16:52:00
summary:    Analysis of Electricity demand from a house on a time-series dataset. An appliance detection systems is also created using K-Means Clustering based on the electricity demand.
categories: projects
comments: true
---

In this post, we are going to analyze electricity consumption data from a house. We have a  time-series dataset which contains the power(kWh), Cost of electricity and Voltage at a particular time stamp. We are further provided with the temperature records during the same day for each hour. You can download the compressed dataset from [here](https://github.com/rishy/electricity-demand-analysis/blob/master/data-science.gz). I'd further recommend you to have a look at the corresponding [ipython notebook](https://github.com/rishy/electricity-demand-analysis/blob/master/Electricity%20Demand.ipynb). 

First part is the Data Analysis Part where we will be doing the basic data cleaning and analysis regarding the power demand and cost incurred. The second part employs a KMeans clustering approach to identify which appliance might be the major cause for the power demand in a particular hour of the day.

So let's start with basic imports and reading of data from the given dataset.

{% highlight python linenos%}
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# Read the sensor dataset into pandas dataframe
sensor_data = pd.read_csv('merged-sensor-files.csv',
                          names=["MTU", "Time", "Power",
                          "Cost", "Voltage"], header = False)

# Read the weather data in pandas series object
weather_data = pd.read_json('weather.json', typ ='series')


# A quick look at the datasets
sensor_data.head(5)
{% endhighlight %}


<div style="max-height:1000px;max-width:1500px;overflow:auto;">
<table>
  <thead>
    <tr>
      <th></th>
      <th>MTU</th>
      <th>Time</th>
      <th>Power</th>
      <th>Cost</th>
      <th>Voltage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>MTU1</td>
      <td>05/11/2015 19:59:06</td>
      <td>4.102</td>
      <td>0.62</td>
      <td>122.4</td>
    </tr>
    <tr>
      <th>1</th>
      <td>MTU1</td>
      <td>05/11/2015 19:59:05</td>
      <td>4.089</td>
      <td>0.62</td>
      <td>122.3</td>
    </tr>
    <tr>
      <th>2</th>
      <td>MTU1</td>
      <td>05/11/2015 19:59:04</td>
      <td>4.089</td>
      <td>0.62</td>
      <td>122.3</td>
    </tr>
    <tr>
      <th>3</th>
      <td>MTU1</td>
      <td>05/11/2015 19:59:06</td>
      <td>4.089</td>
      <td>0.62</td>
      <td>122.3</td>
    </tr>
    <tr>
      <th>4</th>
      <td>MTU1</td>
      <td>05/11/2015 19:59:04</td>
      <td>4.097</td>
      <td>0.62</td>
      <td>122.4</td>
    </tr>
  </tbody>
</table>
</div>
<br>
Let's have a quick look at the weather dataset as well:

    weather_data

<pre>
2015-05-12 00:00:00    75.4
2015-05-12 01:00:00    73.2
2015-05-12 02:00:00    72.1
2015-05-12 03:00:00    71.0
2015-05-12 04:00:00    70.7
.
.
dtype: float64
</pre>



## TASK 1: Data Analysis

### Data Cleaning/Munging:
After having a look at the <b>merged-sensor-files.csv</b> I found out there are some inconsistent rows where header names are repeated and as a result 'pandas' is converting all these columns to 'object' type. This is quite a common problem, which arises while merging multiple csv files into a single file. 


    sensor_data.dtypes
<pre>
MTU        object
Time       object
Power      object
Cost       object
Voltage    object
dtype: object
</pre>

Let's find out and remove these inconsistent rows so that all the columns can be converted to appropriate data types. 

The code below finds all the rows where "Power" column has a string value - "Power" and get the index of these rows. 

{% highlight python linenos%}
# Get the inconsistent rows indexes
faulty_row_idx = sensor_data[sensor_data["Power"] == " Power"].index.tolist()
faulty_row_idx
{% endhighlight %}

<pre>
[3784,
 7582,
 11385,
 .
 .
 81617,
 85327]
</pre> 

and now we can drop these rows from the dataframe

{% highlight python linenos%}
# Drop these rows from sensor_data dataframe
sensor_data.drop(faulty_row_idx, inplace=True)

# This should return an empty list now
sensor_data[sensor_data["Power"] == " Power"].index.tolist()
{% endhighlight %}
<pre>
  []
</pre>

<p>We have cleaned up the sensor_data and now all the columns can be converted to more appropriate data types.</p>

{% highlight python linenos %}
# Type Conversion
sensor_data[["Power", "Cost", "Voltage"]] = sensor_data[["Power",
                                "Cost", "Voltage"]].astype(float)

sensor_data[["Time"]] = pd.to_datetime(sensor_data["Time"])

# Also add an 'Hour' column in sensor_data
sensor_data['Hour'] = pd.DatetimeIndex(sensor_data["Time"]).hour

sensor_data.dtypes
{%endhighlight%}

<pre>
MTU                object
Time       datetime64[ns]
Power             float64
Cost              float64
Voltage           float64
Hour                int32
dtype: object
</pre>

<p>
This is better now. We have got clearly defined datatypes of different columns now. Next step is to convert the weather_data Series to a dataframe so that we can work with it with more ease.
</p>

{% highlight python linenos%}
# Create a dataframe out of weather dataset as well
temperature_data = weather_data.to_frame()

# Reindex it so as to create a two column dataframe
temperature_data.reset_index(level=0, inplace=True)
temperature_data.columns = ["Time", "Temperature"]

# Add the "Hour" column in temperature_data
temperature_data["Hour"] = pd.DatetimeIndex(
                            temperature_data["Time"]).hour

temperature_data.dtypes
{% endhighlight %}

<pre>
  Time           datetime64[ns]
  Temperature           float64
  Hour                    int32
  dtype: object
</pre>

<p>
Since now we have both of our dataframes in place, it'd be a good point to have a look at sum of the basic statistics of both of these data frames.
</p>

{% highlight python linenos %}
sensor_data.describe()
{%endhighlight%}

<div>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Power</th>
      <th>Cost</th>
      <th>Voltage</th>
      <th>Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>88891.000000</td>
      <td>88891.000000</td>
      <td>88891.000000</td>
      <td>88891.000000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>1.315980</td>
      <td>0.202427</td>
      <td>123.127744</td>
      <td>11.531865</td>
    </tr>
    <tr>
      <th>std</th>
      <td>1.682181</td>
      <td>0.252357</td>
      <td>0.838768</td>
      <td>6.921671</td>
    </tr>
    <tr>
      <th>min</th>
      <td>0.113000</td>
      <td>0.020000</td>
      <td>121.000000</td>
      <td>0.000000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>0.255000</td>
      <td>0.040000</td>
      <td>122.600000</td>
      <td>6.000000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>0.367000</td>
      <td>0.060000</td>
      <td>123.100000</td>
      <td>12.000000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>1.765000</td>
      <td>0.270000</td>
      <td>123.700000</td>
      <td>18.000000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>6.547000</td>
      <td>0.990000</td>
      <td>125.600000</td>
      <td>23.000000</td>
    </tr>
  </tbody>
</table>
</div>

<br>
{% highlight python linenos %}
temperature_data.describe()
{%endhighlight%}

<div>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Temperature</th>
      <th>Hour</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>count</th>
      <td>25.000000</td>
      <td>25.00000</td>
    </tr>
    <tr>
      <th>mean</th>
      <td>76.272000</td>
      <td>11.04000</td>
    </tr>
    <tr>
      <th>std</th>
      <td>6.635355</td>
      <td>7.29429</td>
    </tr>
    <tr>
      <th>min</th>
      <td>67.900000</td>
      <td>0.00000</td>
    </tr>
    <tr>
      <th>25%</th>
      <td>69.600000</td>
      <td>5.00000</td>
    </tr>
    <tr>
      <th>50%</th>
      <td>75.400000</td>
      <td>11.00000</td>
    </tr>
    <tr>
      <th>75%</th>
      <td>83.000000</td>
      <td>17.00000</td>
    </tr>
    <tr>
      <th>max</th>
      <td>87.000000</td>
      <td>23.00000</td>
    </tr>
  </tbody>
</table>
</div>

<br>
As apparent from above statistics there is a good amount of variation in <b>Power</b> and corresponding <b>Cost</b> values in <b>sensor\_data</b> dataframe, where average power is <b>1.315980kW</b> and minimum and maximum power used throughout the day is <b>0.11kW</b> and <b>6.54kW</b> respectively. Similarily there is an apparent variation in temperature in <b>temperature\_data</b> dataset, most probably it attributes to day and night time. <br><br>
To get a better understanding of these variations we'll be plotting power and temperatures with the timestamps, so as to find out the peak times for both.<br>
But before moving to visualizations we'll have to create respective grouped datasets from <b>sensor\_data</b> and <b>temperature\_data</b>, grouping by the "Hour" column. This way we can work on hourly basis.

{% highlight python linenos%}
# Group sensor_data by 'Hour' Column
grouped_sensor_data = sensor_data.groupby(
                        ["Hour"], as_index = False).mean()
grouped_sensor_data
{% endhighlight %}


<div>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Hour</th>
      <th>Power</th>
      <th>Cost</th>
      <th>Voltage</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>0.173790</td>
      <td>0.029468</td>
      <td>124.723879</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>0.179594</td>
      <td>0.033805</td>
      <td>124.522469</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2</td>
      <td>0.185763</td>
      <td>0.037013</td>
      <td>123.929979</td>
    </tr>
    <tr>
      <th>.</th>
      <td>.</td>
      <td>.</td>
      <td>.</td>
      <td>.</td>
    </tr>    
    <tr>
      <th>22</th>
      <td>22</td>
      <td>2.542672</td>
      <td>0.387109</td>
      <td>123.542620</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>2.269941</td>
      <td>0.346457</td>
      <td>123.415791</td>
    </tr>
  </tbody>
</table>
</div>

{% highlight python linenos %}
# Group temperature_data by "Hour"
grouped_temperature_data = temperature_data.groupby(
                            ["Hour"], as_index = False).mean()
grouped_temperature_data
{%endhighlight%}

<div>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Hour</th>
      <th>Temperature</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0</td>
      <td>78.25</td>
    </tr>
    <tr>
      <th>1</th>
      <td>1</td>
      <td>73.20</td>
    </tr>
    <tr>
      <th>.</th>
      <td>.</td>
      <td>.</td>
    </tr>
    <tr>
      <th>22</th>
      <td>22</td>
      <td>84.40</td>
    </tr>
    <tr>
      <th>23</th>
      <td>23</td>
      <td>83.00</td>
    </tr>
  </tbody>
</table>
</div>


## Basic Visualizations:

{% highlight python linenos%}
# Generates all the visualizations right inside the ipython notebook
%pylab inline
plt.style.use('ggplot')

fig = plt.figure(figsize=(13,7))
plt.hist(sensor_data.Power, bins=50)
fig.suptitle('Power Histogram', fontsize = 20)
plt.xlabel('Power', fontsize = 16)
plt.ylabel('Count', fontsize = 16)
{% endhighlight %}

![power-histogram](../../../../../images/power-histogram.png)

Looks like most of the time this house is consuming a limited amount of power. Although there is also a good amount of distribution in the range of <b>3.5kW - 5kW</b>, indicating a higher demand.<br/>
Let's now plot the Power Distribution with the day hours.

{% highlight python linenos%}
fig = plt.figure(figsize=(13,7))
plt.bar(grouped_sensor_data.Hour, grouped_sensor_data.Power)
fig.suptitle('Power Distribution with Hours', fontsize = 20)
plt.xlabel('Hour', fontsize = 16)
plt.ylabel('Power', fontsize = 16)
plt.xticks(range(0, 24))
plt.show()
{% endhighlight %}

![power-time-distribution](../../../../../images/power-time-distribution.png)


#### Some of the inferences we can get from this bar chart are:

* Highest Demand is noticed during the evening hours. This is quite expected since most of the equipments would be in 'on' state during this time like AC(during summers), room heaters(during winters), TV, Oven, Washing Machine, Lights, etc.
* Night hours(0000 - 0500) and office hours(0900 - 1600) have very low demand, since most of the appliances will be in 'off' state during this period.
* There is a slight increase in Power during morning hours from 0600 - 0900, which should account for the power used by the appliances during morning activities, lights, geysers, etc.

#### Steady States:

* In the time period <b>0000 - 0500</b>, demand is noticeably less and ranges between <b>0.17kW - 0.18kW</b>
* Another steady period is from <b>1000 - 1500</b>, demand is pretty much steady between <b>0.373kW - 0.376kW</b>
* Steady state with highest demand is from <b>1600 - 1900</b> having a range between <b>4.36kW - 4.25kW</b>

Some sudden changes in Demand during 0700 and 1800 can be attributed because of random events or the usage of certain appliances and may be counted as noise in the dataset. 

Similarily there is a slight oscillation in demand during 0900 which suddenly falls down from 0.38kW to 0.16kW and rises up again to about 0.37kW. Similar change in demand is seen at 2100.

Let's further plot temperature with the Power to see if there is any correlation among these.

{% highlight python linenos%}
fig = plt.figure(figsize=(13,7))
plt.bar(grouped_temperature_data.Temperature,
                    grouped_sensor_data.Power)
fig.suptitle('Power Distribution with Temperature', fontsize = 20)
plt.xlabel('Temperature in Fahrenheit', fontsize = 16)
plt.ylabel('Power', fontsize = 16)
plt.show()
{% endhighlight %}

![power-temp-distribution](../../../../../images/power-temp-distribution.png)

There seems to be a direct correlation between temperature and the demand of power. This makes sense, since with our current dataset which is from May, this shows that cooling appliances like AC, refrigerator, etc. are consuming a lot of power during the peak hours(evening).

## Task 2: Machine Learning

We'll start with merging the <b>grouped\_sensor\_data</b> and <b>grouped\_temperature\_data</b> so that we can work on the complete dataset from a single dataframe.

{% highlight python linenos%}
# Merge grouped_sensor_data and grouped_temperature_data
# using "Hour" as the key
merged_data = grouped_sensor_data.merge(grouped_temperature_data)
{% endhighlight %}

In previous visualization we saw that when temperature is low generally there is less demand of power. But that mainly relates to the cooling appliances in the home. We'll consider the following appliances:

* Cooling Systems
* TV
* Geyser
* Lights
* Oven
* Home Security Systems

and would try to identify there presence or on/off state using the merged dataset.<br>

#### AC, Refrigerator and Other Coooling Systems:
As apparent from "Power Distribution with Temperature" figure, there is a sudden increase in power demand with the rise in temperature. This clearly indicates the <b>ON</b> state of one or more cooling systems in the home. Since these appliances takes a considerable amount of power, this sudden upsurge in the power is quite justified. Clearly <b>Power</b> and <b>Temperature</b> are the two features that indicates the 'ON' state of these appliances. Although 'Cost' feature is also correlated with 'Power' we'd leave it out, since it is more of a causation of Power demand, then a completely independent feature.

#### TV:
During the evening hours(1600 - 2300), an 'ON' television set is probably another factor for increased power demand. It is quite apparent from the <b>Power</b> feature.

#### Geyser, Oven:
Slight increase in power demand during morning hours can be related to the presence of these appliances and is justified again by the <b>Power</b> feature.

#### Lights:
It's quite obvious there is a small contribution(considering house owner was smart and installed LED bulbs ;) ) of lights in the house in the 'Power' demand. And of course it only makes sense to switch 'ON' the lights during *darker times* :D of the day, <b>Hour</b> and Low <b>Power</b> are the indicators of lights.

#### Home Security Systems:
During the office hours there's a very little increase in the Power demand, this can be attributed to home security systems or other automated devices.<br>

Now, we'll be using simple <b>K-Means clustering</b> using <b>scikit-learn</b>. We are going to consider <b>Hour, Power and Temperature</b> feature from the original dataset. For that first of all we'll have to merge the sensor\_data dataframe with grouped\_temperature\_data dataframe.

{% highlight python linenos %}
# Complete merged dataset
data =sensor_data.merge(grouped_temperature_data)

# Lets drop Time, MTU, Cost and Voltage features
data.drop(["Time", "MTU", "Cost", "Voltage"], axis = 1,
                                        inplace = True)

# Import required modules from scikit-learn
from sklearn.cluster import KMeans
from sklearn.cross_validation import train_test_split


# Set a random seed, so we can reproduce the results
np.random.seed(1234)

# Divide the merged dataset into train and test datasets
train_data, test_data = train_test_split(data, test_size = 0.25,
                                              random_state = 42)

# Perform K-Means clustering over the train dataset
kmeans = KMeans(n_clusters = 4, n_jobs = 4)
kmeans_fit = kmeans.fit(train_data) 

predict = kmeans_fit.predict(test_data)

test_data["Cluster"] = predict
{%endhighlight%}

<div>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Power</th>
      <th>Hour</th>
      <th>Temperature</th>
      <th>Cluster</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>52595</th>
      <td>0.114</td>
      <td>8</td>
      <td>69.2</td>
      <td>1</td>
    </tr>
    <tr>
      <th>.</th>
      <td>.</td>
      <td>.</td>
      <td>.</td>
      <td>.</td>
    </tr>
    <tr>
      <th>7834</th>
      <td>1.094</td>
      <td>21</td>
      <td>84.2</td>
      <td>0</td>
    </tr>
    <tr>
      <th>25231</th>
      <td>0.125</td>
      <td>1</td>
      <td>73.2</td>
      <td>2</td>
    </tr>
  </tbody>
</table>
</div>


This looks like a pretty reasonable clustering. We can further assign the labels to these clusters, as an appliance detection model. As apparent from the predicted result, We can set the labels for clusters as:

* <b>0</b> - Cooling Systems
* <b>1</b> - Oven, Geyser
* <b>2</b> - Night Lights
* <b>3</b> - Home Security Systems

We'll create a data frame with these labels and merge it with predicted results.

{% highlight python linenos %}
# Create a dataframe with appliance labels
label_df = pd.DataFrame({"Cluster": [0, 1, 2, 3],
                         "Appliances": ["Cooling System",
                                        "Oven, Geyser",
                                        "Night Lights",
                                        "Home Security Systems"]})

# Merge predicted cluster values for test data set
# with our label dataframe
result = test_data.merge(label_df)
result.head(1)
{%endhighlight%}
<br>

<div>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Power</th>
      <th>Hour</th>
      <th>Temperature</th>
      <th>Cluster</th>
      <th>Appliances</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>0.114</td>
      <td>8</td>
      <td>69.2</td>
      <td>1</td>
      <td>Oven, Geyser</td>
    </tr>
  </tbody>
</table>
</div>



{% highlight python linenos %}
result.tail(1)
{%endhighlight%}


<div>
<table>
  <thead>
    <tr>
      <th></th>
      <th>Power</th>
      <th>Hour</th>
      <th>Temperature</th>
      <th>Cluster</th>
      <th>Appliances</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>22218</th>
      <td>0.306</td>
      <td>15</td>
      <td>80.7</td>
      <td>3</td>
      <td>Home Security Systems</td>
    </tr>
  </tbody>
</table>
</div>


I think this makes sense. As apparent from *result* dataframe, in hours like 8, 9, 10 there is a high possibility that a Oven or Geyser is being used. On the other hand during office hours(1000 - 1600), most probably Home Security Appliances are taking the power.<br>

Starting from the very beginning, i.e. the Data Analysis process, I think with more data we could group it according to the days(for a week's or month's data), or by months(for a year's data). That could've significantly changed the predicted Power values, since the average values over these larger intervals would be smoother.<br>

We'd also have to take care of the seasons and temperature, since different appliances would be taking power in different seasons, so clustering would turn into a bit complicated task compared to what we did with data of just one day.

The most important data that could help in a more accurate analysis would be the power consumption amount of all the appliances in the house. That way it'd be much easier to understand what appliance is taking more power in a certain period of time. 

Furthermore, this would also help during the classification task, since we would already know that certain appliances requires much power, hence we could more accurately classify a sample.<br>

One limitation is the number of features we have in this dataset, to learn new features a simple neural net could also be employed to get some hidden patterns here. 

You can further look at the Github repo with the above code at: [rishy/electricity-demand-analysis](https://github.com/rishy/electricity-demand-analysis). Your feedbacks and comments are always welcomed. 

Related Papers:

* [http://www.sciencedirect.com/science/article/pii/S037877881200151X](http://www.sciencedirect.com/science/article/pii/S037877881200151X)
* [http://cs.gmu.edu/~jessica/publications/astronomy11.pdf](http://cs.gmu.edu/~jessica/publications/astronomy11.pdf)

{% include disqus.html %}