# linear-regression-global-warming
A linear regression model made from scratch in python with numpy. Uses gradient descent to fit a line through the yearly average earth temperatures to see global warming.

## Data
The most important part of machine learning is the data. I am using monthly average global temperatures since 1850 from a kaggle dataset by Berkeley Earth. The dataset can be found [here](https://www.kaggle.com/berkeleyearth/climate-change-earth-surface-temperature-data)


## Processing the Data
The original CSV `original.csv` contains far more data then I need to train. So `create_csv.py` takes in the original CSV and outputs a CSV file. The edited CSV file has the date starting from 1, which is the the first year, to the end year. The start and end years are included in the top of the CSV file.

The first 100 years are cut out because the data is largely inaccurate because of the technology avaliable at the time and because the data is quite irrelevant because the temperature did not increase. Removing 100 years off data also has the added benefit of making the gradient descent process faster.

The first for loop organizes the data and removes any data not needed. It then moves the temmps into an OrderedDict that is used in the next part.
```python
yearsdata = collections.OrderedDict()
for line in lines[1:]:
    values = line.split(",")
    if(len(values) > 2):
        date = values[0].split("-")
        average_temp = values[1]

        year = date[0]
        month = date[1]
        day = date[2]
        if(int(year) >= go_from_year):
            if not year in yearsdata:
                yearsdata[year] = {}

            yearsdata[year][month] = average_temp
```

The next for loop iterates through the first Dict and creates the final CSV. It takes the monthly temps and creates one yearly average, this is in an effort to speed up the gradient descent by removing un-needed data.
```python
for year in yearsdata.values():
    average_yearly_temp = 0;
    num_months_recorded = 0
    for month in year.values():
        if(len(month)) > 0:
            average_yearly_temp += int(float(month))
            num_months_recorded += 1
    new_csv += str(x_year) + "," + str(average_yearly_temp / num_months_recorded) + "\n"
    x_year += 1
```

## Running Linear Regression using Gradient Descent
Now that all of the data has been processed, we can get to the fun part and run the linear regression. Most of the action happens from the run() function

The first thing to do is initialize a few parameters
```python
learning_rate = 0.0001
num_iterations = 50000
```
Chances are these two values will have to be changed for different data, but these are the best ones currently. The num_iterations could probably be decreased to about 30,000, but 50,000 gives a bit more of an accurate result.

```python
b, m, cost_h = gradient_descent_runner(points, initial_b, initial_m, learning_rate, num_iterations)
```
This function runs gradient descent for num_iterations and will return the optimal b and m values for the set of points. It also returns a list of every cost from iteration 1 - num_iterations.

I should mention here that the gradient descent function updates b twice as fast as m. This is probably not optimal for most cases, but I found that in this case, it makes gradient descent finish faster.
```python
new_b = b_current - ((learningRate * 3) * b_gradient)
new_m = m_current - (learningRate * m_gradient)
```

## Results
After 30 iterations, the b and m values it came up with are:
⋅⋅* B: 7.3673
⋅⋅* M: 0.0085

After plotting the points and line, this is what it looks like:
![alt text](https://github.com/Grocode87/linear-regression-global-warming/blob/master/images/1850-2015-gw.png)

Now, I'm definitely not a scientist, I haven't even graduated high school yet. But the graph above makes me pretty certain that global warming is in fact real.

After graphing the cost function
![alt text](https://github.com/Grocode87/linear-regression-global-warming/blob/master/images/cost_1.png)

You can see that the learning rate could probably be a little higher, but its fine for now.

To see the increase in temperature from 1850 to today more clearly, I can do a little bit of math.
```python
start_temp = (m * 1) + b # Get the temperature in the first year (1850)
end_temp = (m * 166) + b # Get the temperature in the last year (2015) 

temp_increase = end_temp - start_temp # Get the total temperature increase from the first year to the last
yearly_increase = temp_increase / 166 # The total temperature increase by the number of years, to find the yearly temp increase
```
This gives me 0.0085, which means that the yearly temperature increase is 0.0085°C.

## More Results
What we did was cool, but its not completely accurate. In the original chart, you can see that as the years go by, the year temperature difference gets bigger and bigger. A linear model is not perfect for this because it cannot make a curved line, but there is a workaround we can do.

By fitting a line to 50 year time slots, you are potentianlly removing some accuracy because you are using less data, which means more noise. But it will also make a line that is more fit to a certain time period. So, to do this I created another file called `create_csv_timerange.py`. This file is similar to create CSV but has an extra parameter called `go_to_year`. By playing around with go_from year and go_to_year, I created a few data files.

1. data-1850-1900.csv
2. data-1900-1950.csv
3. data-1950-2015.csv


