# linear-regression-global-warming
A linear regression model made from scratch in python with numpy. Uses gradient descent to fit a line through the yearly average earth temperatures to see the effect of global warming.

# linear-regression-global-warming
A linear regression model made from scratch in python with numpy. Uses gradient descent to fit a line through the yearly average earth temperatures to see the effect of global warming.

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
TODO
