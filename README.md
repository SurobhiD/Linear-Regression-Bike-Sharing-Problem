# Solving Bike Sharing Problem using Linear Regression

## **Problem Statement :** 

A bike-sharing system is a service in which bikes are made available for shared use to individuals on a short term basis for a price or free. Many bike share systems allow people to borrow a bike from a "dock" which is usually computer-controlled wherein the user enters the payment information, and the system unlocks it. This bike can then be returned to another dock belonging to the same system.
 ![alt text](https://storage.googleapis.com/gweb-uniblog-publish-prod/original_images/image1_hH9B4gs.jpg)
A US bike-sharing provider BoomBikes has recently suffered considerable dips in their revenues due to the ongoing Corona pandemic. The company is finding it very difficult to sustain in the current market scenario. So, it has decided to come up with a mindful business plan to be able to accelerate its revenue as soon as the ongoing lockdown comes to an end, and the economy restores to a healthy state. 

 ![alt text](https://images.hindustantimes.com/rf/image_size_960x540/HT/p2/2020/12/08/Pictures/ht-pune_98af0850-3943-11eb-87ed-5b70cc8f4a19.jpg)

In such an attempt, BoomBikes aspires to understand the demand for shared bikes among the people after this ongoing quarantine situation ends across the nation due to Covid-19. They have planned this to prepare themselves to cater to the people's needs once the situation gets better all around and stand out from other service providers and make huge profits.

They have contracted a consulting company to understand the factors on which the demand for these shared bikes depends. Specifically, they want to understand the factors affecting the demand for these shared bikes in the American market. The company wants to know:
1. Which variables are significant in predicting the demand for shared bikes.
2. How well those variables describe the bike demands
Based on various meteorological surveys and people's styles, the service provider firm has gathered a large dataset on daily bike demands across the American market based on some factors. 

### **Python Libraries Used** 

Here is a list of Python Libraries used along with their version numbers :

 |Library                         |   Version                         |
|-------------------------------|-----------------------------|
|   `statsmodels`           |    0.11.0         |
|`Numpy`            |1.18.1           |
|`pandas`|1.0.1|
|`matplotlib`|3.1.3|
|`seaborn`|0.10.0|
|`sklearn`|0.3.6|

### **Implementation in Python Notebook** 
### **Understanding the Input Data** 

The dataset contain 730 rows and 16 columns. Here are the top few rows along with the list of features :
![bike1](https://user-images.githubusercontent.com/10894854/124223542-89f6f900-db21-11eb-94fc-869c89508671.JPG)
In the dataset  there are three columns named 'casual', 'registered', and 'cnt'. The variable 'casual' indicates the number casual users who have made a rental. The variable 'registered' on the other hand shows the total number of registered users who have made a booking on a given day. Finally, the 'cnt' variable indicates the total number of bike rentals, including both casual and registered. The model should be built taking this 'cnt' as the target variable.

Here are few of the inferences :
- There are no null values in the dataset.
- "instant" column : Since this is just a serial number column and doesn't add any meaning in the analysis, we can drop this.
-  "dteday" column : Since all necessary information ( year, month, season, weekday, working day) has already been extracted from this column we can drop this column
-  "cnt" column : This is our target variable ( y) . Since the total count of users is a sum of the registered users and the casual users, the latter two columns can be removed from the dataset
- There are no arbitrary values in all the categorical variables and the data is pretty clean. However, the non-binary categorical variables will need to have dummy encoding in them so that we can apply Linear Regression on them. These variables that need to be converted to dummies are : Season, Month, Weekday, weathersit

### **EDA : Visualising the Data** 
A boxplot for some of the categorical variables.
![bike2](https://user-images.githubusercontent.com/10894854/124223878-39cc6680-db22-11eb-9125-780804594168.JPG)

1.  Season 3 ( Fall season) has more count and next is Season 2 ( Summer). Meaning people prefer renting bikes in fall and summer seasons compared to Spring and winter. The count is least in Spring season.
2.  More bikes were rented in 2019 that in 2018.
3.  Months - May to October has more bike rentals than the other months which also coincides with the point 1 made above.
4.  Being a holiday has less count of rental means and being a working day has a higher count. This suggests that many rentals are happening for office commute.
5.  It clearly shows that on a clear day the rentals are higher and on a snowy/rainy day the rentals are lower.
6.  There is not much difference in the rental count means on any day of teh week. However we do notice a higher range of rentals on a Friday and Monday.
7.  We don't see any outliers in the variables.

Let's understand the correlation between the variables:
![bike3](https://user-images.githubusercontent.com/10894854/124223980-713b1300-db22-11eb-8a80-1adcdeafe05e.JPG)

We can see from the above Correlation map that the columns 'temp' and 'atemp' are highly correlated. So we will go ahead and drop 'atemp' column since the correlation with other columns for this variable is slightly higher.

Other correlated variables seem to be

- Count and Yr ( 0.57)
- hum and weatherit (0.59)
- Season and mnth(0.83)
- Windspeed is negatively correlated to hum and temp

![bike4](https://user-images.githubusercontent.com/10894854/124224120-ac3d4680-db22-11eb-8b49-bf43dad6dc12.JPG)
Inference :
There seems to be a higher correlation between temp and cnct

### **Data Preparation** 
We will be doing the following :
1. Convert the categorical variables into dummy variables. 
2. Split the dataset into test and train data in 30:70 ratio
3. Perform scaling of features  using MinMaxScaling
4. Divide the data into X and Y

### **Model Building** 
We will be doing the following :

1. We will be using RFE to select the most important features from the entire dataset
2. We will checks the VIF and p-values to determine if we should retain a feature or not
3. We will do a check of residuals for the final model

This is the final model after several rounds of evaluations.
![bike5](https://user-images.githubusercontent.com/10894854/124227311-13112e80-db28-11eb-9e0a-cfca9f4c21d7.JPG)

- Here we can see a good R2 and Adj. R2 value.
- Also all the p- values and < 0.05. So all the variables in this model are significant
- The Rental counts are inversely proportional to the Humidity and the Windspeed. Also, compared to clear weather, Light snow or rain and misty weather negatively affect the count of the rentals.
- Top 3 features contributing significantly are :
1. Temperature ( Higher temperature , higher demand)
2. Light snow and rain ( Less snow and rain , lesser demand)
3. Year ( 2019 having more demand over 2018)
- The Durbin-Watson metric is between the ideal range of 0-4 which suggests no ACF in the residuals

### **Residual Analysis** 
We will be doing the following :

1. Creating a distribution plot of the residuals to check for normality
2. Create a scatter plot with y-predicted to check if there are any correlations

![bike6](https://user-images.githubusercontent.com/10894854/124227540-7bf8a680-db28-11eb-9012-f1d58b2df4b8.JPG)
We can very well see that the residuals are normally distributed with mean zero.

![bike7](https://user-images.githubusercontent.com/10894854/124227544-7d29d380-db28-11eb-912e-72e9f7872c8d.JPG)
We can see that the residuals are randomly spready across the plot with no specific relation with the Y-Pred value on the X-axis.
Hence this proves that:
- The residuals are homoskedastic
- The residuals are not correlated with the predicted values

### **Making Predictions** 
![bike8](https://user-images.githubusercontent.com/10894854/124227764-cda13100-db28-11eb-8c97-1ed69ccbbd4e.JPG)

RMSE = 0.096
R-squared on the test set = 80.6%
The R-Squared of the test set is slightly lower than the training set ( by 4% ) but still the model is able to explain a good amount of variance of Y in the test set.
