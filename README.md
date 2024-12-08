# REGRESSION-MODEL-FOR-METRO-INTERSTATE-TRAFFIC-VOLUME

The purpose of this project is to construct a predictive model using various machine learning algorithms to predict traffic volume.
Metro Interstate Traffic Volume data set contains Hourly weather features and holidays included for impacts on traffic volume

# The dataset contains:

Traffic volume: Measured hourly.
Weather attributes: Temperature, rain, snow, cloud cover.
Temporal attributes: Date, time, holidays, and weekdays.

Initially we have taken Metro_Interstate_traffic data which inclides the columns along with data types:

0 holiday 61 non-null object
1 temp 48204 non-null float64
2 rain_1h 48204 non-null float64
3 snow_1h 48204 non-null float64
4 clouds_all 48204 non-null int64  
 5 weather_main 48204 non-null object
6 weather_description 48204 non-null object
7 date_time 48204 non-null object
8 traffic_volume 48204 non-null int64

# Analyzing the numerical columns rain_1h , snow_1h, temp, cloud_all

1. rain_1hr: the avg value of the rain in 1 hr is 0.334264 and the highest value is 9831.300000
2. snow_1hr: the avg value of the snow in 1 hr is around 0.000222 and the max is 0.5
3. temp: the avg value of the temperature is around 281.2 and max is 300
4. cloud_all: the avg value of the cloud cover is around 49.3 and max is 100

The outliers usually in this case are the max values can be removed

# Analyzing the categorical columns holiday,weather,weather description we get

1. holiday: we have unique values as -Columbus Day' ,'Veterans Day' ,'Thanksgiving Day',etc
2. weather : we have unqiue values as -'Clouds', 'Clear', 'Rain',etc
3. weather description: we have unique values as-'scattered clouds' ,'broken clouds', 'overcast clouds',etc

We now get to know what is the count o feach of the values starting with :

1. holiday : Labor Day has 7 which is the highest
2. weather : 'Clouds' has 15164 which is the highest
3. weather_descpription: 'sky is clear' has 11665 which is the highest

Creating a holiday_bool column which takes the holiday column and returns the boolean value 1 if holiday has some value else if NaN then gives 0.

Convertion of temp to Celcius makes it easier for our understanding

# Dropping the outliers:

1. temp_c: we are dropping all the values less than -50c
2. rain_1hr: we are dropping all the values greater than 9000

# Conversion of weather_main to categorical with specific categories:

1.Create a one-hot encoded version of the weather_main_cat column.
2.pd.get_dummies() generates separate columns for each category in weather_main_cat with prefix weather_main to avoid clashing if other columns are encoded 3. df.drop(columns=["weather_main_cat"], errors="ignore"):This drops the intermediate weather_main_cat column (if it exists)

The code groups rows with the same date_time value and applies the max function to each column within each group. It returns a new DataFrame where each unique date_time is represented by the maximum value of all columns in that group.

Creating a columns for hour wise and then plotting them

creating a column for every day of the week (monday-0, tuesday-1, wednesday-2,thursday-3,friday-4,saturday-5,sunday-6)

# plot a scatter plot of everyday with traffic:

## Monday:

Peak Traffic Hours:
There are clear peaks during the morning (around 6-8 AM) and evening (around 3-6 PM), likely corresponding to rush hours due to commuting patterns.
Low Traffic Periods:

Traffic volume is lowest during the late-night/early-morning hours (around 12 AM to 5 AM).
Midday Stability:

After the morning rush, traffic remains relatively high but stable during the day (from 10 AM to 3 PM).
Evening Decline:

After the evening rush, traffic gradually declines from around 7 PM onwards.

## Tuesday:

Peak Traffic Hours:

There are clear peaks during the morning (around 7-9 AM) and evening (around 4-6 PM), likely corresponding to rush hours due to commuting patterns.
Low Traffic Periods:

Traffic volume is lowest during the late-night/early-morning hours (around 12 AM to 5 AM).
Midday Stability:

After the morning rush, traffic remains relatively high but stable during the day (from 10 AM to 3 PM).
Evening Decline:

After the evening rush, traffic gradually declines from around 7 PM onwards.

## Wednesday:

Peak Traffic Hours:
There are clear peaks during the morning (around 7-9 AM) and evening (around 4-6 PM), likely corresponding to rush hours due to commuting patterns.
Low Traffic Periods:

Traffic volume is lowest during the late-night/early-morning hours (around 12 AM to 5 AM).
Midday Stability:

After the morning rush, traffic remains relatively high but stable but slower during the day (from 10 AM to 3 PM).
Evening Decline:

After the evening rush, traffic gradually declines from around 7 PM onwards.

## Thursday:

Peak Traffic Hours:
There are clear peaks during the morning (around 7-9 AM) and evening (around 3-5 PM), likely corresponding to rush hours due to commuting patterns.
Low Traffic Periods:

Traffic volume is lowest during the late-night/early-morning hours (around 12 AM to 5 AM).
Midday Stability:

After the morning rush, traffic remains relatively high but stable during the day (from 10 AM to 3 PM).
Evening Decline:

After the evening rush, traffic gradually declines from around 6 PM onwards.

## Friday:

Peak Traffic Hours:

Similar to Tuesday and Wednesday, the morning rush hour peak is observed around 7-9 AM, though it may appear slightly less intense.
Unlike other weekdays, the afternoon and early evening (2-6 PM) show a sustained and pronounced peak, reflecting both work-related commutes and early weekend activities.
Low Traffic Periods:

Traffic volume remains very low during the late-night and early-morning hours (from 12 AM to 5 AM).
Evening Decline:

After the extended evening peak, traffic begins to decline later at night but may remain slightly higher compared to earlier weekdays, possibly due to Friday night activities.
Variability Across the Day:

There is more scatter and variability in traffic during the afternoon and evening hours, which could reflect varying schedules, social plans, or leisure activities typical of Fridays.

## Saturday:

Peak Traffic Hours:
Clear peaks are observed during the morning (around 7-9 AM) and evening (around 3-5 PM), likely due to commuting patterns and increased activity.
Low Traffic Periods:
Traffic volume is lowest during the late-night/early-morning hours (12 AM to 5 AM).
Midday Stability:
After the morning rush, traffic remains relatively high but stable during the day (10 AM to 3 PM).
Evening Decline:
After the evening rush, traffic gradually declines starting from around 6 PM onwards.

## sunday:

Peak Traffic Hours:
The highest traffic is observed during the late morning to early afternoon (10 AM to 2 PM). This is likely due to weekend leisure activities, shopping, or outings.
Low Traffic Periods:
Traffic volume is lowest during the early morning hours (12 AM to 6 AM), which is typical for Sundays when most people are resting.
Midday Stability:
Traffic rises significantly in the morning (7 AM onwards) and remains consistently high during the midday hours (10 AM to 2 PM).
Evening Decline:
Traffic starts to decline after 3 PM, with a steady decrease into the late evening and night. This could indicate the end of weekend activities as people prepare for the workweek.

# plot a scatter plot of how the traffic is at a particular hour throughuout the week:

1. 7 am the traffic is steadily increasing as we can see from the scatter plot the volume is shifting towards up indicating the traffic is getting congested
2. from 9 am to 12pm the traffic volume is high and the congested also due to office hours
3. from 3pm the traffic colume is reducing but not significant enough
4. from 4pm to 7pm the traffic is again high dues to disperse to the houses so the traffic volume is also high
5. from 8pm the traffic colume again can be seen as decreasing the volume is reducing
6. from 9pm the traffic is least again reducing back to again 12am

# holiday or not plot

is_day_off is considered as : if sunday or saturday then 1 else 0
then plot for is_day_off==0 meaning if not a holiday then plotting the scatterplot for traffic
Plotting the same for is_day_off==1 meaning if it is a holiday then plotting the scatterplot for traffic

1. during the work days the traffic is consistent as it is touching the 7k band but on work off days the traffic is not that much
   We can infer from the scatters that traffic during the holidays is less than durig the working days.

Finding the Correlation among the attribute of dtype==number gives us a heatmap where we can findthe correlations between the variables

# Results

1. Graph representation type: Scatter Plot

   The red dashed line represents the perfect prediction line where predicted values exactly match actual values.

   ![alt text](images/image-1.png)

   Purpose of the Graph: This scatter plot is used to evaluate how well the Random Forest Regression model predicts the target variable (traffic volume). It provides a visual comparison between the actual values and the predicted values, highlighting the accuracy of the model.

   Insights:

   Cluster Alignment:
   The majority of points align closely along the red dashed line, indicating that the model is making accurate predictions in most cases.

   Dispersion:
   Some points deviate from the line, especially in the lower range (e.g., actual traffic volume under 2000). These deviations suggest areas where the model struggles slightly.

   Model Accuracy:
   The overall spread of points along the diagonal suggests that the model captures the trend of traffic volume effectively

2. Graph representation type: Residuals Plot

   ![alt text](images/image-2.png)

   Purpose: This plot displays the residuals (errors) of the model, i.e., the difference between actual traffic volume and predicted traffic volume. The residuals are plotted against the predicted values to assess the model’s performance.

   Insights:

   Centered Residuals: The residuals seem to be centered around the red dashed line at 0, which indicates that the model does not have significant bias (positive or negative).

   Random Dispersion: The residuals appear to be randomly scattered without clear patterns, suggesting that the model has captured the main relationships in the data well.

3. Graph representation type: Residuals histogram

   ![alt text](images/image-3.png)

   Purpose: This histogram visualizes the distribution of residuals (the errors between predicted and actual traffic volumes) to check for normality.

   Insights:

   Skewed Distribution: The residuals are not perfectly normally distributed and are slightly skewed towards the negative side, which may indicate model bias or some unmodeled patterns.

   Central Peak: The histogram shows a peak around zero, which is expected, but the spread suggests the model might struggle with higher traffic volumes.

4. Graph representatiom type: Precision-Recall Curve

   ![alt text](images/image-4.png)

   Purpose: The graph shows the model's trade-off between precision and recall, helping evaluate its performance, especially for imbalanced datasets.

   This graph demonstrates the model's balance between Precision and Recall, with an average precision (AP) score of 0.83.

   Insights:

   High Precision at Low Recall: The model achieves near-perfect precision when identifying a limited number of true positives.

   Steady Trade-off: Precision gradually declines as recall increases, showing the expected trade-off between capturing more true positives and minimizing false positives.

   Performance Summary: An AP score of 0.83 indicates strong overall performance, making the model suitable for tasks requiring a balance between precision and recall.

5. Graph representatiom type: ROC curve

![alt text](images/image-5.png)

Purpose: The ROC Curve demonstrates the model's ability to differentiate between classes by plotting the True Positive Rate (TPR) against the False Positive Rate (FPR)

This graph represents the Receiver Operating Characteristic (ROC) Curve for a regression model used as a classification task, with an Area Under the Curve (AUC) of 0.95.

Insights:

High AUC Score (0.95): The model performs exceptionally well, effectively distinguishing between classes. AUC values close to 1 indicate excellent classification ability.

Steep Initial Rise: The curve rises sharply near the origin, reflecting a low False Positive Rate (FPR) while achieving a high True Positive Rate (TPR).

Minimal Deviation from Ideal Performance: The curve stays close to the top-left corner, indicating that the model predicts true positives with minimal false positives.

6. Graph representatiom type: Q-Q plot

   ![alt text](images/image-6.png)

   Purpose: The Q-Q plot compares the quantiles of the dataset with the quantiles of a theoretical distribution (typically normal) to assess how well the dataset matches that distribution.

   Insights:

   Good Normal Fit in the Middle: The data closely follows the red reference line in the center, suggesting that the middle portion of the data is approximately normally distributed.

   Deviations in the Tails: The points deviate from the line at the lower and upper ends, indicating outliers or heavy tails, which suggest the data may not follow a perfect normal distribution in these regions.

   Possible Non-Normal Distribution: The deviations in the tails imply the data could be better described by a different distribution (e.g., heavy-tailed or skewed), or the presence of extreme values that need further investigation.
#   I n t e r s t a t e T r a f f i c V o l u m e P r e d i c t i o n - R a n d o m - F o r e s t - R e g r e s s o r - 
 
 