
# Regression Model for Metro Interstate Traffic Volume

The purpose of this project is to construct a predictive model using various machine learning algorithms to predict traffic volume. 
The Metro Interstate Traffic Volume dataset contains hourly weather features and holidays included for impacts on traffic volume.

## The dataset contains:
- **Traffic volume**: Measured hourly.
- **Weather attributes**: Temperature, rain, snow, cloud cover.
- **Temporal attributes**: Date, time, holidays, and weekdays.

---

## Analyzing the Numerical Columns (`rain_1h`, `snow_1h`, `temp`, `cloud_all`)
1. **`rain_1h`**: The average value of the rain in 1 hr is `0.334264`, and the highest value is `9831.300000`.
2. **`snow_1h`**: The average value of the snow in 1 hr is around `0.000222`, and the max is `0.5`.
3. **`temp`**: The average value of the temperature is around `281.2`, and the max is `300`.
4. **`cloud_all`**: The average value of the cloud cover is around `49.3`, and the max is `100`.

*The outliers in this case, typically the max values, can be removed.*

---

## Analyzing the Categorical Columns (`holiday`, `weather`, `weather description`)
1. **`holiday`**: Unique values include `Columbus Day`, `Veterans Day`, `Thanksgiving Day`, etc.
2. **`weather`**: Unique values include `Clouds`, `Clear`, `Rain`, etc.
3. **`weather description`**: Unique values include `scattered clouds`, `broken clouds`, `overcast clouds`, etc.

### Frequency Counts:
1. **`holiday`**: `Labor Day` has `7` occurrences, which is the highest.
2. **`weather`**: `Clouds` has `15,164` occurrences, which is the highest.
3. **`weather description`**: `sky is clear` has `11,665` occurrences, which is the highest.

---

## Data Preprocessing Steps

### Creating a `holiday_bool` Column
- This column returns a boolean value: `1` if `holiday` has some value, else `0` if `NaN`.

### Converting `temp` to Celsius
- Conversion makes it easier for interpretation.

### Dropping the Outliers
1. **`temp_c`**: Dropping all values less than `-50°C`.
2. **`rain_1h`**: Dropping all values greater than `9000`.

### Conversion of `weather_main` to Categorical with Specific Categories
1. Created a one-hot encoded version of the `weather_main_cat` column.
2. Used `pd.get_dummies()` to generate separate columns for each category in `weather_main_cat` with a prefix `weather_main`.
3. Dropped the intermediate `weather_main_cat` column (if it exists).

### Grouping by `date_time`:
- Grouped rows with the same `date_time` and applied the `max` function to each column within each group.
- Returned a new DataFrame where each unique `date_time` is represented by the maximum value of all columns in that group.

---

## Hourly Traffic Analysis

### Hour-wise Traffic Analysis
- Created a column for every day of the week (`Monday=0`, `Tuesday=1`, ..., `Sunday=6`).

### Scatter Plot Observations
#### **Monday**:
- **Peak Traffic Hours**: Morning (6-8 AM) and Evening (3-6 PM).
- **Low Traffic Periods**: Late-night/early-morning (12-5 AM).
- **Midday Stability**: Relatively high traffic from 10 AM to 3 PM.
- **Evening Decline**: Gradual decline after 7 PM.

#### **Tuesday**:
- Similar trends as Monday.

#### **Wednesday**:
- Slightly slower midday stability compared to Tuesday.

#### **Thursday**:
- Evening rush observed slightly earlier (3-5 PM).

#### **Friday**:
- Extended evening peak (2-6 PM) reflects early weekend activities.
- Higher variability during afternoon/evening hours.

#### **Saturday**:
- Morning peak (7-9 AM) and early evening peak (3-5 PM).

#### **Sunday**:
- Highest traffic during late morning to early afternoon (10 AM to 2 PM).

### Weekday vs. Weekend Analysis:
1. **Weekdays**: Traffic consistently touches the `7000` band.
2. **Weekends**: Traffic is generally lower.

---

## Correlation Analysis
- A heatmap of numerical attributes reveals the correlation between variables.

---

## Results

### 1. Scatter Plot of Actual vs. Predicted
- **Purpose**: Evaluate the Random Forest Regression model's predictive accuracy.
- **Insights**:
  - Most points align closely along the red dashed line, indicating accurate predictions.
  - Some deviations, particularly for traffic volume under `2000`.

### 2. Residuals Plot
- **Purpose**: Visualize the difference between actual and predicted traffic volumes.
- **Insights**:
  - Residuals are centered around `0`, indicating minimal bias.
  - Random scatter suggests the model captured key relationships effectively.

### 3. Residuals Histogram
- **Purpose**: Check for normality in residuals distribution.
- **Insights**:
  - Slight skew towards negative values, indicating minor model bias.
  - Central peak around `0` with some spread for higher traffic volumes.

### 4. Precision-Recall Curve
- **Purpose**: Evaluate the model's balance between precision and recall.
- **Insights**:
  - High precision at low recall.
  - Average precision (AP) score: `0.83`, indicating strong performance.

### 5. ROC Curve
- **Purpose**: Assess model's classification ability.
- **Insights**:
  - High AUC score (`0.95`) reflects excellent performance.

### 6. Q-Q Plot
- **Purpose**: Assess how well the dataset matches a normal distribution.
- **Insights**:
  - Middle data aligns with reference line, suggesting approximate normality.
  - Deviations in tails indicate possible outliers or heavy-tailed distribution.

---
