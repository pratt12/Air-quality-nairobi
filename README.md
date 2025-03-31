# **ARMA Model for Air Quality Analysis in Dar es Salaam**  
This code demonstrates **time series forecasting** of PM2.5 levels in Dar es Salaam using **AutoRegressive (AR) modeling** from `statsmodels`library. 

---

## **1. Data Collection & Preprocessing**
### **1.1 MongoDB Connection & Querying**
- Connects to MongoDB and fetches PM2.5 (`P2`) data for **site 11** in Dar es Salaam.
```python
client = MongoClient(host="localhost", port=27017)
db = client["air-quality"]
dar = db["dar-es-salaam"]

# Check available sites and their readings count
sites = dar.distinct("metadata.site")
result = dar.aggregate([
    {"$group": {"_id": "$metadata.site", "count": {"$count": {}}}}
])
readings_per_site = list(result)
```

### **1.2 Data Wrangling (`wrangle` function)**
- Filters PM2.5 readings (`< 100` to remove outliers).
- Resamples to **hourly frequency** and forward-fills missing values.
- Converts timestamps to **Dar es Salaam timezone**.
```python
def wrangle(collection):
    results = collection.find(
        {"metadata.site": 11, "metadata.measurement": "P2"},
        projection={"P2": 1, "timestamp": 1, "_id": 0},
    )
    df = pd.DataFrame(results).set_index("timestamp")
    df.index = df.index.tz_localize("UTC").tz_convert("Africa/Dar_es_Salaam")
    df = df[df["P2"] < 100]  # Remove outliers
    df = df.resample("1H").mean().fillna(method='ffill')
    return df["P2"]  # Return as a Series

y = wrangle(dar)  # y = PM2.5 time series
```

---

## **2. Exploratory Data Analysis (EDA)**
### **2.1 Time Series Plot**
- Visualizes raw PM2.5 trends.
```python
fig, ax = plt.subplots(figsize=(15, 6))
y.plot(xlabel="Time", ylabel="PM2.5", title="PM2.5 Time Series", ax=ax)
```

### **2.2 Rolling Average (7-Day Smoothing)**
- Helps identify long-term trends.
```python
fig, ax = plt.subplots(figsize=(15, 6))
y.rolling(168).mean().plot(ax=ax, xlabel="Date", ylabel="PM2.5", title="7-Day Rolling Avg")
```

### **2.3 Autocorrelation (ACF & PACF)**
- **ACF** helps determine **MA (Moving Average) terms**.
- **PACF** helps determine **AR (AutoRegressive) terms**.
```python
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf

fig, ax = plt.subplots(figsize=(15, 6))
plot_acf(y, ax=ax, title="ACF for PM2.5")
plot_pacf(y, ax=ax, title="PACF for PM2.5")
```
**Observations:**
- **ACF** shows slow decay → suggests **non-stationarity** (may need differencing).
- **PACF** cuts off after lag `p` → suggests **AR(p)** model.

---

## **3. Train-Test Split (90-10)**
- **90% training**, **10% testing** (sequential split for time series).
```python
cutoff_test = int(len(y) * 0.9)
y_train = y[:cutoff_test]
y_test = y[cutoff_test:]
```

---

## **4. Baseline Model (Mean Prediction)**
- Predicts the **mean PM2.5** of the training set.
```python
y_train_mean = y_train.mean()
y_pred_baseline = [y_train_mean] * len(y_train)
mae_baseline = mean_absolute_error(y_train, y_pred_baseline)
print("Baseline MAE:", mae_baseline)
```

---

## **5. AutoRegressive (AR) Model Selection**
### **5.1 Grid Search for Best Lag (`p`)**
- Tests `p = 1 to 30` and selects the best based on **Mean Absolute Error (MAE)**.
```python
from statsmodels.tsa.ar_model import AutoReg

p_params = range(1, 31)
maes = []

for p in p_params:
    model = AutoReg(y_train, lags=p).fit()
    y_pred = model.predict().dropna()
    mae = mean_absolute_error(y_train.iloc[p:], y_pred)
    maes.append(mae)

mae_series = pd.Series(maes, index=p_params, name="MAE")
best_p = mae_series.idxmin()  # Best lag (p) with lowest MAE
```

### **5.2 Train Best AR Model**
```python
best_model = AutoReg(y_train, lags=best_p).fit()
print(best_model.summary())
```

### **5.3 Check Residuals**
- Residuals should be **white noise** (no autocorrelation).
```python
y_train_resid = best_model.resid
y_train_resid.plot(title="Residuals Plot")
plot_acf(y_train_resid, title="ACF of Residuals")
```

---

## **6. Walk-Forward Validation (WFV)**
- Simulates **real-time forecasting** by updating the model with new data.
```python
y_pred_wfv = pd.Series()
history = y_train.copy()

for i in range(len(y_test)):
    model = AutoReg(history, lags=best_p).fit()
    next_pred = model.forecast()  # Predict next step
    y_pred_wfv = y_pred_wfv.append(next_pred)
    history = history.append(y_test[next_pred.index])  # Update history

y_pred_wfv.name = "prediction"
```

### **6.1 Plot Predictions vs. Actual**
```python
df_pred_test = pd.DataFrame({"y_test": y_test, "y_pred_wfv": y_pred_wfv})
fig = px.line(df_pred_test)
fig.update_layout(
    title="WFV Predictions vs Actual",
    xaxis_title="Date",
    yaxis_title="PM2.5",
)
```

---

## **Key Takeaways**
| **Step** | **Code** | **Purpose** |
|----------|---------|------------|
| **Data Collection** | `dar.find({filter})` | Fetch PM2.5 data from MongoDB |
| **Preprocessing** | `.resample("1H").mean().fillna("ffill")` | Clean and resample data |
| **EDA** | `plot_acf(y)`, `plot_pacf(y)` | Check autocorrelation |
| **Train-Test Split** | `y[:cutoff]`, `y[cutoff:]` | 90-10 split |
| **Baseline** | `[y_train.mean()] * len(y_train)` | Simple mean prediction |
| **AR Model Selection** | `AutoReg(y_train, lags=p).fit()` | Find best lag (`p`) |
| **Walk-Forward Validation** | `model.forecast()` | Simulate real-time predictions |
