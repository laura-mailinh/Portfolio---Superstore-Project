import numpy as np
import pandas as pd

file = 'Sample - Superstore.csv'
df = pd.read_csv(file)

#Data Cleaning
print(df.info())
print(df.describe())
print(df.isnull().sum())
print(df.duplicated().sum())

df["Order Date"] = pd.to_datetime(df["Order Date"], errors="coerce")
df = df.dropna(subset=["Order Date"])

#Phase 1: Customer Revenue Contribution
customer_revenue = df.groupby("Customer ID")["Sales"].sum().sort_values(ascending=False).reset_index()
total_revenue = df["Sales"].sum()
customer_revenue["Revenue %"] = (customer_revenue["Sales"]/total_revenue * 100)
customer_revenue["Cumulative %"] = customer_revenue["Revenue %"].cumsum()
print(customer_revenue.round(2))

n_customers = len(customer_revenue)
top_20_count = int(0.2 * n_customers)
top_20_revenue = customer_revenue.head(top_20_count)["Sales"].sum()
top_20_percent = top_20_revenue / total_revenue * 100
print("Top 20% customers contribute:", top_20_percent.round(2),"%")

top_10_count = int(0.1 * n_customers)
top_10_revenue = customer_revenue.head(top_10_count)["Sales"].sum()
top_10_percent = top_10_revenue / total_revenue * 100
print("Top 10% customers contribute:", top_10_percent.round(2),"%")

#Revenue by Segment
customer_revenue["Segment"] = pd.qcut(customer_revenue["Sales"], q=3, labels=["Low", "Medium", "High"])
segment_revenue = customer_revenue.groupby("Segment")["Sales"].sum().round(2).reset_index()
segment_revenue["Revenue %"] = (segment_revenue["Sales"] / total_revenue * 100).round(2)
segment_revenue = segment_revenue.sort_values("Sales", ascending=False)
print(customer_revenue["Segment"].value_counts())
print(segment_revenue)

#Phase 2: Customer Retention (Repeat Purchase Analysis)
orders_per_customer = df.groupby("Customer ID")["Order ID"].nunique()
repeat_customers = orders_per_customer[orders_per_customer > 1].count()
onetime_customers = orders_per_customer[orders_per_customer == 1].count()
total_customers = orders_per_customer.count()
repeat_rate = repeat_customers / total_customers * 100
print("Repeat Customers:", repeat_customers)
print("One-time Customers:", onetime_customers)
print("Repeat Purchase Rate", repeat_rate.round(2),"%")

#purchase frequency -> customer behavior
purchase_frequency = orders_per_customer.value_counts().sort_index().reset_index()
purchase_frequency.columns = ["Number of Orders", "Number of Customers"]
print(purchase_frequency)

#Phase 3: Profitability Analysis
discount_profit = df.groupby("Discount").agg({"Profit":"mean", "Sales": "mean", "Quantity":"mean"}).reset_index()
print(discount_profit.round(2))

category_profit_margin = df.groupby("Category").agg({"Sales":"sum", "Profit":"sum", "Quantity":"sum"}).reset_index()
category_profit_margin["Category Profit Margin %"] = category_profit_margin["Profit"]/category_profit_margin["Sales"] * 100
print(category_profit_margin.round(2))

profit_products = df.groupby("Product Name")["Profit"].sum().round(2).reset_index()
top_profit = profit_products.sort_values("Profit", ascending=False).head(5)
top_loss = profit_products.sort_values("Profit").head(5)
print(top_profit)
print(top_loss)

product_demand = df.groupby("Category")["Quantity"].sum().reset_index()
print(product_demand)

#Phase 4: Regional Performance Strategy
region_performance = df.groupby("Region").agg({"Sales":"sum", "Profit":"sum"}).reset_index()
region_performance["Profit Margin %"] = region_performance["Profit"]/region_performance["Sales"] * 100
print(region_performance.round(2))

df["Year"] = df["Order Date"].dt.year
region_growth = df.groupby(["Year", "Region"])["Sales"].sum().round(2).reset_index()
print(region_growth)

#Phase 5: Sales Forecasting
#monthly sales trend
df["YearMonth"] = df["Order Date"].dt.to_period("M").dt.to_timestamp()
monthly_sales = df.groupby("YearMonth")["Sales"].sum().reset_index()
monthly_sales = monthly_sales.sort_values("YearMonth")
print(monthly_sales.round(2))

df["Month"] = df["Order Date"].dt.month
seasonality = df.groupby("Month")["Sales"].sum().reset_index()
print(seasonality)

#Sales Forecast
train = monthly_sales[:-6].copy()
test = monthly_sales[-6:].copy()

#Moving Average Forecast
window = 3
moving_avg_forecast = []
history = train["Sales"].tolist()
for i in range(len(test)):
    forecast = sum(history[-window:])/window
    moving_avg_forecast.append(forecast)
    history.append(test["Sales"].iloc[i])

test["MA_Forecast"] = moving_avg_forecast

#Exponential Smoothing
from statsmodels.tsa.holtwinters import ExponentialSmoothing
model_es = ExponentialSmoothing(train["Sales"], trend="add", seasonal="add", seasonal_periods=12)
fit_es = model_es.fit()
es_forecast = fit_es.forecast(len(test))
test["ES_Forecast"] = es_forecast.values

#Prophet Forecast
from prophet import Prophet
prophet_train = train[["YearMonth", "Sales"]].copy()
prophet_train = prophet_train.rename(columns={"YearMonth":"ds","Sales":"y"})
prophet_train["ds"] = pd.to_datetime(prophet_train["ds"])
model_prophet = Prophet()
model_prophet.fit(prophet_train)
future = model_prophet.make_future_dataframe(periods=len(test),freq="M")
forecast_prophet = model_prophet.predict(future)
prophet_forecast = forecast_prophet["yhat"].tail(len(test)).values
test["Prophet_Forecast"] = prophet_forecast
print(test.round(2))

from sklearn.metrics import mean_absolute_error
mae_ma = mean_absolute_error(test["Sales"], test["MA_Forecast"])
rmse_ma = np.sqrt(((test["Sales"] - test["MA_Forecast"])**2).mean())

mae_es = mean_absolute_error(test["Sales"], test["ES_Forecast"])
rmse_es = np.sqrt(((test["Sales"] - test["ES_Forecast"])**2).mean())

mae_prophet = mean_absolute_error(test["Sales"], test["Prophet_Forecast"])
rmse_prophet = np.sqrt(((test["Sales"] - test["Prophet_Forecast"])**2).mean())

mape_ma = np.mean(np.abs((test["Sales"] - test["MA_Forecast"]) / test["Sales"])) * 100
accuracy_ma = 100 - mape_ma

mape_es = np.mean(np.abs((test["Sales"] - test["ES_Forecast"]) / test["Sales"])) * 100
accuracy_es = 100 - mape_es

mape_prophet = np.mean(np.abs((test["Sales"] - test["Prophet_Forecast"]) / test["Sales"])) * 100
accuracy_prophet = 100 - mape_prophet

comparison = pd.DataFrame({"Model" : ["Moving Average","Exponential Smoothing","Prophet"],
    "MAE" : [mae_ma, mae_es, mae_prophet],
    "RMSE" : [rmse_ma, rmse_es, rmse_prophet],
    "Accuracy %" : [accuracy_ma, accuracy_es, accuracy_prophet]})
print(comparison.round(2))

best_model = comparison.sort_values("RMSE").iloc[0]
print("Best Model")
print(best_model)

#Forecast next year
final_model = ExponentialSmoothing(monthly_sales["Sales"], trend="add",seasonal="add",seasonal_periods=12)
final_fit = final_model.fit(optimized=True)
future_forecast = final_fit.forecast(12)
future_dates = pd.date_range(start=monthly_sales["YearMonth"].max(),periods=13,freq="M")[1:]
forecast_df = pd.DataFrame({"Month":future_dates,"Forecast_Sales":future_forecast.values})
print("Future Forecast Table")
print(forecast_df.round(2))

#Forecast Table
df["Month_Num"] = df["Order Date"].dt.month
monthly = df.groupby(["Year", "Month_Num"])["Sales"].sum().reset_index()
seasonality_avg = monthly.groupby("Month_Num")["Sales"].mean().reset_index()
seasonality_avg.columns = ["Month_Num", "Historical_Avg_Sales"]

forecast_df["Month_Num"] = forecast_df["Month"].dt.month
forecast_clean = forecast_df.copy()
#forecast_clean = forecast_df.groupby("Month_Num")["Forecast_Sales"].mean().reset_index())
forecast_table = pd.merge(seasonality_avg,forecast_clean, on="Month_Num", how="left")

last_year = df[df["Year"] == df["Year"].max()]
baseline = last_year.groupby(last_year["Order Date"].dt.month)["Sales"].sum().reset_index()
baseline.columns = ["Month_Num", "Last_Year_Sales"]
final_forecast_table = pd.merge(baseline, forecast_table, on="Month_Num", how="left")

print(final_forecast_table)

#Export files
customer_revenue.to_csv("customer_revenue.csv", index=False)
purchase_frequency.to_csv("purchase_frequency.csv", index=False)
segment_revenue.to_csv("segment_revenue.csv", index=False)
product_demand.to_csv("product_demand.csv", index=False)
discount_profit.to_csv("discount_profit.csv", index=False)
category_profit_margin.to_csv("category_profit_margin.csv", index=False)
top_profit.to_csv("top_profit.csv", index=False)
top_loss.to_csv("top_loss.csv", index=False)
region_growth.to_csv("region_growth.csv", index=False)
region_performance.to_csv("region_performance.csv", index=False)
monthly_sales.to_csv("monthly_sales.csv", index=False)
seasonality.to_csv("seasonality.csv", index=False)
comparison.to_csv("model_comparison.csv", index=False)
final_forecast_table.to_csv("final_forecast_sales.csv", index=False)



