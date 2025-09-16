# Strategic Analysis of the US Hass Avocado Market
![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)

A deep-dive data science project exploring the US Hass Avocado market dynamics from 2015 to 2018. This repository contains a multi-faceted analysis covering **Exploratory Data Analysis (EDA)**, **Time-Series Forecasting**, and **Regional Market Segmentation** to derive actionable business intelligence.

---

## Table of Contents
1.  [Project Objective](#1-project-objective)
2.  [Problem Statement](#2-problem-statement)
3.  [Technologies & Libraries Used](#3-technologies--libraries-used)
4.  [Project Workflow & Methodology](#4-project-workflow--methodology)
    - [4.1 Data Preparation & EDA](#41-data-preparation--exploratory-data-analysis-eda)
    - [4.2 Time-Series Forecasting](#42-time-series-forecasting-of-average-selling-price-asp)
    - [4.3 Regional Market Segmentation](#43-regional-market-segmentation-with-k-means-clustering)
5.  [Key Insights & Conclusion](#5-key-insights--conclusion)
6.  [Business Impact & Pain Points Solved](#6-business-impact--pain-points-solved)
7.  [Project Enhancements (Future Scope)](#7-project-enhancements-future-scope)
8.  [Project Governance & Best Practices](#8-project-governance--best-practices)
    - [Reproducibility & Quality](#reproducibility--quality)
    - [Limitations](#limitations)
    - [Ethics, Privacy & Security](#ethics-privacy--security)
9.  [Setup & Installation](#9-setup--installation)

---

## 1. Project Objective 🎯

The primary objective of this project is to leverage historical sales data for Hass avocados to build a robust analytical framework. This framework aims to uncover deep insights into market behavior, forecast future price trends, and identify distinct regional market segments. The ultimate goal is to provide data-driven recommendations that can inform strategic decisions in pricing, supply chain management, and marketing.

## 2. Problem Statement 🤔

A national avocado distributor faces several business challenges due to a lack of deep, quantitative understanding of its market. Key operational and strategic questions remain unanswered:

1.  **Market Dynamics**: What are the fundamental drivers of sales? How do key metrics like Average Selling Price (ASP), sales volume, and revenue behave over time, and what is the relationship between them? Are there predictable seasonal patterns?
2.  **Forecasting Accuracy**: How can we accurately predict future avocado prices? Reliable forecasts are critical for financial planning, negotiating with suppliers, and setting competitive prices.
3.  **Regional Differences**: Is the US market for avocados monolithic, or does it consist of distinct regional segments? A one-size-fits-all marketing and distribution strategy may be inefficient if regions exhibit unique purchasing behaviors, price sensitivities, and seasonal demands.

This project addresses these questions by performing a comprehensive analysis to transform raw data into a strategic asset.

---

## 3. Technologies & Libraries Used 🛠️

This project leverages a combination of standard data science libraries and advanced modeling frameworks.

* **Data Manipulation & Analysis**: `Pandas`, `NumPy`
* **Statistical Modeling & Time Series**: `Statsmodels` (for OLS, ARIMA, SARIMA, ADF test)
* **Machine Learning & Clustering**: `Scikit-learn` (for StandardScaler, PCA, KMeans, silhouette_score, metrics)
* **Deep Learning**: `TensorFlow` / `Keras` (for LSTM and GRU models)
* **Data Visualization**: `Matplotlib`

---

## 4. Project Workflow & Methodology 📊

The analysis is structured as a sequential workflow, with each notebook building upon the insights of the previous one.

### 4.1. Data Preparation & Exploratory Data Analysis (EDA)

**(Notebook: `01_EDA.ipynb`)**

The first step was to thoroughly explore the dataset and derive meaningful Key Performance Indicators (KPIs) to understand the market's pulse.

#### **Methodology:**
* **Data Cleaning**: Standardized column names and converted date columns to datetime objects.
* **Feature Engineering**: Created a `revenue_proxy` feature, calculated as $ASP \times Total~Volume$.
* **KPI Calculation**: Aggregated data to calculate several crucial metrics:
    * National and regional weekly **Average Selling Price (ASP)** and **Total Volume**.
    * Week-over-week **Volume Growth %** and **ASP Change %**.
    * **Market Share** by region and avocado type (conventional vs. organic).
    * A monthly **Seasonality Index** for both price and volume to quantify seasonal fluctuations.
    * An exploratory **Price Elasticity of Demand** using OLS regression of $log(Volume)$ on $log(Price)$, controlling for region and month fixed effects.

#### **Key Findings:**
* **Strong Seasonality**: Both price and volume exhibit strong, predictable annual patterns. Volume consistently peaks around the Super Bowl (February) and Cinco de Mayo (May), while ASP peaks in late summer.
* **Inverse Price-Volume Relationship**: As expected, periods of high volume generally correspond to lower average prices, and vice-versa.
* **High Price Elasticity**: The exploratory regression estimated a price elasticity of approximately **-5.6**. This indicates a highly elastic market where a 1% increase in price is associated with a ~5.6% decrease in demand, holding other factors constant. This is a critical insight for pricing strategy.
* **Organic vs. Conventional**: Organic avocados consistently command a significant price premium over their conventional counterparts across all regions.


### 4.2. Time-Series Forecasting of Average Selling Price (ASP)

**(Notebook: `02_TimeSeries_ARIMA_SARIMA_LSTM_GRU.ipynb`)**

Building on the time-series nature of the data, this notebook focuses on forecasting the national weekly ASP. A comparison was made between traditional statistical models and more complex deep learning models.

#### **Methodology:**
* **Series Preparation**: The national weekly average ASP was extracted and checked for stationarity using the Augmented Dickey-Fuller (ADF) test, which indicated the series was non-stationary.
* **Train-Validation Split**: The data was split into an 85% training set (Jan 2015 - Sep 2017) and a 15% validation set (Oct 2017 - Mar 2018).
* **Model Development**:
    1.  **ARIMA**: A baseline non-seasonal model was tuned over a small hyperparameter grid. The best model was ARIMA(1,1,0).
    2.  **SARIMA**: A seasonal model was tested to capture the 52-week annual seasonality observed in the EDA.
    3.  **LSTM & GRU**: Recurrent Neural Networks were built using a sliding window approach (12 weeks of history to predict the next week). Simple architectures were used for fast training.
* **Evaluation**: Models were benchmarked on the validation set using **Root Mean Squared Error (RMSE)** and **Mean Absolute Percentage Error (MAPE)**.

#### **Results & Conclusion:**
The deep learning models significantly outperformed the statistical models, with the **GRU model achieving the lowest error rates** by a substantial margin. This suggests that the non-linear patterns in the ASP data are better captured by RNNs.

| Model  | RMSE    | MAPE    |
| :----- | :------ | :------ |
| GRU    | 0.0098  | 5.82%   |
| LSTM   | 0.0427  | 13.71%  |
| SARIMA | 0.1533  | 26.31%  |
| ARIMA  | 0.1716  | 27.74%  |

The winning model (GRU) was then retrained on the entire dataset to generate a **24-week forward-looking forecast**.


### 4.3. Regional Market Segmentation with K-Means Clustering

**(Notebook: `03_Segmentation_Clustering.ipynb`)**

This analysis shifts from a national to a regional perspective, aiming to group regions into meaningful clusters based on their market characteristics.

#### **Methodology:**
* **Feature Engineering**: A rich set of features was engineered for each of the 54 regions:
    * `mean_asp` & `median_asp`
    * `mean_volume` & `std_volume` (volatility)
    * `std_asp` (price volatility)
    * `yoy_volume_growth` (approximate year-over-year growth)
    * `seasonality_strength` (variance of the monthly volume index)
* **Preprocessing**: Features were standardized using `StandardScaler` to ensure equal weighting in the clustering algorithm.
* **Clustering**: **K-Means clustering** was applied. The optimal number of clusters (`k`) was determined using the Elbow and Silhouette methods, both of which pointed to `k=2`.
* **Visualization**: Principal Component Analysis (PCA) was used to reduce the 7-dimensional feature space to 2 dimensions for easy visualization of the clusters.

#### **Key Findings:**
The analysis uncovered two highly distinct clusters with significant business implications:

* **Cluster 1 ("The Aggregate Anomaly")**: This cluster consists of a single member: **TotalUS**. Its volume characteristics are orders of magnitude larger than any individual region, making it an outlier that behaves unlike any of its constituent parts.
* **Cluster 0 ("The Regional Markets")**: This cluster contains all 53 individual regions. This finding strongly suggests that a **national-level strategy is not representative** of on-the-ground realities. Regional markets, while diverse among themselves, are fundamentally different from the national aggregate.


---

## 5. Key Insights & Conclusion 💡

This multi-pronged analysis yielded several key strategic insights:

1.  **High Price Elasticity**: The market is highly sensitive to price changes. This highlights the importance of a carefully considered pricing strategy to balance revenue and volume goals.
2.  **Predictable Seasonality**: Demand and price follow strong, predictable annual cycles. This allows for proactive planning in inventory, logistics, and marketing campaigns to align with consumer behavior.
3.  **Advanced Forecasting Power**: For national ASP, GRU-based neural networks provide significantly more accurate forecasts than traditional ARIMA/SARIMA models, likely due to their ability to capture complex, non-linear dependencies.
4.  **Market Heterogeneity**: The US is not a single, uniform market. The "TotalUS" aggregate masks the diverse behaviors of individual regions. Effective strategy requires a regional focus rather than a one-size-fits-all national approach.

In conclusion, this project provides a data-driven blueprint for optimizing avocado sales strategy. By understanding elasticity, forecasting prices, and segmenting markets, a distributor can make more informed, profitable decisions.

## 6. Business Impact & Pain Points Solved ✅

This project directly addresses critical pain points for a business operating in the CPG/agriculture space:

| Pain Point                                    | Solution Provided by This Project                                                                      |
| :-------------------------------------------- | :----------------------------------------------------------------------------------------------------- |
| 📉 **Reactive & Inefficient Pricing** | An **elasticity model** provides a quantitative basis for setting prices that maximize revenue or volume. |
| 🚚 **Inventory Mismatch & Supply Chain Risk** | The **time-series forecast** enables proactive inventory management, reducing stockouts and spoilage.          |
| 📢 **Generic, One-Size-Fits-All Marketing** | **Regional segmentation** allows for targeted marketing campaigns and distribution strategies tailored to specific market behaviors. |
| ❓ **Lack of Market Understanding** | **Comprehensive KPIs and visualizations** provide a clear and deep understanding of market dynamics and seasonality. |

---

## 7. Project Enhancements (Future Scope) 🚀

This foundational analysis opens the door to several advanced analytical initiatives.

#### **Advanced Analytics & Modeling**
* **Customer Lifetime Value (CLV)**: Implement BG/NBD and Gamma-Gamma models for a deeper understanding of customer value.
* **Churn & Uplift Modeling**: Use causal uplift models to identify which marketing offers are most effective for specific customer segments.
* **Marketing Mix Modeling (MMM) & Attribution**: Develop top-down MMM and bottom-up Multi-Touch Attribution (MTA) models to optimize marketing channel ROI.
* **Advanced Elasticity Models**: Build more granular demand models to optimize promotional calendars and discount strategies.
* **Personalization**: Develop recommender systems (content-based and collaborative filtering) for personalized customer experiences.

#### **MLOps & Data Governance**
* **Automation**: Convert the forecasting notebook into a production-ready pipeline using tools like **Airflow** or **dbt**.
* **Dashboarding**: Create an executive dashboard using **Streamlit** or **Tableau** to visualize KPIs and forecast results for stakeholders.
* **Data Quality**: Implement automated data quality tests using frameworks like **Great Expectations** to ensure model inputs are reliable.

---

## 8. Project Governance & Best Practices

### Reproducibility & Quality
* Random seeds are fixed where relevant (e.g., K-Means, TensorFlow) to ensure reproducible results.
* Notebooks have a clear, logical cell ordering.
* Plots are professionally labeled and sized for presentation.
* Key assumptions are documented in the markdown cells of each notebook.

### Limitations
* **Data Coverage**: The dataset lacks explicit features for holidays, promotions, or stockout events, which could improve model accuracy.
* **MAPE Sensitivity**: MAPE can be unstable when actual values are close to zero. RMSE is a more robust metric in such cases.
* **Deep Learning Models**: RNNs like LSTM and GRU require a substantial amount of historical data to perform well. For shorter time series, classical models may be more appropriate.
* **Clustering**: The results of K-Means are sensitive to feature scaling and selection. The identified segments should be periodically reviewed as business dynamics evolve.

### Ethics, Privacy & Security
* This project uses a public dataset with no Personally Identifiable Information (PII).
* No plaintext keys, tokens, or credentials are stored in the notebooks.
* All reporting is done on an aggregated basis (national or regional) to prevent re-identification.

---

## 9. Setup & Installation ⚙️

To reproduce this analysis, follow these steps:

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/your-username/avocado-analysis.git](https://github.com/your-username/avocado-analysis.git)
    cd avocado-analysis
    ```

2.  **Create a virtual environment (recommended):**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3.  **Install the required packages:**
    ```bash
    pip install -r requirements.txt
    ```
    *(Note: `requirements.txt` should be created containing pandas, numpy, matplotlib, scikit-learn, statsmodels, and tensorflow).*

4.  **Run the Jupyter Notebooks:**
    ```bash
    jupyter notebook
    ```
    Navigate through the notebooks in numerical order (`01_*`, `02_*`, `03_*`) to follow the project workflow.
