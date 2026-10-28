# Retail Analytics Suite: Sales Forecasting & Customer Segmentation

An end-to-end Machine Learning and Business Intelligence solution designed for modern retail enterprises. This project demonstrates how combining **Supervised ML (Sales Demand Forecasting)** and **Unsupervised ML (Customer Segmentation)** solves the dual retail challenges of **inventory inefficiency** and **unfocused marketing spend**.

---

## 📊 Business Problem & Motivation

1. **Inventory Stockouts & Excess Holding Costs:** Retailers often rely on static historical averages, leading to either stockouts during seasonal peaks (lost revenue) or excess inventory holding penalties.
2. **Generic Marketing Waste:** Broadcasting blanket discounts degrades gross margin while missing opportunities to cultivate high-value customer relationships.

---

## 🛠️ Solution Architecture & Machine Learning Models

### 1. Sales Forecasting (Supervised Learning)
- **Features Processed:** Product Category, Sub-Category, Region, Customer Segment, Order Month (Seasonality), Order Quantity, and Applied Discount.
- **Pipeline:** Categorical features encoded with `OneHotEncoder`, numeric features scaled with `StandardScaler`.
- **Models Evaluated:**
  - **Baseline:** Linear Regression ($R^2 \approx 0.76$, RMSE: ~$407)
  - **Final Model:** **Random Forest Regressor ($R^2 \approx 0.89 - 0.91$, RMSE: ~$237)**
- **Proven Accuracy:** **89%+ $R^2$ score**, capturing non-linear seasonal spikes and discount-volume interactions.

### 2. Customer Segmentation (Unsupervised Learning)
- **Aggregated Customer Metrics:** `TotalSpend`, `AvgOrderValue`, `OrderFrequency`, and `TotalProfit`.
- **Algorithm:** **K-Means Clustering** ($k=3$), confirmed via the **Elbow Method (Inertia)** and **Silhouette Analysis**.
- **Identified Customer Personas:**
  - 🌟 **High-Value VIP Loyalists:** Average spend ~$13.5k, high order frequency $\rightarrow$ Targeted with VIP loyalty perks and early access.
  - 🔄 **Core Regular Buyers:** Average spend ~$8.4k, consistent mid-sized basket $\rightarrow$ Targeted with personalized cross-sell bundles.
  - 🏷️ **Budget / Occasional Shoppers:** Lower average spend ~$4.3k $\rightarrow$ Re-engaged with flash sales and free shipping incentives.

---

## 🚀 Repository Structure

```
retail-analytics-suite/
├── Retail_Analytics_Forecasting_Segmentation.ipynb   # Interactive Google Colab / Jupyter Notebook
├── retail_analytics.py                                # Production-ready Python modeling pipeline
├── requirements.txt                                   # Python dependencies
└── README.md                                          # Technical documentation and results
```

---

## 💻 How to Run

### Option 1: On Google Colab
1. Open [Google Colab](https://colab.research.google.com).
2. Upload `Retail_Analytics_Forecasting_Segmentation.ipynb`.
3. Run all cells (`Ctrl + F9`).

### Option 2: Local Python Execution
```bash
git clone https://github.com/MEKAM-SRUJAN/retail-analytics-suite.git
cd retail-analytics-suite
pip install -r requirements.txt
python retail_analytics.py
```

---

## 👤 Author
**Mekam Srujan**  
- **Email:** srujanmekam1@gmail.com  
- **GitHub:** [github.com/MEKAM-SRUJAN](https://github.com/MEKAM-SRUJAN)
