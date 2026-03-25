**🚀 Customer Segmentation using Unsupervised Machine Learning**

**📌 Capstone Project: Advanced Customer Segmentation**
________________________________________
🎯 Project Overview

Customer segmentation is a critical component of modern data-driven businesses. This project leverages unsupervised machine learning techniques to segment customers based on transactional and behavioural data.
The goal is to uncover hidden patterns and transform raw data into actionable business insights that improve:
•	Marketing strategies 
•	Customer retention 
•	Revenue generation 
________________________________________
❗ Problem Statement

Organizations often struggle to understand their customers due to lack of labeled data. This leads to:
•	Inefficient marketing campaigns 
•	Poor targeting of high-value customers 
•	Increased customer churn 
✅ Solution Approach
This project solves the problem by:
•	Identifying meaningful customer segments 
•	Analyzing purchasing behavior 
•	Enabling data-driven decision making 
________________________________________
📊 Dataset Description

Dataset: Retail Transaction Dataset
Features:
•	Customer ID
•	Product ID 
•	Quantity 
•	Price 
•	Total Amount 
•	Transaction Date 
•	Payment Method 
•	Store Location 
•	Product Category 
•	Discount 

🔧 Feature Engineering

•	Recency (R): Days since last purchase 
•	Frequency (F): Number of transactions 
•	Monetary (M): Total spending 
•	Average Transaction Value 
•	Customer Lifetime Value (CLV) 
________________________________________
⚙️ Machine Learning Lifecycle

1️⃣ Data Preprocessing
•	Handling missing values 
•	Outlier detection & treatment 
•	Encoding categorical variables 
•	Feature scaling (StandardScaler) 
________________________________________
2️⃣ Exploratory Data Analysis (EDA)
•	Univariate, bivariate, multivariate analysis 
•	Distribution plots 
•	Correlation heatmaps 
•	Pattern discovery 
________________________________________
3️⃣ Feature Engineering
•	RFM analysis 
•	Behavioral feature extraction 
•	Derived metrics (CLV, avg spend) 
________________________________________
🤖 Clustering Algorithms Implemented

•	✅ K-Means Clustering 

•	✅ Hierarchical Clustering

•	✅ DBSCAN 

•	✅ Gaussian Mixture Model (GMM) 
________________________________________
📊 Sample Visualizations

🔹 K-Means Clustering

  <img width="640" height="480" alt="kmeans" src="https://github.com/user-attachments/assets/517bdff4-55d1-45c6-bd24-196019a515c2" />

🔹 DBSCAN Clustering

  <img width="640" height="480" alt="dbscan" src="https://github.com/user-attachments/assets/0f53fb1c-f5eb-4004-83ba-3ee61a6447f2" />

________________________________________
📉 Cluster Optimization

Evaluation techniques used:
•	Elbow Method 
•	Silhouette Score 
•	Davies-Bouldin Index 
👉 Final Decision: K = 4 clusters
✔ Reason:
•	Best silhouette score 
•	Clear cluster separation 
•	Business interpretability 
________________________________________
📉 Dimensionality Reduction

•	PCA (Principal Component Analysis) 
•	Reduced high-dimensional data into 2D 
•	Enabled visualization of clusters 
________________________________________
🧠 Model Comparison
## 🧠 Model Comparison

| Model          | Performance        | Silhouette Score | Remarks                                      |
|----------------|-------------------|------------------|----------------------------------------------|
| **K-Means**    | ⭐ **Best**        | Highest          | Well-separated clusters, highly scalable     |
| **GMM**        | Good              | Slightly lower   | Handles overlapping clusters effectively     |
| **Hierarchical** | Moderate        | Moderate         | Useful for hierarchy but less scalable       |
| **DBSCAN**     | Poor              | Low / Unstable   | High noise, not suitable for this dataset    |

👉 Selected Model: K-Means Clustering
________________________________________
📈 Customer Segments

🟢 Cluster 0 – High-Value Customers
•	High spending 
•	Frequent purchases 
👉 Strategy: Loyalty programs, premium offers
________________________________________
🔵 Cluster 1 – Budget Customers
•	Low spending 
•	Price-sensitive 
👉 Strategy: Discounts, bundle offers
________________________________________
🔴 Cluster 2 – At-Risk Customers
•	Low engagement 
•	High churn probability 
👉 Strategy: Retention campaigns
________________________________________
🟡 Cluster 3 – New Customers
•	Recent activity 
•	Moderate spending 
👉 Strategy: Onboarding & engagement
________________________________________
💡 Business Insights

•	💰 Cluster 0 generates maximum revenue 
•	⚠️ Cluster 2 has highest churn risk 
•	📊 Cluster 1 responds well to discounts 
•	🚀 Cluster 3 has growth potential

👉 Enables:

•	Personalized marketing 
•	Better retention strategies 
•	Increased profitability 
________________________________________
🚀 Advanced Techniques
🔍 Feature Importance

Key drivers of segmentation:

•	Total spending 
•	Purchase frequency 
•	Customer activity 
________________________________________
💰 Customer Lifetime Value (CLV)

CLV = Total Spend × Frequency
👉 Insights:

•	High-value customers have highest CLV 
•	At-risk customers show declining CLV 
________________________________________
📁 Project Structure

customer-segmentation-unsupervised-mastani/

customer-segmentation-unsupervised-mastani/
│
├── data/
│   ├── raw/
│   │   └── retail_transactions.csv         
│   │
│   ├── processed/
│       ├── cleaned_data.csv                
│       └── rfm_features.csv                 
│
├── notebooks/
│   ├── 01_data_preprocessing.ipynb          
│   ├── 02_eda.ipynb                        
│   ├── 03_feature_engineering.ipynb        
│   ├── 04_clustering_models.ipynb           
│   ├── 05_model_comparison.ipynb            
│   ├── 06_visualization.ipynb              
│
├── src/
│   ├── data_preprocessing.py                
│   ├── feature_engineering.py               
│   │
│   ├── clustering/
│   │   ├── kmeans.py                        
│   │   ├── hierarchical.py                  
│   │   ├── dbscan.py                        
│   │   └── gmm.py                           
│   │
│   ├── evaluation.py                        
│   └── utils.py                             
│
├── results/
│   ├── cluster_plots/
│   │   
│   ├── pca_outputs/              
│   │
│   ├── metrics/
│   |
├── reports/
│   ├── final_report.pdf
│   └── presentation.pptx
│
├── requirements.txt
├── README.md
└── main.py
________________________________________
▶️ How to Run the Project

git clone https://github.com/Mastani9680/customer-segmentation-unsupervised

cd customer-segmentation-unsupervised

pip install -r requirements.txt

python main.py

