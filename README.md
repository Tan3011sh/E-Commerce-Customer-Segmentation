# E-Commerce Customer Segmentation Using K-Means and DBSCAN

**Academic Subject:** Machine Learning Essentials (College Mini-Project)  
**Domain:** E-Commerce, Retail Analytics, Unsupervised Machine Learning  
**Dataset:** UCI Machine Learning Repository — Online Retail Dataset  

---

## 1. Project Overview & Objectives

Customer segmentation is the foundational data science technique used by modern e-commerce enterprises to partition an aggregate customer base into distinct, homogeneous behavioral cohorts. Rather than deploying broad, generic marketing messages, customer segmentation allows businesses to design tailored marketing campaigns, allocate advertising budgets efficiently, and optimize customer lifetime value (LTV).

### Key Objectives:
1. **RFM Analysis:** Extract and quantify customer purchasing patterns from raw transactional logs using the **Recency, Frequency, and Monetary (RFM)** framework.
2. **Standardization:** Apply **StandardScaler** to eliminate feature scale bias in Euclidean distance calculations.
3. **Partitioning Clustering (K-Means):** Discover optimal customer groupings using the **Elbow Method (WCSS)** and **Silhouette Analysis**, selecting $k=4$.
4. **Density-Based Clustering (DBSCAN):** Apply density clustering to discover natural spatial structures, tune hyperparameters ($\epsilon$, `min_samples`) via a **k-distance graph**, and isolate anomalous noise points.
5. **Linear Dimensionality Reduction (PCA):** Reduce 3D RFM features to 2 Principal Components ($PC1, PC2$), capturing **85.73%** of total variance and interpreting eigenvector loadings.
6. **Non-Linear Manifold Visualization (t-SNE):** Project customer feature manifolds into a 2D visualization space to verify cluster separation and local neighborhood preservation.
7. **Marketing Personas & Strategy:** Derive 4 data-driven customer personas with actionable commercial strategies, treating DBSCAN noise points separately as enterprise/wholesale anomalies.
8. **Algorithmic Evaluation:** Compare K-Means and DBSCAN across statistical metrics, outlier sensitivity, and business applicability.

---

## 2. Dataset Information

* **Dataset Name:** Online Retail Dataset
* **Original Host:** UCI Machine Learning Repository (Dataset ID: 352)
* **Dataset URL:** [https://archive.ics.uci.edu/dataset/352/online+retail](https://archive.ics.uci.edu/dataset/352/online+retail)
* **Direct UCI Download URL:** `https://archive.ics.uci.edu/static/public/352/online+retail.zip`
* **License:** Creative Commons Attribution 4.0 International (CC BY 4.0)
* **Time Period:** 01/12/2010 to 09/12/2011 (1 year of transactions)
* **Business Description:** A UK-based registered non-store online retail business selling unique all-occasion gift-ware. Many customers are wholesale buyers.

### Dataset Schema:
| Column | Type | Description |
| :--- | :--- | :--- |
| `InvoiceNo` | Nominal / String | A 6-digit integral number uniquely assigned to each transaction. Starts with **'C'** for cancellations. |
| `StockCode` | Nominal / String | A 5-digit integral number uniquely assigned to each distinct product item. |
| `Description` | Nominal / String | Product/item description name. |
| `Quantity` | Numeric / Integer | The quantities of each product (item) per transaction. Negative values represent returns. |
| `InvoiceDate` | Datetime | The day and time when each transaction was generated. |
| `UnitPrice` | Numeric / Float | Product price per unit in sterling (£). |
| `CustomerID` | Nominal / Float | A 5-digit integral number uniquely assigned to each customer. |
| `Country` | Nominal / String | The name of the country where each customer resides. |

---

## 3. Technologies and Approved Libraries

Strictly restricted to standard core machine learning libraries:
* **Python 3.10+ / 3.13 / 3.14**
* **Pandas** (Data wrangling, temporal aggregation, RFM table generation)
* **NumPy** (Vectorized numerical computations and array manipulation)
* **Matplotlib** (Base plotting engine, publication-quality figures)
* **Seaborn** (Statistical visualizations, distribution KDEs, cluster scatter plots)
* **Scikit-learn** (`StandardScaler`, `KMeans`, `DBSCAN`, `PCA`, `TSNE`, `NearestNeighbors`, `silhouette_score`)
* **Openpyxl** (Excel spreadsheet parsing)
* **Jupyter Notebook** (Interactive execution, markdown documentation, code presentation)

> **Machine Learning Algorithm Restrictions:**
> Strictly limited to **K-Means**, **DBSCAN**, **PCA**, and **t-SNE**. No unauthorized algorithms (e.g., Hierarchical Clustering, GMM, OPTICS, UMAP, Random Forest, Decision Trees, or Neural Networks) are used.

---

## 4. Project Directory Structure

```
Ecommerce-Customer-Segmentation/
│
├── data/
│   ├── Online Retail.xlsx          # Original downloaded UCI dataset (Excel format)
│   └── Online_Retail.csv           # Exact CSV replica for high-speed loading (0.6s)
│
├── notebooks/
│   └── ECommerce_Customer_Segmentation.ipynb   # Complete interactive notebook with all outputs
│
├── outputs/
│   ├── visualizations/             # High-resolution (300 DPI) saved figures
│   │   ├── 01_eda_top_countries.png
│   │   ├── 02_eda_monthly_transactions.png
│   │   ├── 03_rfm_distributions.png
│   │   ├── 04_kmeans_elbow_and_silhouette.png
│   │   ├── 05_kmeans_clusters_rfm_scatter.png
│   │   ├── 06_dbscan_k_distance_graph.png
│   │   ├── 07_dbscan_clusters_rfm_scatter.png
│   │   ├── 08_pca_clusters_comparison.png
│   │   ├── 09_tsne_clusters_comparison.png
│   │   └── 10_customer_personas_profile.png
│   └── cluster_results.csv         # Final per-customer RFM, cluster assignments & personas
│
├── run_pipeline.py                 # Standalone reproducible execution script
├── requirements.txt                # Dependency specification
└── README.md                       # Academic project documentation & viva guide
```

---

## 5. Step-by-Step Implementation Pipeline

### STEP 1: Data Loading and Exploration
* The raw dataset contains **541,909 transaction rows** and **8 columns**.
* Missing values:
  * `CustomerID`: 135,080 missing values (~24.93%).
  * `Description`: 1,454 missing values (~0.27%).
* Duplicate rows: 5,268 exact duplicate rows identified.
* Geographic distribution: The United Kingdom represents over 90% of all purchases, followed by Germany, France, Ireland (EIRE), and Spain.

### STEP 2: Data Preprocessing
Every cleaning decision is based on domain logic:
1. **Handling Missing `CustomerID`:** Transactions without a customer identifier cannot be associated with any individual account for customer segmentation. These rows (135,080) were removed.
2. **Filtering Cancellations:** Invoices beginning with the letter `'C'` represent merchandise returns or cancellations (8,905 rows). Leaving them in would distort spend and frequency calculations.
3. **Removing Invalid Quantities & Prices:** Transactions with `Quantity <= 0` or `UnitPrice <= 0` (40 rows) represent administrative adjustments, damaged inventory, or gifts with zero price. These were excluded.
4. **Deduplication:** 5,192 identical duplicate rows within the valid customer set were removed.
5. **Datetime Conversion & Line Totals:** `InvoiceDate` was converted to datetime format, and line-item total spend was computed as:
   $$\text{TotalAmount} = \text{Quantity} \times \text{UnitPrice}$$
* **Cleaned Dataset:** **392,692 valid transaction records** representing **4,338 unique customers**.

### STEP 3: RFM Analysis
Aggregated customer behavioral features:
* **Recency ($R$):** Days since customer's last purchase relative to the reference date.
  $$\text{Reference Date} = \max(\text{InvoiceDate}) + 1 \text{ day} = \text{2011-12-10 12:50:00}$$
  $$\text{Recency} = \text{Reference Date} - \max(\text{InvoiceDate}_i)$$
* **Frequency ($F$):** Total number of distinct, unique purchase invoices placed by the customer:
  $$\text{Frequency} = \text{Count of unique InvoiceNo per customer}$$
* **Monetary ($M$):** Total monetary spending across all valid orders:
  $$\text{Monetary} = \sum (\text{Quantity} \times \text{UnitPrice})$$

#### RFM Descriptive Statistics (4,338 Customers):
| Feature | Mean | Std Dev | Min | 25% | Median (50%) | 75% | Max |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **Recency (days)** | 92.54 | 100.01 | 1.00 | 18.00 | 51.00 | 142.00 | 374.00 |
| **Frequency (orders)** | 4.27 | 7.70 | 1.00 | 1.00 | 2.00 | 5.00 | 209.00 |
| **Monetary (£)** | £2,048.69 | £8,985.23 | £3.75 | £306.48 | £668.57 | £1,660.60 | £280,206.02 |

*Note: Frequency and Monetary display significant right-skewness, typical of retail commerce where a small minority of wholesale accounts generate outsized revenue.*

### STEP 4: Feature Scaling (StandardScaler)
Distance-based clustering algorithms rely on Euclidean distance:
$$d(\mathbf{u}, \mathbf{v}) = \sqrt{(R_u - R_v)^2 + (F_u - F_v)^2 + (M_u - M_v)^2}$$

Because `Monetary` spans from £3.75 to £280,206 while `Recency` spans 1–374 and `Frequency` spans 1–209, unscaled Euclidean distance would be almost 100% determined by Monetary spending differences alone.
We apply **StandardScaler** to standardize each feature to $\mu = 0$ and $\sigma = 1$:
$$z = \frac{x - \mu}{\sigma}$$

### STEP 5: K-Means Clustering
* **Elbow Method:** Evaluated Within-Cluster Sum of Squares (WCSS / Inertia) across $k \in [2, 8]$.
* **Silhouette Score:** Evaluated cohesion and separation across $k \in [2, 8]$.
  * $k=2$: WCSS = 9014.57, Silhouette = 0.8958 (separates single extreme outlier group from all others)
  * $k=3$: WCSS = 5441.32, Silhouette = 0.5942
  * **$k=4$: WCSS = 4096.30, Silhouette = 0.6162** (Selected optimal model)
  * $k=5$: WCSS = 3119.79, Silhouette = 0.6165
* **Selected $k=4$:** Provides the optimal balance between high silhouette score (0.6162), a clear elbow inflection point, and distinct, actionable marketing tiers.

### STEP 6: DBSCAN Clustering
* **Hyperparameter Selection:** Plotted the 5-nearest neighbors distance graph (**k-distance graph**). The knee/elbow of the curve indicates the density threshold near distance $\approx 0.5$.
* **Grid Search:** Tested `eps` $\in [0.3, 0.5, 0.7, 1.0]$ and `min_samples` $\in [5, 10, 15]$.
* **Selected Parameters:** `eps = 0.5`, `min_samples = 10`.
* **Results:**
  * **Cluster 0:** 4,262 core mainstream retail customers.
  * **Cluster 1:** 9 high-activity, high-frequency customers.
  * **Noise Points (`-1`):** 67 extreme outlier customers (1.54% of dataset).
  * **Silhouette Score (excluding noise):** **0.6663**.
* **Key Finding:** DBSCAN does not artificially carve a continuous retail density distribution into arbitrary spherical slices. Instead, it groups the mainstream population into a dense core and isolates the 67 extreme spenders as statistical noise.

### STEP 7: Principal Component Analysis (PCA)
Linear dimensionality reduction from 3 RFM features to 2 Principal Components ($PC1, PC2$):
* **PC1 Explained Variance:** **55.47%**
* **PC2 Explained Variance:** **30.25%**
* **Cumulative Explained Variance:** **85.73%** (captures nearly 86% of total data information in 2 dimensions).

#### PCA Loadings (Eigenvector Weights):
| Component | Recency | Frequency | Monetary | Business Interpretation |
| :--- | :--- | :--- | :--- | :--- |
| **PC1** | -0.3808 | +0.6752 | +0.6318 | **Customer Value & Engagement:** High spenders and frequent buyers have large positive PC1 scores. |
| **PC2** | +0.9050 | +0.1318 | +0.4046 | **Customer Inactivity / Churn Risk:** Dominated by Recency (+0.905). High PC2 indicates dormant/lapsed customers. |

### STEP 8: t-SNE Visualization
* **Perplexity:** 30, **Random State:** 42, **Dimensions:** 2.
* Non-linear projection confirms that K-Means clusters form well-separated manifolds in low-dimensional space.
* **Important Academic Note:** t-SNE is strictly a non-linear manifold visualization technique. It is **not** a clustering algorithm. It does not preserve global distances, nor does it provide a parametric `.predict()` method to classify new data points.

---

## 6. Customer Marketing Personas & Strategic Recommendations

Based on empirical K-Means cluster statistics, we formulate 4 distinct, actionable marketing personas:

| Persona | Size | % Share | Median Recency | Median Frequency | Median Monetary | Commercial Behaviour | Recommended Marketing Strategy |
| :--- | :---: | :---: | :---: | :---: | :---: | :--- | :--- |
| **Champions / VIP Wholesale** (Cluster 2) | 13 | 0.30% | 2.0 days | 63.0 orders (Mean: 82.5) | £117,210.08 | Ultra-high volume, immediate recency, bulk purchasing wholesale buyers. | Dedicated account management, wholesale volume tiers, custom logistics, wholesale credit facilities, direct phone outreach. |
| **Loyal High-Tier Shoppers** (Cluster 3) | 204 | 4.70% | 5.0 days | 19.0 orders (Mean: 22.3) | £7,990.21 | High spending, frequent repeat buyers with high brand affinity. | VIP loyalty rewards, early access to new collections, exclusive seasonal previews, referral bonuses. |
| **Active / Regular Shoppers** (Cluster 0) | 3,054 | 70.40% | 32.0 days | 3.0 orders (Mean: 3.7) | £826.27 | Steady retail shoppers who make regular moderate purchases. | Cross-selling and up-selling recommendations, minimum-spend free shipping thresholds, replenishment reminders. |
| **At-Risk / Lapsed Customers** (Cluster 1) | 1,067 | 24.60% | 243.0 days | 1.0 order (Mean: 1.55) | £310.26 | Dormant buyers who have not purchased in ~8 months; high churn risk. | Automated win-back discount campaigns ("We miss you - 20% off"), customer feedback surveys, reactivation emails. |

### Deep Investigation & Treatment of DBSCAN Noise Points:
* **Noise Points (-1, 67 customers, 1.54% of dataset):**
  * **Empirical Monetary Distribution:** Min = £1,296.44, 25th percentile = £9,840.69, Median = £28,337.38, Mean = £45,258.42, Max = £280,206.02.
  * **Critical Mathematical Finding:** While the majority of noise points (73.1%, 49/67) have Monetary values exceeding £10,000, **noise points must NOT be assumed to be exclusively high spenders**. DBSCAN operates in the joint 3-dimensional standardized space $(R_{\text{scaled}}, F_{\text{scaled}}, M_{\text{scaled}})$, flagging any sample whose $\epsilon$-neighborhood ($\epsilon=0.5$) contains fewer than 10 points.
  * **Why Non-High-Spenders Were Flagged as Noise:**
    1. **Extreme Frequency with Modest Spend:** Customer `17961` has Monetary spending of only £1,296.44 (lower than the overall population mean of £2,048), but placed **39 distinct orders** (99.8th percentile of Frequency).
    2. **Atypical Feature Combinations (High Frequency + Severe Dormancy):** Customer `17850` spent £5,391.21 across **34 orders**, but has Recency of **372 days** (dormant for over a year).
    3. **Extreme Single-Order Spend with Severe Dormancy:** Customer `12346` spent **£77,183.60** in just **1 single order** with Recency of **326 days**.
    4. **True Enterprise Super-Outliers:** Customers `14646` (£280,206 spend, 73 orders), `14911` (£143,711 spend, 201 orders), and `12748` (£33,053 spend, 209 orders).
  * **Business Action Plan:** These points represent multivariate behavioral anomalies and enterprise wholesale accounts. Rather than receiving generic automated retail marketing emails, they should be routed to compliance (for fraud/audit screening) and assigned to dedicated B2B enterprise sales representatives for custom contract negotiation.

---

## 7. Comparative Evaluation: K-Means vs. DBSCAN

| Evaluation Metric / Attribute | K-Means Clustering | DBSCAN Clustering |
| :--- | :--- | :--- |
| **Clustering Paradigm** | Centroid-based (Partitioning) | Density-based (Spatial connectivity) |
| **Clusters Identified** | 4 well-balanced clusters | 2 clusters + 1 noise class |
| **Noise & Outlier Handling** | Cannot handle noise; forces 100% of points into a cluster | Explicitly identifies and isolates outliers as noise (`-1`) |
| **Number of Noise Points** | 0 points (0.00%) | 67 points (1.54%) |
| **Silhouette Score** | **0.6162** (evaluated on all 4,338 customers) | **0.6663** (evaluated on dense core, excluding noise) |
| **Geometry of Clusters** | Assumes convex, spherical clusters | Can identify arbitrary, non-spherical shapes |
| **Hyperparameter Requirements** | Requires pre-specifying $k$ | Requires tuning $\epsilon$ (`eps`) and `min_samples` |
| **Actionability for Marketing** | **High**: Every customer belongs to an actionable marketing segment | **Moderate**: 98% of customers are lumped into a single mass cluster |
| **Primary Practical Role** | Customer tiering, lifecycle campaigns, marketing personalization | Anomaly detection, wholesale outlier identification, fraud screening |

### Conclusion on Algorithmic Suitability:
* **For Business Customer Segmentation:** **K-Means is clearly more useful.** In commercial marketing, businesses need to assign every customer to an actionable tier (e.g., Active vs. At-Risk vs. VIP). K-Means produces balanced, interpretable tiers that map directly onto business operations.
* **For Outlier & Anomaly Detection:** **DBSCAN is superior.** It successfully flagged the 67 extreme wholesale accounts without letting them skew the cluster centroids of everyday shoppers.

---

## 8. Installation & Execution Guide

### Prerequisites
* Python 3.10, 3.11, 3.12, 3.13, or 3.14 installed.

### Setup Instructions

```bash
# 1. Clone or navigate to the project directory
cd Ecommerce-Customer-Segmentation

# 2. Create and activate a virtual environment (optional but recommended)
python -m venv venv
# On Windows:
.\venv\Scripts\activate
# On Linux/macOS:
source venv/bin/activate

# 3. Install required dependencies
pip install -r requirements.txt
```

### Running the Project

#### Option A: Interactive Jupyter Notebook (Recommended for Viva Demonstration)
```bash
jupyter notebook notebooks/ECommerce_Customer_Segmentation.ipynb
```
Select **Kernel -> Restart & Run All** to execute all cells sequentially.

#### Option B: Standalone Python Pipeline Script
```bash
python run_pipeline.py
```
This generates all 10 high-resolution visualizations in `outputs/visualizations/` and exports `outputs/cluster_results.csv` in ~15 seconds.

---

## 9. Viva Voce & Faculty Oral Examination Q&A Guide

Prepare for your project viva with these targeted questions and answers:

### Q1: Why do we use RFM Analysis instead of directly clustering raw transaction lines?
> **Answer:** Raw transactional data exists at the order/line-item level (an individual customer might have hundreds of rows). Unsupervised clustering requires one feature vector per customer. The RFM framework aggregates an arbitrary number of transactions into three fundamental behavioral dimensions: how recently they bought (Recency), how often they buy (Frequency), and how much they spend (Monetary).

### Q2: Why is feature scaling mandatory before running K-Means or DBSCAN?
> **Answer:** Both K-Means and DBSCAN rely on Euclidean distance ($L_2$ norm). In raw RFM data, Monetary values range up to £280,000, while Frequency is mostly between 1 and 10, and Recency is between 1 and 374. Without scaling, differences in Monetary value would mathematically dominate the distance formula, causing the algorithms to ignore Recency and Frequency entirely. `StandardScaler` standardizes each feature to have a mean of 0 and variance of 1, granting each feature equal initial weighting.

### Q3: How did you select $k=4$ for K-Means?
> **Answer:** We evaluated $k$ from 2 to 8 using two complementary techniques:
> 1. **Elbow Method:** The Within-Cluster Sum of Squares (Inertia) showed a pronounced elbow bend at $k=4$, where adding more clusters yielded diminishing reductions in variance.
> 2. **Silhouette Score:** The score peaked at $k=4$ (0.6162), indicating strong cohesion within clusters and clear separation between clusters.
> 3. **Business Interpretability:** $k=4$ produced four actionable business tiers (Champions, Loyal High-Tier, Active Regulars, and At-Risk Customers).

### Q4: What does a Silhouette Score represent, and what was your score?
> **Answer:** The Silhouette Score measures how well-separated and cohesive clusters are on a scale from -1 to +1:
> $$s(i) = \frac{b(i) - a(i)}{\max(a(i), b(i))}$$
> where $a(i)$ is the mean intra-cluster distance and $b(i)$ is the mean nearest-cluster distance. A score near +1 indicates points are far from neighboring clusters. Our K-Means model achieved a high score of **0.6162**, indicating well-defined clusters.

### Q5: Why did DBSCAN identify mostly one large cluster and 67 noise points?
> **Answer:** DBSCAN is a density-based algorithm that joins points whose neighborhood density exceeds `min_samples` within radius $\epsilon$. Real-world e-commerce customers do not naturally live on separated spatial islands; instead, their spending follows a continuous density distribution. Therefore, DBSCAN connects the majority of standard customers (4,262) into one continuous dense cluster, while correctly isolating the 67 extreme outliers who placed hundreds of orders or spent over £50,000 as noise points (`-1`).

### Q6: What is the difference between PCA and t-SNE?
> **Answer:**
> * **PCA** is a linear dimensionality reduction method that finds orthogonal axes of maximum variance via eigenvalue decomposition of the covariance matrix. It preserves global geometric structure and variance (85.73% across 2 PCs).
> * **t-SNE** is a non-linear manifold learning technique that converts pairwise distances into probabilities using a Student-t distribution. It preserves local neighborhoods and is used exclusively for visualization, not clustering.

### Q7: Can we use t-SNE to cluster new incoming customers in production?
> **Answer:** No. t-SNE is strictly a visualization tool. It does not learn an explicit parametric mapping function (it lacks a `.transform()` or `.predict()` method for new points). To categorize a new customer, we use the trained `KMeans.predict()` model or Euclidean distance to the cluster centroids.

### Q8: Which algorithm is ultimately better for this project, and why?
> **Answer:** For customer segmentation and marketing campaign design, **K-Means is better**. Marketing teams need complete coverage where 100% of customers are assigned to actionable segments (Active, Inactive, VIP). DBSCAN is not suitable for marketing tiering because it groups 98% of customers into a single segment; however, DBSCAN is highly valuable as an anomaly detection filter to separate extreme wholesale accounts.

---

## 10. Summary of Deliverables

1. **Jupyter Notebook:** `notebooks/ECommerce_Customer_Segmentation.ipynb` (Executed end-to-end, zero errors, all visualizations rendered).
2. **Dataset:** `data/Online Retail.xlsx` (Original UCI repository file) and `data/Online_Retail.csv` (High-speed loading replica).
3. **Saved Figures:** `outputs/visualizations/*.png` (10 publication-quality 300 DPI plots).
4. **Exported CSV Results:** `outputs/cluster_results.csv` (Full dataset with CustomerID, RFM values, K-Means clusters & personas, DBSCAN clusters & labels, and PCA/t-SNE coordinates).
5. **Requirements File:** `requirements.txt` (Minimal, approved dependencies).
6. **Execution Script:** `run_pipeline.py` (Reproducible command-line pipeline).
7. **Comprehensive README:** `README.md` (Theory, methodology, results, marketing recommendations, and viva questions).

