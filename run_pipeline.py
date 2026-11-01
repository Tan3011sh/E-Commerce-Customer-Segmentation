"""
E-Commerce Customer Segmentation Pipeline Script
Generates all visualizations and exports cluster results.
"""

import os
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.neighbors import NearestNeighbors

# Set plotting style
sns.set_theme(style="whitegrid", palette="tab10")
plt.rcParams['font.sans-serif'] = 'DejaVu Sans'
plt.rcParams['axes.edgecolor'] = '#cccccc'
plt.rcParams['axes.linewidth'] = 0.8

VIS_DIR = "outputs/visualizations"
os.makedirs(VIS_DIR, exist_ok=True)
os.makedirs("outputs", exist_ok=True)

print("=" * 70)
print("STARTING E-COMMERCE CUSTOMER SEGMENTATION PIPELINE")
print("=" * 70)

# ---------------------------------------------------------
# STEP 1: DATA LOADING & EXPLORATION
# ---------------------------------------------------------
print("\n[STEP 1] Loading Dataset...")
data_path = "data/Online_Retail.csv"
if not os.path.exists(data_path):
    data_path = "data/Online Retail.xlsx"

if data_path.endswith('.csv'):
    df_raw = pd.read_csv(data_path)
else:
    df_raw = pd.read_excel(data_path)

print(f"Dataset successfully loaded. Raw Shape: {df_raw.shape}")
print(f"Columns: {list(df_raw.columns)}")

# EDA Visualizations
print("Generating EDA Plots...")

# 1. Top Countries by transaction count
fig, ax = plt.subplots(figsize=(10, 5))
top_countries = df_raw['Country'].value_counts().head(10)
sns.barplot(x=top_countries.values, y=top_countries.index, hue=top_countries.index, palette="Blues_r", legend=False, ax=ax)
ax.set_title("Top 10 Countries by Number of Transactions", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Number of Transactions", fontsize=11)
ax.set_ylabel("Country", fontsize=11)
for i, v in enumerate(top_countries.values):
    ax.text(v + 1500, i, f"{v:,}", va='center', fontsize=9, fontweight='medium')
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/01_eda_top_countries.png", dpi=300)
plt.close()
print("Saved 01_eda_top_countries.png")

# ---------------------------------------------------------
# STEP 2: DATA PREPROCESSING
# ---------------------------------------------------------
print("\n[STEP 2] Preprocessing Data...")
# 1. Drop missing CustomerID
initial_rows = len(df_raw)
df_clean = df_raw.dropna(subset=['CustomerID']).copy()
df_clean['CustomerID'] = df_clean['CustomerID'].astype(int)
print(f"1. Removed {initial_rows - len(df_clean):,} rows with missing CustomerID.")

# 2. Remove cancellations
cancellation_mask = df_clean['InvoiceNo'].astype(str).str.startswith('C')
n_cancelled = cancellation_mask.sum()
df_clean = df_clean[~cancellation_mask]
print(f"2. Removed {n_cancelled:,} cancellation transactions (InvoiceNo starting with 'C').")

# 3. Handle non-positive quantities and unit prices
invalid_qty_price = (df_clean['Quantity'] <= 0) | (df_clean['UnitPrice'] <= 0)
n_invalid = invalid_qty_price.sum()
df_clean = df_clean[~invalid_qty_price]
print(f"3. Removed {n_invalid:,} transactions with non-positive Quantity or UnitPrice.")

# 4. Remove duplicate records
n_dups = df_clean.duplicated().sum()
df_clean = df_clean.drop_duplicates()
print(f"4. Removed {n_dups:,} duplicate transaction rows.")

# 5. Datetime conversion
df_clean['InvoiceDate'] = pd.to_datetime(df_clean['InvoiceDate'])
df_clean['TotalAmount'] = df_clean['Quantity'] * df_clean['UnitPrice']

print(f"Cleaned dataset shape: {df_clean.shape}")
print(f"Unique customers: {df_clean['CustomerID'].nunique():,}")
print(f"Date range: {df_clean['InvoiceDate'].min()} to {df_clean['InvoiceDate'].max()}")

# Monthly trend plot
df_clean['YearMonth'] = df_clean['InvoiceDate'].dt.to_period('M')
monthly_txns = df_clean.groupby('YearMonth').size()

fig, ax = plt.subplots(figsize=(11, 5))
monthly_txns.plot(kind='line', marker='o', color='#1f77b4', linewidth=2.5, markersize=7, ax=ax)
ax.set_title("Monthly Cleaned Transaction Volume (Dec 2010 - Dec 2011)", fontsize=14, fontweight='bold', pad=12)
ax.set_xlabel("Year-Month", fontsize=11)
ax.set_ylabel("Number of Orders/Lines", fontsize=11)
ax.grid(True, linestyle='--', alpha=0.6)
plt.xticks(range(len(monthly_txns)), [str(ym) for ym in monthly_txns.index], rotation=45)
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/02_eda_monthly_transactions.png", dpi=300)
plt.close()
print("Saved 02_eda_monthly_transactions.png")

# ---------------------------------------------------------
# STEP 3: RFM ANALYSIS
# ---------------------------------------------------------
print("\n[STEP 3] Performing RFM Analysis...")
# Reference date is set to 1 day after the latest transaction date in the dataset
reference_date = df_clean['InvoiceDate'].max() + pd.Timedelta(days=1)
print(f"Reference Date for Recency Calculation: {reference_date}")

rfm = df_clean.groupby('CustomerID').agg({
    'InvoiceDate': lambda x: (reference_date - x.max()).days,
    'InvoiceNo': 'nunique',
    'TotalAmount': 'sum'
}).reset_index()

rfm.columns = ['CustomerID', 'Recency', 'Frequency', 'Monetary']
rfm['Monetary'] = rfm['Monetary'].round(2)

print("RFM Summary Statistics:")
print(rfm[['Recency', 'Frequency', 'Monetary']].describe().T)

# RFM Distributions Plot
fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))

sns.histplot(rfm['Recency'], kde=True, color='#2ca02c', ax=axes[0], bins=30)
axes[0].set_title("Recency Distribution (Days)", fontsize=12, fontweight='bold')
axes[0].set_xlabel("Recency (Days)")
axes[0].set_ylabel("Customer Count")

sns.histplot(rfm['Frequency'], kde=True, color='#1f77b4', ax=axes[1], bins=30)
axes[1].set_title("Frequency Distribution (Orders)", fontsize=12, fontweight='bold')
axes[1].set_xlabel("Frequency (Orders)")
axes[1].set_ylabel("Customer Count")

sns.histplot(rfm['Monetary'], kde=True, color='#d62728', ax=axes[2], bins=30)
axes[2].set_title("Monetary Distribution (£)", fontsize=12, fontweight='bold')
axes[2].set_xlabel("Monetary (£)")
axes[2].set_ylabel("Customer Count")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/03_rfm_distributions.png", dpi=300)
plt.close()
print("Saved 03_rfm_distributions.png")

# ---------------------------------------------------------
# STEP 4: FEATURE SCALING
# ---------------------------------------------------------
print("\n[STEP 4] Feature Scaling with StandardScaler...")
scaler = StandardScaler()
rfm_scaled = scaler.fit_transform(rfm[['Recency', 'Frequency', 'Monetary']])
print("Scaled RFM features shape:", rfm_scaled.shape)
print("Feature Means (approx 0):", rfm_scaled.mean(axis=0).round(4))
print("Feature Standard Deviations (approx 1):", rfm_scaled.std(axis=0).round(4))

# ---------------------------------------------------------
# STEP 5: K-MEANS CLUSTERING
# ---------------------------------------------------------
print("\n[STEP 5] K-Means Clustering...")
k_range = range(2, 9)
wcss = []
silhouette_scores_km = []

for k in k_range:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    labels = km.fit_predict(rfm_scaled)
    wcss.append(km.inertia_)
    sil = silhouette_score(rfm_scaled, labels)
    silhouette_scores_km.append(sil)
    print(f"  k={k} -> WCSS (Inertia): {km.inertia_:.2f}, Silhouette Score: {sil:.4f}")

# Plot Elbow and Silhouette curves
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Elbow Curve
axes[0].plot(list(k_range), wcss, marker='o', color='#1f77b4', linewidth=2, markersize=8)
axes[0].set_title("Elbow Method for Optimal k", fontsize=13, fontweight='bold')
axes[0].set_xlabel("Number of Clusters (k)", fontsize=11)
axes[0].set_ylabel("Within-Cluster Sum of Squares (WCSS)", fontsize=11)
axes[0].axvline(x=4, color='#d62728', linestyle='--', label='Optimal k = 4')
axes[0].legend()

# Silhouette Scores
axes[1].plot(list(k_range), silhouette_scores_km, marker='s', color='#2ca02c', linewidth=2, markersize=8)
axes[1].set_title("Silhouette Scores for Different k", fontsize=13, fontweight='bold')
axes[1].set_xlabel("Number of Clusters (k)", fontsize=11)
axes[1].set_ylabel("Silhouette Score", fontsize=11)
axes[1].axvline(x=4, color='#d62728', linestyle='--', label='Selected k = 4 (Score = 0.616)')
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/04_kmeans_elbow_and_silhouette.png", dpi=300)
plt.close()
print("Saved 04_kmeans_elbow_and_silhouette.png")

# Fit K-Means with optimal k = 4
optimal_k = 4
kmeans_final = KMeans(n_clusters=optimal_k, random_state=42, n_init=10)
rfm['KMeans_Cluster'] = kmeans_final.fit_predict(rfm_scaled)

print(f"\nK-Means (k={optimal_k}) Cluster Distribution:")
print(rfm['KMeans_Cluster'].value_counts().sort_index())

# Scatter plots for K-Means clusters
fig, axes = plt.subplots(1, 3, figsize=(18, 5))
palette_km = sns.color_palette("tab10", optimal_k)

sns.scatterplot(data=rfm, x='Recency', y='Frequency', hue='KMeans_Cluster', palette=palette_km, ax=axes[0], alpha=0.7, s=40)
axes[0].set_title("Recency vs. Frequency (K-Means)", fontsize=12, fontweight='bold')

sns.scatterplot(data=rfm, x='Recency', y='Monetary', hue='KMeans_Cluster', palette=palette_km, ax=axes[1], alpha=0.7, s=40)
axes[1].set_title("Recency vs. Monetary (K-Means)", fontsize=12, fontweight='bold')

sns.scatterplot(data=rfm, x='Frequency', y='Monetary', hue='KMeans_Cluster', palette=palette_km, ax=axes[2], alpha=0.7, s=40)
axes[2].set_title("Frequency vs. Monetary (K-Means)", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/05_kmeans_clusters_rfm_scatter.png", dpi=300)
plt.close()
print("Saved 05_kmeans_clusters_rfm_scatter.png")

# ---------------------------------------------------------
# STEP 6: DBSCAN CLUSTERING
# ---------------------------------------------------------
print("\n[STEP 6] DBSCAN Clustering...")

# k-distance graph to guide eps selection (k = 2 * dimensions - 1 = 5)
k_neighbors = 5
nn = NearestNeighbors(n_neighbors=k_neighbors)
nn.fit(rfm_scaled)
distances, _ = nn.kneighbors(rfm_scaled)
distances = np.sort(distances[:, k_neighbors - 1])

fig, ax = plt.subplots(figsize=(8, 4.5))
ax.plot(distances, color='#9467bd', linewidth=2)
ax.set_title(f"K-Distance Graph (k={k_neighbors}) for DBSCAN eps Selection", fontsize=13, fontweight='bold')
ax.set_xlabel("Sorted Data Points", fontsize=11)
ax.set_ylabel(f"{k_neighbors}-th Nearest Neighbor Distance", fontsize=11)
ax.axhline(y=0.5, color='#d62728', linestyle='--', label='Selected eps = 0.5 (Elbow knee)')
ax.legend()
plt.tight_layout()
plt.savefig(f"{VIS_DIR}/06_dbscan_k_distance_graph.png", dpi=300)
plt.close()
print("Saved 06_dbscan_k_distance_graph.png")

# Apply DBSCAN with eps=0.5, min_samples=10
dbscan = DBSCAN(eps=0.5, min_samples=10)
rfm['DBSCAN_Cluster'] = dbscan.fit_predict(rfm_scaled)

n_db_clusters = len(set(rfm['DBSCAN_Cluster'])) - (1 if -1 in rfm['DBSCAN_Cluster'] else 0)
n_db_noise = (rfm['DBSCAN_Cluster'] == -1).sum()
pct_noise = (n_db_noise / len(rfm)) * 100

print(f"DBSCAN identified {n_db_clusters} cluster(s) and {n_db_noise} noise points ({pct_noise:.2f}%).")
print(rfm['DBSCAN_Cluster'].value_counts().sort_index())

# Calculate Silhouette score for DBSCAN excluding noise points
mask_db = rfm['DBSCAN_Cluster'] != -1
if len(set(rfm.loc[mask_db, 'DBSCAN_Cluster'])) >= 2:
    db_sil = silhouette_score(rfm_scaled[mask_db], rfm.loc[mask_db, 'DBSCAN_Cluster'])
    print(f"DBSCAN Silhouette Score (excluding noise): {db_sil:.4f}")
else:
    db_sil = None
    print("DBSCAN Silhouette Score: Only 1 non-noise cluster identified; silhouette score is undefined.")

# DBSCAN scatter plot
fig, axes = plt.subplots(1, 2, figsize=(14, 5))
# Color map with gray for noise (-1)
db_colors = { -1: '#7f7f7f', 0: '#1f77b4', 1: '#ff7f0e' }

sns.scatterplot(
    data=rfm, x='Recency', y='Monetary',
    hue='DBSCAN_Cluster', palette=db_colors,
    style='DBSCAN_Cluster', markers={-1: 'X', 0: 'o', 1: 's'},
    alpha=0.8, s=45, ax=axes[0]
)
axes[0].set_title("Recency vs. Monetary (DBSCAN)", fontsize=12, fontweight='bold')

sns.scatterplot(
    data=rfm, x='Frequency', y='Monetary',
    hue='DBSCAN_Cluster', palette=db_colors,
    style='DBSCAN_Cluster', markers={-1: 'X', 0: 'o', 1: 's'},
    alpha=0.8, s=45, ax=axes[1]
)
axes[1].set_title("Frequency vs. Monetary (DBSCAN)", fontsize=12, fontweight='bold')

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/07_dbscan_clusters_rfm_scatter.png", dpi=300)
plt.close()
print("Saved 07_dbscan_clusters_rfm_scatter.png")

# ---------------------------------------------------------
# STEP 7: PRINCIPAL COMPONENT ANALYSIS (PCA)
# ---------------------------------------------------------
print("\n[STEP 7] Principal Component Analysis (PCA)...")
pca = PCA(n_components=2, random_state=42)
pca_transformed = pca.fit_transform(rfm_scaled)
rfm['PC1'] = pca_transformed[:, 0]
rfm['PC2'] = pca_transformed[:, 1]

var_ratio = pca.explained_variance_ratio_
cum_var = var_ratio.sum()
print(f"PC1 Explained Variance: {var_ratio[0]*100:.2f}%")
print(f"PC2 Explained Variance: {var_ratio[1]*100:.2f}%")
print(f"Cumulative Explained Variance: {cum_var*100:.2f}%")

loadings = pd.DataFrame(pca.components_, columns=['Recency', 'Frequency', 'Monetary'], index=['PC1', 'PC2'])
print("PCA Loadings (Eigenvectors):")
print(loadings.round(4))

# PCA visualization: K-Means vs DBSCAN
fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(data=rfm, x='PC1', y='PC2', hue='KMeans_Cluster', palette=palette_km, ax=axes[0], alpha=0.8, s=45)
axes[0].set_title(f"PCA 2D Projection: K-Means Clusters (k={optimal_k})", fontsize=12, fontweight='bold')
axes[0].set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)")
axes[0].set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)")

sns.scatterplot(data=rfm, x='PC1', y='PC2', hue='DBSCAN_Cluster', palette=db_colors, ax=axes[1], alpha=0.8, s=45)
axes[1].set_title("PCA 2D Projection: DBSCAN Clusters & Outliers", fontsize=12, fontweight='bold')
axes[1].set_xlabel(f"PC1 ({var_ratio[0]*100:.1f}% Variance)")
axes[1].set_ylabel(f"PC2 ({var_ratio[1]*100:.1f}% Variance)")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/08_pca_clusters_comparison.png", dpi=300)
plt.close()
print("Saved 08_pca_clusters_comparison.png")

# ---------------------------------------------------------
# STEP 8: t-SNE VISUALIZATION
# ---------------------------------------------------------
print("\n[STEP 8] t-SNE Non-Linear Dimensionality Reduction...")
tsne = TSNE(n_components=2, perplexity=30, random_state=42, n_iter_without_progress=300)
tsne_transformed = tsne.fit_transform(rfm_scaled)
rfm['tSNE1'] = tsne_transformed[:, 0]
rfm['tSNE2'] = tsne_transformed[:, 1]
print("t-SNE completed successfully.")

fig, axes = plt.subplots(1, 2, figsize=(15, 6))

sns.scatterplot(data=rfm, x='tSNE1', y='tSNE2', hue='KMeans_Cluster', palette=palette_km, ax=axes[0], alpha=0.8, s=35)
axes[0].set_title("t-SNE 2D Manifold Visualization: K-Means Clusters", fontsize=12, fontweight='bold')
axes[0].set_xlabel("t-SNE Dimension 1")
axes[0].set_ylabel("t-SNE Dimension 2")

sns.scatterplot(data=rfm, x='tSNE1', y='tSNE2', hue='DBSCAN_Cluster', palette=db_colors, ax=axes[1], alpha=0.8, s=35)
axes[1].set_title("t-SNE 2D Manifold Visualization: DBSCAN Clusters", fontsize=12, fontweight='bold')
axes[1].set_xlabel("t-SNE Dimension 1")
axes[1].set_ylabel("t-SNE Dimension 2")

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/09_tsne_clusters_comparison.png", dpi=300)
plt.close()
print("Saved 09_tsne_clusters_comparison.png")

# ---------------------------------------------------------
# STEP 9: CUSTOMER MARKETING PERSONAS
# ---------------------------------------------------------
print("\n[STEP 9] Customer Marketing Personas Profiling...")

km_profile = rfm.groupby('KMeans_Cluster').agg(
    CustomerCount=('CustomerID', 'count'),
    Mean_Recency=('Recency', 'mean'),
    Median_Recency=('Recency', 'median'),
    Mean_Frequency=('Frequency', 'mean'),
    Median_Frequency=('Frequency', 'median'),
    Mean_Monetary=('Monetary', 'mean'),
    Median_Monetary=('Monetary', 'median')
).reset_index()

print("K-Means Cluster Profiling:")
print(km_profile.round(2))

# Persona assignment based on actual statistics
# Cluster 2: Very high frequency (82.5), high monetary (£127,188), low recency (7.4 days) -> Champions / VIP Wholesale
# Cluster 3: High frequency (22.4), high monetary (£12,690), low recency (15.5 days) -> Loyal / High-Tier Shoppers
# Cluster 0: Moderate frequency (3.7), moderate monetary (£1,354), recent (43.7 days) -> Active / Regular Shoppers
# Cluster 1: Low frequency (1.5), low monetary (£479), high recency (248 days) -> At-Risk / Lapsed Customers

persona_map = {
    0: "Active / Regular Shoppers",
    1: "At-Risk / Lapsed Customers",
    2: "Champions / VIP Wholesale",
    3: "Loyal / High-Tier Shoppers"
}
rfm['KMeans_Persona'] = rfm['KMeans_Cluster'].map(persona_map)

# DBSCAN Labels
db_label_map = {
    -1: "Noise / Extreme Outliers",
    0: "Core Mainstream Retail Shoppers",
    1: "High-Activity Sub-Cluster"
}
rfm['DBSCAN_Label'] = rfm['DBSCAN_Cluster'].map(db_label_map)

# Persona Summary Visualization
fig, axes = plt.subplots(1, 3, figsize=(16, 5))

sns.barplot(data=rfm, x='KMeans_Persona', y='Recency', hue='KMeans_Persona', estimator=np.median, errorbar=None, palette='viridis', legend=False, ax=axes[0])
axes[0].set_title("Median Recency by Persona (Days)", fontsize=12, fontweight='bold')
axes[0].tick_params(axis='x', rotation=30)

sns.barplot(data=rfm, x='KMeans_Persona', y='Frequency', hue='KMeans_Persona', estimator=np.median, errorbar=None, palette='viridis', legend=False, ax=axes[1])
axes[1].set_title("Median Frequency by Persona (Orders)", fontsize=12, fontweight='bold')
axes[1].tick_params(axis='x', rotation=30)

sns.barplot(data=rfm, x='KMeans_Persona', y='Monetary', hue='KMeans_Persona', estimator=np.median, errorbar=None, palette='viridis', legend=False, ax=axes[2])
axes[2].set_title("Median Monetary by Persona (£)", fontsize=12, fontweight='bold')
axes[2].tick_params(axis='x', rotation=30)

plt.tight_layout()
plt.savefig(f"{VIS_DIR}/10_customer_personas_profile.png", dpi=300)
plt.close()
print("Saved 10_customer_personas_profile.png")

# ---------------------------------------------------------
# STEP 10: COMPARISON & EXPORT RESULTS
# ---------------------------------------------------------
print("\n[STEP 10] Comparison and Exporting Cluster Results...")

# Save cluster_results.csv
output_csv = "outputs/cluster_results.csv"
export_cols = [
    'CustomerID', 'Recency', 'Frequency', 'Monetary',
    'KMeans_Cluster', 'KMeans_Persona',
    'DBSCAN_Cluster', 'DBSCAN_Label',
    'PC1', 'PC2', 'tSNE1', 'tSNE2'
]
rfm[export_cols].to_csv(output_csv, index=False)
print(f"Exported customer cluster results to {output_csv} ({len(rfm)} rows)")

print("\n" + "=" * 70)
print("PIPELINE EXECUTION COMPLETE! ALL ARTIFACTS GENERATED SUCCESSFULLY.")
print("=" * 70)
