# Customer Segmentation Project

A machine learning project that groups mall customers into meaningful segments based on demographic information, annual income, and spending behavior. The analysis uses unsupervised learning with the K-Means clustering algorithm.

## Project Files

- `cusotmer_segmentation.ipynb` — Jupyter Notebook containing data exploration, preprocessing, clustering, evaluation, visualization, and customer category assignment
- `Mall_Customers.csv` — Customer dataset used for the analysis

## Dataset

The project uses the Mall Customers dataset, which contains **200 customer records** and the following columns:

| Column | Description |
|---|---|
| `CustomerID` | Unique identifier for each customer |
| `Gender` | Customer gender |
| `Age` | Customer age in years |
| `Annual Income (k$)` | Annual income measured in thousands of dollars |
| `Spending Score (1-100)` | Spending behavior score, where higher values indicate greater spending activity |

The dataset does not contain missing values. `Gender` is converted to numerical values, with Male represented as `1` and Female represented as `0`. `CustomerID` is removed because it is an identifier and does not provide useful information for clustering.

## Model: K-Means Clustering

K-Means is an unsupervised machine learning algorithm that divides data points into a selected number of groups called clusters. Each customer is assigned to the cluster whose center, or centroid, is closest to the customer's standardized feature values.

The notebook follows these steps:

1. Load and inspect the customer data.
2. Check for missing values and calculate descriptive statistics.
3. Convert the gender column into numerical values.
4. Remove the `CustomerID` column.
5. Standardize the features using `StandardScaler` so that variables with different scales contribute fairly.
6. Train K-Means models for cluster counts from 2 through 12.
7. Compare cluster quality using inertia and silhouette score.
8. Select six clusters for the final customer segmentation model.
9. Analyze cluster averages and visualize the resulting customer groups.

The final model is created with:

```python
KMeans(n_clusters=6, random_state=42)
```

## Choosing the Number of Clusters

The notebook evaluates different values of `k` using:

- **Elbow method:** Examines the decrease in inertia as the number of clusters increases. The selected value is based on the point where adding more clusters provides diminishing improvement.
- **Silhouette score:** Measures how well each customer fits within its assigned cluster compared with other clusters. A higher score generally indicates better-separated groups.

Both methods are used together to support the selection of six customer clusters.

## Visualizations and Analysis

The notebook includes:

- Feature correlation heatmap
- Elbow chart showing cluster inertia
- Cluster size counts
- Average feature values for each cluster
- Annual income versus spending score scatter plot
- Two-dimensional PCA visualization of the clusters
- Income and age cluster comparison
- Gender and income cluster comparison

Principal Component Analysis (PCA) is used to reduce the standardized feature space to two dimensions for visualization. PCA helps display the clusters on a two-dimensional chart, while the original standardized features are used for the primary clustering analysis.

## Customer Categories

After analyzing the average characteristics of each cluster, the notebook assigns descriptive customer categories:

- **Sensible**
- **Careful Saver**
- **Impetuous Spender**
- **Top VIP Spenders**
- **Careless Spender**
- **Average Spender**

These labels are descriptive interpretations of the cluster characteristics and are not directly learned by K-Means. They help make the numerical clusters easier to understand for marketing and customer analysis.

## Requirements

Run the notebook with Python and install the following libraries if needed:

```bash
pip install pandas numpy matplotlib seaborn scikit-learn jupyter
```

Start Jupyter Notebook with:

```bash
jupyter notebook
```

Then open `cusotmer_segmentation.ipynb` and run the cells in order.

## Use Cases

Customer segmentation can help businesses:

- Identify high-value customers
- Design targeted marketing campaigns
- Understand spending behavior
- Create personalized offers
- Separate cautious customers from frequent or high-value spenders

> This project is intended for educational and exploratory analysis. The resulting segments should be validated with business knowledge and additional customer data before being used for real marketing decisions.
