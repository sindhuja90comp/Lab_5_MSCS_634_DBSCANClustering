# Lab 5: DBSCAN Clustering  
**MSCS 634**

## Purpose of the Lab

The purpose of this lab is to explore and compare clustering techniques, focusing on DBSCAN (Density-Based Spatial Clustering of Applications with Noise) and potentially other algorithms such as hierarchical clustering. The lab demonstrates how to apply these unsupervised learning methods to real-world datasets using Python and Jupyter Notebook, analyze the results, and visualize the clusters.

## Contents

- Jupyter Notebook(s) implementing clustering algorithms (DBSCAN and potentially others)
- Data loading, preprocessing, clustering, and visualization steps

## How to Run

1. Clone this repository.
2. Open the Jupyter Notebook in your preferred environment.
3. Follow the cells to run the code and view results.

## Key Insights from Clustering Results and Visualizations

- DBSCAN effectively identifies clusters of varying shapes and sizes, and can detect outliers as noise points.
- Visualization of the clusters revealed that DBSCAN can find meaningful groupings in data, unlike k-means which may force points into clusters even when they do not fit well.
- The choice of parameters (epsilon and min_samples for DBSCAN) significantly affects cluster formation and the interpretation of noise.
- Comparing DBSCAN with hierarchical clustering (if included) shows DBSCAN's strength in handling clusters of arbitrary shape and its robustness to outliers.

## Challenges Faced and Decisions Made

- Selecting optimal parameter values for DBSCAN (especially "eps" and "min_samples") required experimentation and domain knowledge. Too small an "eps" resulted in many noise points, while too large merged distinct clusters.
- Visualizing high-dimensional data for clustering required dimensionality reduction (e.g., PCA or t-SNE), which can sometimes obscure the true cluster structure.
- Deciding when to use DBSCAN versus other clustering algorithms (like k-means or hierarchical clustering) was based on data distribution and presence of noise.
- Handling and interpreting outliers as "noise" was a key decision in evaluating the clustering results.

## Requirements

- Python 3.x
- Jupyter Notebook
- scikit-learn
- pandas
- numpy
- matplotlib
- seaborn (optional, for advanced visualization)

Install the requirements with:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn


# Lab 5: DBSCAN Clustering  
**MSCS 634**

## Purpose of the Lab

The purpose of this lab is to explore and compare clustering techniques, focusing on DBSCAN (Density-Based Spatial Clustering of Applications with Noise) and potentially other algorithms such as hierarchical clustering. The lab demonstrates how to apply these unsupervised learning methods to real-world datasets using Python and Jupyter Notebook, analyze the results, and visualize the clusters.

## Contents

- Jupyter Notebook(s) implementing clustering algorithms (DBSCAN and potentially others)
- Data loading, preprocessing, clustering, and visualization steps

## How to Run

1. Clone this repository.
2. Open the Jupyter Notebook in your preferred environment.
3. Follow the cells to run the code and view results.

## Key Insights from Clustering Results and Visualizations

- DBSCAN effectively identifies clusters of varying shapes and sizes, and can detect outliers as noise points.
- Visualization of the clusters revealed that DBSCAN can find meaningful groupings in data, unlike k-means which may force points into clusters even when they do not fit well.
- The choice of parameters (epsilon and min_samples for DBSCAN) significantly affects cluster formation and the interpretation of noise.
- Comparing DBSCAN with hierarchical clustering (if included) shows DBSCAN's strength in handling clusters of arbitrary shape and its robustness to outliers.

## Challenges Faced and Decisions Made

- Selecting optimal parameter values for DBSCAN (especially "eps" and "min_samples") required experimentation and domain knowledge. Too small an "eps" resulted in many noise points, while too large merged distinct clusters.
- Visualizing high-dimensional data for clustering required dimensionality reduction (e.g., PCA or t-SNE), which can sometimes obscure the true cluster structure.
- Deciding when to use DBSCAN versus other clustering algorithms (like k-means or hierarchical clustering) was based on data distribution and presence of noise.
- Handling and interpreting outliers as "noise" was a key decision in evaluating the clustering results.


Install the requirements with:

```bash
pip install scikit-learn pandas numpy matplotlib seaborn
```

## References

- [DBSCAN Algorithm - scikit-learn documentation](https://scikit-learn.org/stable/modules/clustering.html#dbscan)
- [Clustering in Machine Learning](https://scikit-learn.org/stable/modules/clustering.html)
- Course notes and assignment instructions

---

For more, see the source code and notebooks in the repository:  
[GitHub Repository](https://github.com/sindhuja90comp/Lab_5_MSCS_634_DBSCANClustering)


## Data

- **Dataset:** Wine dataset from scikit-learn  
  - 178 samples, 13 numerical features  
  - No missing values

## Workflow

### 1. Data Preparation and Exploration

- Loads the Wine dataset
- Explores structure using `.head()`, `.info()`, and `.describe()`
- Standardizes features for optimal clustering performance

### 2. Hierarchical Clustering

- Applies **Agglomerative Clustering** (with 3 clusters)
- Visualizes clusters using scatter plots
- Generates a **dendrogram** to show cluster linkages
- Experiments with different numbers of clusters

### 3. DBSCAN Clustering

- Applies **DBSCAN** with specified `eps` and `min_samples`
- Visualizes clusters (and noise points)
- Evaluates clustering quality using:
  - **Silhouette Score**
  - **Homogeneity Score**
  - **Completeness Score**
- Shows how parameter tuning affects clustering results

### 4. Analysis and Insights

- **Comparison** of hierarchical and density-based approaches
- **Key observations**:
  - Hierarchical clustering gives interpretable dendrograms, best for smaller datasets
  - DBSCAN is robust to noise but sensitive to parameters (`eps`, `min_samples`)

## Example Usage

Open and run the notebook in Jupyter Lab/Notebook:

```sh
jupyter notebook wine_data.ipynb
```

Follow the cells sequentially for:
- Data loading and exploration
- Running both clustering algorithms
- Viewing visualizations and evaluation metrics
  

Install dependencies (if needed):

```sh
pip install scikit-learn pandas matplotlib scipy numpy
```

## Results Summary

- **Hierarchical Clustering**: Clear groupings, dendrogram reveals cluster structure.
- **DBSCAN**: Finds clusters of various shapes, identifies noise/outliers, but requires careful parameter tuning.

## References

- [Scikit-learn documentation: Clustering](https://scikit-learn.org/stable/modules/clustering.html)
- [Wine Dataset Information](https://scikit-learn.org/stable/datasets/toy_dataset.html#wine-dataset)
