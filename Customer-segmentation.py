import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

# Load the data using an absolute path
df = pd.read_csv('C:/Users/Dell/Downloads/Mall_Customers.csv')
print(df.head())
print(df.info())
print(df.describe())

# The rest of your code remains the same
sns.set(style="whitegrid")

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.histplot(df['Age'], kde=True, bins=20, color='blue')
plt.title('Age Distribution')

plt.subplot(1, 3, 2)
sns.histplot(df['Annual Income (k$)'], kde=True, bins=20, color='green')
plt.title('Annual Income Distribution')

plt.subplot(1, 3, 3)
sns.histplot(df['Spending Score (1-100)'], kde=True, bins=20, color='red')
plt.title('Spending Score Distribution')

plt.tight_layout()
plt.show()

df = df.drop('CustomerID', axis=1)
df['Gender_Encoded'] = df['Gender'].map({'Male': 0, 'Female': 1})
df = df.drop('Gender', axis=1)

correlation_matrix = df.corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=0.5)
plt.title('Correlation Matrix')
plt.show()

plt.figure(figsize=(15, 5))

plt.subplot(1, 3, 1)
sns.boxplot(df['Age'])
plt.title('Age Box Plot')

plt.subplot(1, 3, 2)
sns.boxplot(df['Annual Income (k$)'])
plt.title('Annual Income Box Plot')

plt.subplot(1, 3, 3)
sns.boxplot(df['Spending Score (1-100)'])
plt.title('Spending Score Box Plot')

plt.tight_layout()
plt.show()

scaler = StandardScaler()
df_scaled = scaler.fit_transform(df)
data_scaled_df = pd.DataFrame(df_scaled, columns=df.columns)

wcss = []
for i in range(1, 11):
    kmeans = KMeans(n_clusters=i, random_state=42)
    kmeans.fit(data_scaled_df)
    wcss.append(kmeans.inertia_)

plt.plot(range(1, 11), wcss, marker='o')
plt.title('Elbow Method')
plt.xlabel('Number of clusters')
plt.ylabel('WCSS')
plt.show()

optimal_clusters = 5  # Set this based on the elbow method result
kmeans = KMeans(n_clusters=optimal_clusters, random_state=42)
df['Cluster'] = kmeans.fit_predict(data_scaled_df)

silhouette_avg = silhouette_score(data_scaled_df, df['Cluster'])
print(f'Silhouette Score: {silhouette_avg}')

cluster_centers = pd.DataFrame(scaler.inverse_transform(kmeans.cluster_centers_), columns=df.columns[:-1])
print(cluster_centers)

plt.figure(figsize=(10, 6))
sns.scatterplot(x=df['Annual Income (k$)'], y=df['Spending Score (1-100)'], hue=df['Cluster'], palette='viridis')
plt.title('Customer Segmentation Based on Annual Income and Spending Score')
plt.show()
