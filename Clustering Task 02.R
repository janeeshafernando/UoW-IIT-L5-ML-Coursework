# Loading necessary packages
library(ggplot2) 
library(ggcorrplot)
library(dplyr) 
library(NbClust) 
library(factoextra) 
library(cluster) 
library(gridExtra)
library(fpc)

# Define constants 
QUALITY_COLUMN_INDEX <- 12

# Loading the dataset
wine_data <- read.csv("wine_dataset.csv", header = TRUE)
head(wine_data)

# Remove the 'quality' column from the wine dataset
wine_features <- wine_data[, -QUALITY_COLUMN_INDEX]
head(wine_features)

# Store the quality column separately 
wine_quality <- wine_data$quality

# Get the column names from the wine_features data frame
column_names <- names(wine_features)

# Summary of wine features dataframe 
summary(wine_features)

# Pre-processing Task 01 - Outlier Detection and Removal

# Function to create box plots for each column
create_boxplots <- function(data_frame) {
  for (col in names(data_frame)) {
    print(ggplot(data_frame, aes(x = "", y = .data[[col]])) +
            geom_boxplot(fill = "#0D98BA") +
            ggtitle(paste("Box Plot for", col)) +
            theme_minimal())
  }
}

create_boxplots(wine_features)

# Define outlier threshold constant
OUTLIER_THRESHOLD <- 1.5

# Function to remove outliers from the dataframe
remove_outliers <- function(data_frame) {
  numeric_columns <- sapply(data_frame, is.numeric)  # Identify numeric columns
  outlier_indices <- integer(0)  # Initialize empty vector to collect outlier indices
  
  for (col in names(data_frame)[numeric_columns]) {
    Q1 <- quantile(data_frame[[col]], 0.25, na.rm = TRUE)
    Q3 <- quantile(data_frame[[col]], 0.75, na.rm = TRUE)
    IQR <- Q3 - Q1
    lower_bound <- Q1 - OUTLIER_THRESHOLD * IQR
    upper_bound <- Q3 + OUTLIER_THRESHOLD * IQR
    
    # Collect indices of outliers in this column
    col_outliers <- which(data_frame[[col]] < lower_bound | data_frame[[col]] > upper_bound)
    outlier_indices <- unique(c(outlier_indices, col_outliers))
  }
  
  # Remove all rows that have outliers in any of the numeric columns
  cleaned_data <- data_frame[-outlier_indices, ]
  
  return(cleaned_data)
}

# Removing outliers

# Remove outliers from the initial dataset
wine_clean_1 <- remove_outliers(wine_features)

# Create box plots using the cleaned data
create_boxplots(wine_clean_1)

# Apply a second round of outlier removal for any remaining outliers
wine_clean_2 <- remove_outliers(wine_clean_1)

# Create box plots using the cleaned data
create_boxplots(wine_clean_2)

# Perform a third pass of outlier removal to ensure the data is thoroughly cleaned
final_wine_clean <- remove_outliers(wine_clean_2)

# Create box plots using the cleaned data
create_boxplots(final_wine_clean)

# Pre-processing Task 02 - Scaling 

# Scale the features using Z-Score Standardization
wine_scaled <- as.data.frame(scale(final_wine_clean))

# Print the first few rows of the scaled data frame
head(wine_scaled)

# Summary of scale wine features dataframe 
summary(wine_scaled)

# Check if normalization is needed by computing the variance of each variable
apply(wine_scaled, 2, var)

# Compute the Covariance Matrix
wine_cov_matrix <- cov(wine_scaled)
wine_cov_matrix

# Plot the Heat Map of the covariance matrix
ggcorrplot(as.matrix((wine_cov_matrix)))

# Calculate the eigenvalues & eigenvectors
wine_eigen <- eigen(wine_cov_matrix)
wine_eigen
str(wine_eigen)

# Access eigenvalues and eigenvectors separately
wine_eigenvalues <- wine_eigen$values
wine_eigenvectors <- wine_eigen$vectors

# Change the direction of the eigenvectors to positive side for logical interpretation
wine_eigenvectors <- -wine_eigenvectors  
wine_eigenvectors

# Assign row and column names to the dataframe
row.names(wine_eigenvectors) <- column_names
colnames(wine_eigenvectors) <- c("PC1", "PC2", "PC3", "PC4", "PC5", "PC6", "PC7", "PC8", "PC9", "PC10", "PC11")
wine_eigenvectors

# Calculate the Proportion of Variance Explained (PVE)
wine_eigenvalues
PVE <- wine_eigenvalues / sum(wine_eigenvalues)
round(PVE, 2)

# Cumulative PVE plot
cumPVE <- qplot(c(1:11), cumsum(PVE)) + geom_line() +   xlab("Principal Component") + 
  ylab(NULL) +  ggtitle("Cumulative Scree Plot") +  ylim(0, 1)

grid.arrange(cumPVE)

# Based on the PVE, select only the first seven PCs
principal_components <- wine_eigenvectors[, 1:7]
principal_components

# Transfer data points to the new dimensional space

# Calculate all principal components for the first 7 PCs
PCA_scores <- as.matrix(wine_scaled) %*% principal_components

# Create a dataframe with Principal Components scores
transformed_data <- data.frame(PCA_scores)
head(transformed_data)

# Setting the seed function for reproducible results
set.seed(123)

# Determine optimal number of clusters
cluster_analysis_result = NbClust(transformed_data, distance = "euclidean", min.nc = 2, 
                                  max.nc = 10, method = "kmeans", index = "all")

# Determining the relevant number of clusters using Elbow Method
fviz_nbclust(transformed_data, kmeans, method = 'wss')

# Using gap statistic to determine the number of clusters
fviz_nbclust(transformed_data, kmeans, method = 'gap_stat')

# Determining the relevant number of clusters using average silhouette method
fviz_nbclust(transformed_data, kmeans, method = 'silhouette')

# Performing k-means Clustering

# Perform k-means clustering with k=2 on the scaled data
kmeans_clusters = kmeans(transformed_data, 2) 
# Print the results of k-means clustering with k=2
print(kmeans_clusters) 

# Plot the clusters
fviz_cluster(kmeans_clusters, data = transformed_data)

# Extract the total within-cluster sum of squares for k=2
within_cluster_sum_of_squares = kmeans_clusters$tot.withinss 
cat("Total within-cluster sum of squares: ", within_cluster_sum_of_squares)  

# Extract the between-cluster sum of squares for k=2
between_cluster_sum_of_squares = kmeans_clusters$betweenss
cat("Between Sum of Squares:", between_cluster_sum_of_squares)  

# Calculate the silhouette plot for k=2
silhouette_values <- silhouette(kmeans_clusters$cluster, dist(transformed_data)) 
# Visualize the silhouette plot for k=2
fviz_silhouette(silhouette_values)   

# Illustrate the Calinski-Harabasz index
calinski_harabasz_index = calinhara(transformed_data, kmeans_clusters$cluster, 2)
calinski_harabasz_index

