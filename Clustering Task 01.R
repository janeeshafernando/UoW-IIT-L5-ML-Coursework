# Loading necessary packages 
library(ggplot2)
library(dplyr)
library(NbClust)
library(factoextra)
library(cluster)

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

# Setting the seed for reproducibility
set.seed(200)

# Determine optimal number of clusters
cluster_results <- NbClust(wine_scaled, distance="euclidean", min.nc=2, max.nc=10, method="kmeans", index="all")

# Determining the relevant number of clusters using Elbow Method
fviz_nbclust(wine_scaled, kmeans, method="wss")

# Using gap statistic to determine the number of clusters
fviz_nbclust(wine_scaled, kmeans, method='gap_stat')

# Using silhouette method to determine the number of clusters
fviz_nbclust(wine_scaled, kmeans, method='silhouette')

# Performing k-means Clustering

# Perform k-means clustering with k=2
kmeans_result <- kmeans(wine_scaled, 2)
# Print the results of k-means clustering
print(kmeans_result)

# Plot the clusters
fviz_cluster(kmeans_result, data = wine_scaled)

# Extract the total within-cluster sum of squares for k=2
wss <- kmeans_result$withinss
cat("Total within-cluster sum of square: ", wss, "\n")

# Extract the between-cluster sum of squares for k=2
bss <- kmeans_result$betweenss
cat("Between-cluster sum of squares: ", bss, "\n")

# Calculate and visualize the silhouette plot for k=2
silhouette_plot <- silhouette(kmeans_result$cluster, dist(wine_scaled))
fviz_silhouette(silhouette_plot)
