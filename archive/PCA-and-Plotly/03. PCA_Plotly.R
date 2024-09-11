# Principal Component Analysis
library(plyr)
library(tidyr)
library(dplyr)
library(plotly)
# We'll run PCA on the popular IRIS dataset.
# The aim of PCA is reduce the number of features in a dataset while not losing information
# gained through the features.
# PCA creates linear combinations of features that aim to explain variability in the data.
# The Eigenvectors 

glimpse(iris)

# Produce long format of data for plotly plots
iris_long <- iris %>%
  gather(key = "Feature",
         value = "Value",
         -Species)

# Create 3 plots for each unqiue response value
p1 <- plot_ly(iris_long %>% filter(Species == "setosa"),
              x = ~Value,
              color = ~Feature,type = "histogram",
              text  = ~paste0(Species,"\n",
                              Feature,"\n"))

p2 <- plot_ly(iris_long %>% filter(Species == "versicolor"),
              x = ~Value,
              color = ~Feature,type = "histogram",
              text  = ~paste0(Species,"\n",
                              Feature,"\n"))

p3 <- plot_ly(iris_long %>% filter(Species == "virginica"),
              x = ~Value,
              color = ~Feature,type = "histogram",
              text  = ~paste0(Species,"\n",
                              Feature,"\n"))

# Use subplots to plot together
subplot(p1,
        style(p2, showlegend = FALSE),
        style(p3, showlegend = FALSE),
        nrows = 3)
  

# Get feature data:
x_df <- iris %>% 
  dplyr::select(-Species)

# Get response data:
y_df <- iris %>% 
  dplyr::select(Species)


# Standardize data:
x_standard_df <- data.frame(apply(x_df,
                                  2,
                                  FUN = function(x){(x - min(x))/(max(x) - min(x))}))





# Center data:
x_centered_df <- data.frame(apply(x_standard_df,
                                2,
                                FUN = function(x){ x - mean(x)}))

# Convert to matrix for matrix multiplication:
x_centered_matrix <- as.matrix(x_centered_df)

# Calculate covariance matrix (alternatively we could use Rs built in cov() function):
cov_matrix <- (t(x_centered_matrix) %*% x_centered_matrix) / (nrow(x_centered_matrix) - 1)

# Get Eigenvalues:
eigenvalues_df <- data.frame("Eigenvalue" = eigen(cov_matrix)$values)
num_eigenvalues <- nrow(eigenvalues_df)

# Add a variation explained column:
eigenvalues_df <- eigenvalues_df %>%
  mutate("Variation" = Eigenvalue / sum(Eigenvalue),
         "CumulativeVariation" = cumsum(Variation))


# Get Eigenvectors:
eigenvectors_df <- as.data.frame(eigen(cov_matrix)$vectors)


# Plotly Plot:
screeplot <- plot_ly() %>%
  # Add Variation Bars
  add_trace(x = 1:num_eigenvalues,
            y = eigenvalues_df$Variation,
            name = "Variation",
            type = "bar",
            text = ~paste0("PC", 1:num_eigenvalues, "\n",
                           "Variation Explained: ", round(eigenvalues_df$Variation * 100,2), "%", "\n",
                           "Eigenvalue: ", round(eigenvalues_df$Eigenvalue,2)),
  hoverinfo = 'text') %>%
  # Add Cumulative Line
  add_trace(x = 1:num_eigenvalues,
            y = eigenvalues_df$CumulativeVariation,
            name = "Cumulative Variation",
            type = "scatter",
            mode = "lines+markers",
            text = ~paste0("PC", 1:num_eigenvalues, "\n",
                           "Cumulative Variation Explained: ", round(eigenvalues_df$CumulativeVariation * 100,2), "%", "\n",
                           "Eigenvalue: ", round(eigenvalues_df$Eigenvalue,2)),
            hoverinfo = 'text') %>%
  layout(title = "Scree Plot for PCA on Iris Data",
         xaxis = list(title = "Principal Components"),
         yaxis = list(title = "Variation",
                      tickformat = ',.0%'))

screeplot

# We'll pick PC1 and PC2

# Build our transformation matrix
transformation_matrix <- as.matrix(eigenvectors_df[,1:2])

# Transform the data: the dot product of the centered data and our transformation matrix:
transformed_df <- as.data.frame(x_centered_matrix %*% transformation_matrix)

# Rename our componenets
names(transformed_df) <- c("PC1", "PC2")

# Add on response variable (species)
transformed_df <- cbind(transformed_df,
                          y_df)

# Plot
scatterplot <- plot_ly(data = transformed_df) %>%
  add_trace(x = ~PC1,
            y = ~PC2,
            color = ~Species,
            type = "scatter",
            mode = "markers") %>%
  layout(title = "Iris Species after PCA")

scatterplot

# Bi Plot

# Produce a bi plot by adding the eigenvectors * eigenvalues to our plot
# eigenvalues
# eigenvectors
# 
# biplot_matrix <- eigen(cov_matrix)$vectors * eigen(cov_matrix)$values
# 
# biplot_df <- as.data.frame(biplot_matrix[,1:2])
# 
# names(biplot_df) <- c("PC1", "PC2")
# 
# biplot_df <- biplot_df %>%
#   mutate(Feature = names(x_df))
# 
# plot_ly() %>%
#   add_trace(data = biplot_df,
#             x = ~PC1,
#             y = ~PC2,
#             type = 'scatter',
#             mode = 'markers+text',
#             text = ~Feature)



# Let's fit a tree based method on both the transformed data and the original data and 
# see how much performance we've lost in the process:

# Train %
kTrainObs <- round(nrow(transformed_df) * 0.7, digits = 0)


# Split original data
original_shuffled <- iris[sample.int(nrow(iris),
                                 replace = FALSE),]

original_train <-  original_shuffled[1:kTrainObs,]
original_test <- original_shuffled[(kTrainObs + 1):nrow(original_shuffled),]

# Split transformed data
transformed_shuffled <- transformed_df[sample.int(nrow(transformed_df),
                                                        replace = FALSE),]

transformed_train <-  transformed_shuffled[1:kTrainObs,]
transformed_test <- transformed_shuffled[(kTrainObs + 1):nrow(transformed_shuffled),]



# Fit Trees
tree_original <- rpart(Species ~., 
                       data = original_train,
                       method = "class")

tree_pca <- rpart(Species ~., 
                  data = transformed_train,
                  method = "class")


# Results
original_results <- table(predict(tree_original, original_test, type = "class"),
                              original_test$Species)

transformed_results <- table(predict(tree_pca, transformed_test, type = "class"),
                                 transformed_test$Species)




cov(transformed_df)
