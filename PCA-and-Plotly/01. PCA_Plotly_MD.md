Principal Component Analysis and Plotly
================

NOTE: PLOTLY PLOTS ARE CURRENTLY NOT WORKING
--------------------------------------------

Outline
-------

We are going to outline the concept of *Principal Component Analysis* (PCA) and use it to reduce the size of the feature space of the Iris data set. We'll then fit a tree based model to both the original Iris data set and the new PCA-transformed data and compare model performance. We'll be using plotly to produce various plots including a Scree plot for our PCA process.

Motivation for PCA
------------------

Why might someone want to use PCA? Consider a dataset where we have a very large number of features, in the thousands or millions perhaps. Fitting a predictive model to such a dataset might be computationally impossible, we can only throw so much computing power at the problem. If we could reduce the number of features while still maintaining **most** of the information from the original feature space then we might take this deal due to the benefits of working with a much smaller number of features. Reducing the feature space of your data can result in a less complicated model and a more reasonable training time.

Outline of PCA
--------------

PCA rebases your data and creates new features called *principal components* that are linear combinations of the current features. These principal components are made by finding lines, or directions in your data cloud where the most variability is present. If we were to fit multiples lines through our data cloud (through the mean of our data) and project our data points onto the lines, then the first principal component would be the line that has the largest sum of square distances from the mean. The other principal componenets would be lines that are orthogonal to the first.

Coding:
-------

Now that we've got a bit of the explanation out of the way, we can start working in R.

### Packages

We'll start by loading all of our packages:

``` r
library(plyr)
library(tidyr)
library(dplyr)
library(plotly)
```

### Iris Data

Now we'll get our data. We're going to use the popular Iris data set to explore PCA. Let's explore it briefly:

``` r
glimpse(iris)
```

    ## Observations: 150
    ## Variables: 5
    ## $ Sepal.Length <dbl> 5.1, 4.9, 4.7, 4.6, 5.0, 5.4, 4.6, 5.0, 4.4, 4.9,...
    ## $ Sepal.Width  <dbl> 3.5, 3.0, 3.2, 3.1, 3.6, 3.9, 3.4, 3.4, 2.9, 3.1,...
    ## $ Petal.Length <dbl> 1.4, 1.4, 1.3, 1.5, 1.4, 1.7, 1.4, 1.5, 1.4, 1.5,...
    ## $ Petal.Width  <dbl> 0.2, 0.2, 0.2, 0.2, 0.2, 0.4, 0.3, 0.2, 0.2, 0.1,...
    ## $ Species      <fct> setosa, setosa, setosa, setosa, setosa, setosa, s...

We have 4 features:

-   Sepal Length
-   Sepal Width
-   Petal Length
-   Petal Width

and a target variable Species:

``` r
unique(iris$Species)
```

    ## [1] setosa     versicolor virginica 
    ## Levels: setosa versicolor virginica

We can produce some plots:

``` r
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
# subplot(p1,
#         style(p2, showlegend = FALSE),
#         style(p3, showlegend = FALSE),
#         nrows = 3)
```

### Split Features and Response

We will split our data set into two separate data sets, one containing the 4 features and the other containing the response variable Species:

``` r
# Features
x_df <- iris %>% 
  dplyr::select(-Species)

# Response
y_df <- iris %>% 
  dplyr::select(Species)
```

### Standardize the features

The first step is to standardise our data and then to center the data by subtracting the mean from each data point. Centering the data is an important step in the PCA process.

``` r
# Standardize data
x_standard_df <- data.frame(apply(x_df,
                                  2,
                                  FUN = function(x){(x - min(x))/(max(x) - min(x))}))

# Center data
x_centered_df <- data.frame(apply(x_standard_df,
                                2,
                                FUN = function(x){ x - mean(x)}))
```

### Convert from DataFrame to Matrix

We'll convert our data to a matrix to allow for matrix multiplication:

``` r
x_centered_matrix <- as.matrix(x_centered_df)
```

### Covariance Matrix

We now calculate the covariance matrix of our data (We have calculated this the long way but we could alternatively use the cov() function in R):

``` r
cov_matrix <- (t(x_centered_matrix) %*% x_centered_matrix) / (nrow(x_centered_matrix) - 1)
```

### Eigenvectors

As mentioned, matrices can be thought of as linear tranformations of vectors. The covariance matrix happens to apply an important transformaton with respect to the data it is generated from. Given a vector *v*, applying the covariance matrix to *v* it will rotate *v* towards the direction of the most variation in the sample data. As this process is repeated it, the direction of *v* will slowly converge to the direction of the most variation. As we can recall from earlier, that is exactly what we're looking for when trying to calculate the prinipal components for PCA. Now, we could iteratively apply the covariance matrix transformation to some arbitrary vector and we will arrive at a point of convergence, or we could simply find a vector *v* such that applying the covariance matrix to *v* does not change its direction, namely the eigenvectors of the covariance matrix!

``` r
eigenvectors_df <- as.data.frame(eigen(cov_matrix)$vectors)

# take a look:
eigenvectors_df
```

    ##           V1           V2         V3         V4
    ## 1  0.4249421 -0.423202708 -0.7135724  0.3621300
    ## 2 -0.1507482 -0.903967112  0.3363160 -0.2168178
    ## 3  0.6162670  0.060383083 -0.0659003 -0.7824487
    ## 4  0.6456889  0.009839255  0.6110345  0.4578492

### Eigenvalues

Each eigenvector has a corresponding eigenvalue. This value corresponds to the magnitude of the transformation. That is, if we consider our eigenvector *e*, and apply our covariance matrix transformation to it, the eigenvalue tells us how much the eigenvector *e* will grow or shrink (remembering that the direction of the eigenvector does not change).

``` r
eigenvalues_df <- data.frame("Eigenvalue" = eigen(cov_matrix)$values)

# take a look:
eigenvalues_df
```

    ##    Eigenvalue
    ## 1 0.232453251
    ## 2 0.032468204
    ## 3 0.009596846
    ## 4 0.001764319

As for PCA, the eigenvalues tell us how much of the variability in the data is explained by each prinicpal component, the higher the eigenvalue, the more variability it explains. We can make this a little easier to intepret by creating a specific column:

``` r
# add variation and cumulative variation columns:
eigenvalues_df <- eigenvalues_df %>%
  mutate("Variation" = Eigenvalue / sum(Eigenvalue),
         "CumulativeVariation" = cumsum(Variation))

# take a look:
eigenvalues_df
```

    ##    Eigenvalue   Variation CumulativeVariation
    ## 1 0.232453251 0.841360382           0.8413604
    ## 2 0.032468204 0.117518082           0.9588785
    ## 3 0.009596846 0.034735614           0.9936141
    ## 4 0.001764319 0.006385922           1.0000000

We'll also store the number of eigenvalues as a variable for use with plotting, this is equal to the number of features:

``` r
num_eigenvalues <- nrow(eigenvalues_df)
```

### Scree Plot

A Scree Plot is a plot used to visualise how much of the variability in the original data set is being explained by each principal component. Despite being all a little subjective, it is also a useful plot when determining how many principal componenets to keep.

``` r
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

# screeplot
```

### Create Principal Components

Now we can transform our data onto our new linear combinations of our old features.

### Transformation Matrix

First we build our transformation matrix:

``` r
transformation_matrix <- as.matrix(eigenvectors_df[,1:2])
```

### Transform the Data

Take the dot product of our centered data and our transformation matrix:

``` r
transformed_df <- as.data.frame(x_centered_matrix %*% transformation_matrix)

# take a look:
head(transformed_df, n = 5)
```

    ##           V1          V2
    ## 1 -0.6307029 -0.10757791
    ## 2 -0.6229049  0.10425983
    ## 3 -0.6695204  0.05141706
    ## 4 -0.6541528  0.10288487
    ## 5 -0.6487881 -0.13348758

Rename the columns:

``` r
names(transformed_df) <- c("PC1", "PC2")
```

Bolt on our response variable:

``` r
transformed_df <- cbind(transformed_df,
                          y_df)
```

### Scatterplot

Now that we only have two features, we can plot all the information in our data set in a single 2D scatter plot:

``` r
# Plot
scatterplot <- plot_ly(data = transformed_df) %>%
  add_trace(x = ~PC1,
            y = ~PC2,
            color = ~Species,
            type = "scatter",
            mode = "markers") %>%
  layout(title = "Iris Species after PCA")

# scatterplot
```
