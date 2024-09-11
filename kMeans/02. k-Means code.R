# Packages ----------------------------------------------------------------
library(tidyr)
library(plyr)
library(dplyr)
library(ggplot2)
library(gganimate)

# Generate Data -----------------------------------------------------------

# We generate some data using 2 different normal distributions

# data for cluster 1
c1_data <- data.frame("cluster" = "A",
                      "x" = rnorm(100,0,1),
                      "y" = rnorm(100,0,1))

c2_data <- data.frame("cluster" = "B",
                      "x" = rnorm(100,3,1),
                      "y" = rnorm(100,3,1))

# Join clusters together
data <- rbind(c1_data,
              c2_data)

# Plot
ggplot(data,aes(x = x, y = y, col = cluster)) +
  geom_point() +
  theme_minimal()

# Set Seed
set.seed(0216)

# shuffle data
data <- sample(data)

# remove cluster column in preperation for k-means algorithm
data <- data %>%
  dplyr::select(-cluster)


# K-Means Clustering ------------------------------------------------------

kMeansVersionRo <- function(data, clusters, iterations){
  
  # STEP 1: INITIALIZE LISTS
  
  # initialize means list
  means_list <- vector("list", length = iterations)
  
  # intiialize closest distances list
  closest_distances_list <- vector("list", length = iterations)

  # STEP 2: GET STARTING MEANS
  
  # Get starting means - from random sampled points in the data set
  means <- sample_n(data,
                    clusters,
                    replace = FALSE)

  
  # STEP 3: START K-MEANS PROCESS
  
  # Begin looping through iterations
  for (i in 1:iterations){
  
    # Put current means into means_list
    means_list[[i]] <- means %>%
      mutate(kmeanid = 1:clusters,
             Iteration = i)
    
    # Calculate Euclidean distance from each observation to each mean
    distances <- vector("list", length = clusters)
    
    for (j in 1:clusters){
      distances[[j]] <- ldply(apply(data,
                                    1,
                                    FUN = function(x){
                                      sqrt(rowSums((x - means[j,])^2))})) %>%
        mutate("kmeanid" = j)
    }
    
    # Pull together into one dataframe
    distances <- ldply(distances) %>%
      mutate(index = ((row_number() - 1) %/% clusters) + 1,
             index = rep(seq(nrow(data)), clusters)) %>%
      rename("Distance" = V1)
    
    # add index to data
    data_with_index <- data %>%
      mutate(index = seq(nrow(data)))
    
    # bolt on x and y values using the index
    distances <- merge(distances,
                       data_with_index,
                       by = "index")
    
    # Find min mean
    closest_distances <- distances %>%
      group_by(index) %>%
      filter(Distance == min(Distance)) %>%
      ungroup() %>%
      mutate(Iteration = i)
    
    # Store closest_distances
    closest_distances_list[[i]] <- closest_distances
    
    # Update means
    means <- closest_distances %>%
      dplyr::select(-c(index,Distance, Iteration)) %>%
      group_by(kmeanid) %>%
      summarise_all(.funs = mean) %>%
      ungroup() %>%
      dplyr::select(-kmeanid)
    
    # Print iteration number
    print(paste0("Iteration: ", i))

  }
  
  # STEP 4: RETURN OUTPUT
  
  output <- list("means" = means_list,
                 "data" = closest_distances_list)
  
  
}

# Use function:
cluster_means <- kMeansVersionRo(data, k = 2, iterations = 5)


# Animation Plot ----------------------------------------------------------

# Data
plot_points <- ldply(cluster_means$data)

# Means
plot_means <- ldply(cluster_means$means)

# Plot
p <- ggplot() +
  geom_point(data = plot_points,
             aes(x = x,
                 y = y,
                 col = as.factor(kmeanid)),
             size = 2,
             alpha = 0.8,
             shape = 4) +
  geom_point(data = plot_means,
             aes(x = x,
                 y = y,
                 col = as.factor(kmeanid)),
             size = 5)+
  theme_minimal() +
  guides(col = guide_legend(title = "Cluster")) +
  theme(plot.title = element_text(size = 20))

# Animate plot by iteration
p + transition_manual(Iteration) + labs(title = "k-Means Iteration: {frame}")



