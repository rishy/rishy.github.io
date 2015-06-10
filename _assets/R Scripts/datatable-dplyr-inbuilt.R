# Import the required libraries
# If not installed then use install.packages('...')
# Plots Order:
# 1. Row Subsetting
# 2. Column Subsetting
# 3. Group_by Operation
# 4.
# 5. 

library(data.table)
library(dplyr)
library(microbenchmark)
library(ggplot2)
library(reshape2)

# Set a random seed, so we can reproduce the results
set.seed(1234)

# Breakpoints for data frame sizes
breaks = c(100, 1000, 10000, 100000, 1000000)

# Create a list of 5 data frames of varying lengths
df_list <- lapply(breaks, function(x){
  data.frame(a = runif(x, 0, 1000000), b = runif(x, 0, 1000000),
             c = runif(x, 0, 1000000), d = runif(x, 0, 1000000),
             e = runif(x, 0, 1000000))
})

# Create a list of 5 corresponding data tables 
dt_list <- lapply(df_list, as.data.table)

# Variables to store running time info 
# of in-built R functions, dplyr and data.table
r_time = c()
dplyr_time = c()
dtable_time = c()

plot_variations <- function(R_list, dplyr_list, datatable_list){
  
  # Data frame to hold the timings of microbenchmark 
  # mean values from in-built R, dplyr and data.table
  wide_df = data.frame(size = c(1, 2, 3, 4, 5), R = R_list,
                       dplyr = dplyr_list, datatable = datatable_list)
  
  plot_df = melt(wide_df, id = 'size')
  
  # Plot sizes with time benchmarks
  ggplot(data=plot_df, aes(x=size, y=value, colour=variable,
                           title = "Row Subsetting Comparison")) +
    geom_line() + geom_point() + ylab("time(in ms)") +
    scale_x_continuous(labels = breaks)
}

# Row Subsetting comparisons

# 1. microbenchmark for in-built R row subsetting
r_time <- sapply(df_list, function(x){
  mean(microbenchmark(x[x$a > 100000, ], unit = "ms")$time)/(1000000)
})

# 2. microbenchmark for dplyr row subsetting with filter()
dplyr_time <- sapply(df_list, function(x){
  mean(microbenchmark(filter(x, a > 100000), unit = "ms")$time)/(1000000)
})

# 3. microbenchmark for data.table row subsetting
dtable_time <- sapply(dt_list, function(x){
  mean(microbenchmark(x[a > 100000], unit = "ms")$time)/(1000000)
})

# Row Subsetting Comparison plot with ggplot
plot_variations(r_time, dplyr_time, dtable_time)


# Column Subsetting comparisons

# 1. microbenchmark for in-built R Column subsetting
r_time <- sapply(df_list, function(x){
  mean(microbenchmark(x[c("a", "b")], unit = "ms")$time)/(1000000)
})

# 2. microbenchmark for dplyr Column subsetting with select()
dplyr_time <- sapply(df_list, function(x){
  mean(microbenchmark(select(x, a, b), unit = "ms")$time)/(1000000)
})

# 3. microbenchmark for data.table Column subsetting
dtable_time <- sapply(dt_list, function(x){
  mean(microbenchmark(x[, .(a, b)], unit = "ms")$time)/(1000000)
})

# Row subsetting Comparison plot with ggplot
plot_variations(r_time, dplyr_time, dtable_time)


# Group_By Operation comparisons

# Let's add a year column in our data frames
years <- seq(1950, 2015)

df_list_year <- lapply(df_list, function(x){
  # Get the number of rows in this data frame
  len <- dim(x)[1]
  
  # Get a sample of years 
  year_sample <- sample(years, size = len, replace = T)
  
  # Add the new column
  x$year <- year_sample  
  x
})

# Create a list of 5 corresponding data tables with years column added
dt_list_year <- lapply(df_list_year, as.data.table)

# 1. microbenchmark for in-built R group(aggregate) operation
r_time <- sapply(df_list_year, function(x){
  mean(microbenchmark(aggregate(year ~ ., data = x, mean),
                      unit = "ms", times = 5L)$time)/(1000000)
})

# 2. microbenchmark for dplyr group(aggregate) subsetting with group_by()
dplyr_time <- sapply(df_list_year, function(x){
  mean(microbenchmark(group_by(x, year),
                      unit = "ms", times = 5L)$time)/(1000000)
})

# 3. microbenchmark for data.table group(aggregate) operations
dtable_time <- sapply(dt_list_year, function(x){
  mean(microbenchmark(x[, mean(a), by = year], unit = "ms",
                      times = 5L)$time)/(1000000)
})

# Group(aggregate) operations Comparison plot with ggplot
plot_variations(r_time, dplyr_time, dtable_time)
