# Load the required libraries
library(keras)
library(tensorflow)

# Set the seed for reproducibility
set.seed(1)

# Define the path to the dataset
path <- 'C:/Users/yueru/OneDrive/Desktop/CU/2023 Spring/AI/Research_Paper'
positive_dir <- file.path(path, 'Positive')
negative_dir <- file.path(path, 'Negative')

# Load the image data as an array
positive_files <- list.files(positive_dir, full.names = TRUE)
negative_files <- list.files(negative_dir, full.names = TRUE)
all_files <- c(positive_files, negative_files)
all_labels <- c(rep(1, length(positive_files)), rep(0, length(negative_files)))
image_data <- lapply(all_files, jpeg::readJPEG) # Use jpeg package to read JPEG files
image_data <- array_reshape(image_data,  c(length(all_files), dim(image_data[[1]])))


shuffled_indices <- sample(length(all_labels))
# Use the shuffled indices to reorder both arrays
image_data <- image_data[shuffled_indices, , , ]
all_labels <- all_labels[shuffled_indices]

x_train <- image_data[1:150, , , ]
y_train <- all_labels[1:150]
x_test <- image_data[151:198, , , ]
y_test <- all_labels[151:198]

# Define the CNN model
model <- keras_model_sequential()
model %>%
  layer_conv_2d(filters = 64, kernel_size = c(3, 3), activation = "relu", 
                input_shape = c(227, 227, 3)) %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_conv_2d(filters = 128, kernel_size = c(3, 3), activation = "relu") %>%
  layer_max_pooling_2d(pool_size = c(2, 2)) %>%
  layer_dropout(rate = 0.5) %>%
  layer_flatten() %>%
  layer_dense(units = 128, activation = "relu") %>%
  layer_dropout(rate = 0.5) %>%
  layer_dense(units = 1, activation = "sigmoid")

# Compile the model
model %>% compile(
  loss = "binary_crossentropy",
  optimizer = optimizer_rmsprop(learning_rate = 0.001),
  metrics = list("accuracy")
)

# Train the model
history <- model %>% fit(
  x_train, y_train,
  epochs = 100,
  validation_split = 0.2
)

# Evaluate the model
metrics <- model %>% evaluate(x_test,y_test)
print(metrics)
