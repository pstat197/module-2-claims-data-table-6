
library(tidyverse)
library(tidymodels)
library(keras3)
library(tensorflow)

# load raw or pre-cleaned data
load('data/claims-clean-example.RData')  # or claims-clean-headers.RData

# apply preprocessing functions from task 1 (ty janice)
claims_tfidf <- nlp_fn(claims_clean)

# partition into train/test
set.seed(197) #same seed
split <- initial_split(claims_tfidf, prop = 0.8, strata = bclass)
train_data <- training(split)
test_data  <- testing(split)

# prepare matrices for NN
X_train <- train_data %>% select(-.id, -bclass) %>% as.matrix()
X_test  <- test_data  %>% select(-.id, -bclass) %>% as.matrix()

y_train <- train_data$bclass %>% as.numeric() - 1
y_test  <- test_data$bclass  %>% as.numeric() - 1

# remove constant columns (variance = 0)
non_constant <- apply(X_train, 2, var) != 0
X_train <- X_train[, non_constant]
X_test  <- X_test[,  non_constant]

# Define neural network

model <- keras_model_sequential() %>%
  layer_dense(units = 64, activation = "relu", input_shape = ncol(X_train) #transforms into 64 features
              ) %>%  # L2 regularization
  layer_dropout(0.15) %>%                                    # slightly higher dropout (initially tried 0.2)
  layer_dense(units = 32, activation = "relu"#round 2; takes whatever's left and further transforms into 32 features
             ) %>%
  layer_dropout(0.15) %>%#randomly turns off 20% of the neurons
  layer_dense(units = 16, activation = "relu") %>%          # #round 3; takes whatever's left and further transforms into 16 features
  layer_dense(units = 1, activation = "sigmoid")            # output layer for binary prediction

model %>% compile(
  loss = "binary_crossentropy", #inaccuracy
  optimizer = "adam", #algo that adjusts weights
  metrics = "binary_accuracy" #actual accuracy value
)

# train the model
history <- model %>% fit(
  X_train, y_train,
  validation_split = 0.2,
  epochs = 25,
  batch_size = 32
)

# Evaluate accuracy on test set
metrics <- model %>% evaluate(X_test, y_test)
cat("Test Accuracy:", metrics$binary_accuracy, "\n") #accuracy =  0.7833334, ~25% better than logistic PCR 

# Save the model 
save_model(model, "results/task3-model.keras")

#tldr; NN learns complex patterns in the TF-IDF features, generalizes well thanks to lower `dropout` + more rounds of denser layers, and achieves ~25% higher accuracy on the test set than logistic principal component regression





