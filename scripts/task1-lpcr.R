# scripts/task1-compare-lpcr.R

library(tidyverse)
library(rsample)
library(irlba)

source("scripts/preprocessing.R")

# logistic PCR - return accuracy, pca object, and fitted model
run_lpcr <- function(clean_data) {
  
  claims_tfidf <- nlp_fn(clean_data)
  
  set.seed(197)
  split <- initial_split(claims_tfidf, prop = 0.8, strata = bclass)
  train_data <- training(split)
  test_data  <- testing(split)
  
  X_train <- train_data %>%
    select(-.id, -bclass) %>%
    as.matrix()
  
  X_test <- test_data %>%
    select(-.id, -bclass) %>%
    as.matrix()

  # remove zero-variance columns
  non_constant <- apply(X_train, 2, var) != 0
  X_train <- X_train[, non_constant]
  X_test  <- X_test[,  non_constant]
  
  y_train <- train_data$bclass
  y_test  <- test_data$bclass
  
  pca <- prcomp_irlba(X_train, n = 50, center = TRUE, scale. = TRUE)
  
  Z_train <- pca$x
  Z_test  <- scale(X_test, center = pca$center, scale = pca$scale) %*%
    pca$rotation
  
  train_df <- data.frame(bclass = y_train, Z_train)
  test_df  <- data.frame(bclass = y_test, Z_test)
  
  logit_fit <- glm(bclass ~ ., data = train_df, family = binomial())
  test_prob <- predict(logit_fit, newdata = test_df, type = "response")
  
  test_pred <- ifelse(test_prob > 0.5,
                      levels(y_train)[2],
                      levels(y_train)[1]) |>
    factor(levels = levels(y_train))
  
  accuracy <- mean(test_pred == y_test) 
  
  return(list(
    accuracy = accuracy,
    pca = pca,
    model = logit_fit
  ))
}

# 1. Accuracy WITHOUT headers
load("data/claims-clean-example.RData")      # paragraphs only
results_no_headers <- run_lpcr(claims_clean)
accuracy_no_headers <- results_no_headers$accuracy
print(paste("Accuracy without headers:", accuracy_no_headers))
#output = Accuracy without headers: 0.546835443037975

# create folder
dir.create("results/Task1Results", showWarnings = FALSE)

# Save PCA + model
saveRDS(results_no_headers$model, 
        file = "results/Task1Results/task1-model_no_headers.rds")
saveRDS(results_no_headers$pca,   
        file = "results/Task1Results/pca_no_headers.rds")



# 2. Accuracy WITH headers
load("data/claims-clean-headers.RData")      # headers + paragraphs
results_headers <- run_lpcr(claims_clean)
accuracy_headers <- results_headers$accuracy
print(paste("Accuracy with headers:", accuracy_headers))
#output = "Accuracy with headers: 0.530952380952381"

# Save PCA + model
saveRDS(results_headers$model, 
        file = "results/Task1Results/task1-model_headers.rds")
saveRDS(results_headers$pca,   
        file = "results/Task1Results/pca_headers.rds")
