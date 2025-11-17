# scripts/task1-compare-lpcr.R

library(tidyverse)
library(rsample)
library(irlba)

source("scripts/preprocessing.R")

# logistic PCR - return accuracy
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
  
  mean(test_pred == y_test) 
}

# 1. Accuracy WITHOUT headers
load("data/claims-clean-example.RData")      # paragraphs only
accuracy_no_headers <- run_lpcr(claims_clean)
print(paste("Accuracy without headers:", accuracy_no_headers))
#output = Accuracy without headers: 0.546835443037975


# 2. Accuracy WITH headers
load("data/claims-clean-headers.RData")      # headers + paragraphs
accuracy_headers <- run_lpcr(claims_clean)
print(paste("Accuracy with headers:", accuracy_headers))
#output = "Accuracy with headers: 0.530952380952381"
