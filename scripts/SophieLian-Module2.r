# ========================================
# Logistic PCA Regression with Headers
# ========================================

# Required packages
library(tidyverse)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tokenizers)
library(tm)
library(glmnet)
library(caret)

set.seed(1234)

# Source original preprocessing functions
source('scripts/preprocessing.R')

# ----------------------------------------
# 1. Define new parse function including headers
# ----------------------------------------
parse_fn_headers <- function(.html){
  read_html(.html) %>%
    html_elements('p, h1, h2, h3, h4, h5, h6') %>%
    html_text2() %>%
    str_c(collapse = ' ') %>%
    rm_url() %>%
    rm_email() %>%
    str_remove_all('\'') %>%
    str_replace_all(paste(c('\n', 
                            '[[:punct:]]', 
                            'nbsp', 
                            '[[:digit:]]', 
                            '[[:symbol:]]'),
                          collapse = '|'), ' ') %>%
    str_replace_all("([a-z])([A-Z])", "\\1 \\2") %>%
    tolower() %>%
    str_replace_all("\\s+", " ")
}

# ----------------------------------------
# 2. Load raw data
# ----------------------------------------
load('data/claims-raw.RData')

# ----------------------------------------
# 3. Train/test split
# ----------------------------------------
set.seed(1234)
partitions <- initial_split(claims_raw, prop = 0.8)
train_raw <- training(partitions)
test_raw  <- testing(partitions)

# ----------------------------------------
# 4. Preprocess datasets
# ----------------------------------------
# Paragraph-only
claims_train_paragraph <- train_raw %>%
  parse_data()  # uses original parse_fn
claims_test_paragraph <- test_raw %>%
  parse_data()

# Paragraph + headers
claims_train_headers <- train_raw %>%
  parse_data() %>%
  rowwise() %>%
  mutate(text_clean = parse_fn_headers(text_tmp)) %>%
  unnest(text_clean)

claims_test_headers <- test_raw %>%
  parse_data() %>%
  rowwise() %>%
  mutate(text_clean = parse_fn_headers(text_tmp)) %>%
  unnest(text_clean)

# ----------------------------------------
# 5. Define LPCR function using glmnet
# ----------------------------------------
lpcr_glmnet <- function(train_df, test_df, n_pcs = 10){
  
  # ---- Train TF-IDF ----
  tfidf_train <- train_df %>%
    unnest_tokens(word, text_clean) %>%
    anti_join(stop_words) %>%
    count(.id, bclass, word) %>%
    bind_tf_idf(word, .id, n)
  
  X_train <- tfidf_train %>%
    select(.id, word, tf_idf) %>%
    pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%
    left_join(select(train_df, .id, bclass), by = ".id")
  
  y_train <- as.numeric(X_train$bclass) - 1
  X_train_mat <- as.matrix(X_train %>% select(-.id, -bclass))
  
  # ---- PCA ----
  pca <- prcomp(X_train_mat, center = TRUE, scale. = TRUE)
  pcs_train <- pca$x[, 1:n_pcs]
  
  # ---- Fit logistic regression with glmnet ----
  fit <- cv.glmnet(pcs_train, y_train, family = "binomial", alpha = 0)
  
  # ---- Prepare test set ----
  tfidf_test <- test_df %>%
    unnest_tokens(word, text_clean) %>%
    anti_join(stop_words) %>%
    count(.id, bclass, word) %>%
    bind_tf_idf(word, .id, n)
  
  X_test <- tfidf_test %>%
    select(.id, word, tf_idf) %>%
    pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0) %>%
    left_join(select(test_df, .id, bclass), by = ".id")
  
  # Align columns to train
  X_test_mat <- as.matrix(X_test %>% select(-.id, -bclass))
  # Fill missing columns with 0
  missing_cols <- setdiff(colnames(X_train_mat), colnames(X_test_mat))
  if(length(missing_cols) > 0){
    X_test_mat <- cbind(X_test_mat, matrix(0, nrow = nrow(X_test_mat), ncol = length(missing_cols)))
    colnames(X_test_mat)[(ncol(X_test_mat)-length(missing_cols)+1):ncol(X_test_mat)] <- missing_cols
  }
  # Ensure same column order
  X_test_mat <- X_test_mat[, colnames(X_train_mat)]
  
  # ---- PCA transform test set ----
  pcs_test <- scale(X_test_mat, center = pca$center, scale = pca$scale) %*% pca$rotation[, 1:n_pcs]
  
  # ---- Predict ----
  preds_prob <- predict(fit, newx = pcs_test, s = "lambda.min", type = "response")
  preds_class <- ifelse(preds_prob > 0.5, 1, 0)
  
  # ---- Accuracy ----
  y_test <- as.numeric(X_test$bclass) - 1
  accuracy <- mean(preds_class == y_test)
  
  return(accuracy)
}

# ----------------------------------------
# 6. Run models and compute test accuracy
# ----------------------------------------
accuracy_paragraph <- lpcr_glmnet(claims_train_paragraph, claims_test_paragraph, n_pcs = 10)
accuracy_headers   <- lpcr_glmnet(claims_train_headers,   claims_test_headers, n_pcs = 10)

cat("LPCR test accuracy (paragraph-only):", round(accuracy_paragraph, 3), "\n")
cat("LPCR test accuracy (paragraph + headers):", round(accuracy_headers, 3), "\n")

