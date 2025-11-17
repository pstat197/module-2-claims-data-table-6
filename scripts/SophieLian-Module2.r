# Logistic PCA Regression with Headers

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
library(tidymodels)

set.seed(11162025)

source('scripts/preprocessing.R')


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

parse_data_headers <- function(.df){
  .df %>%
    filter(str_detect(text_tmp, '<!')) %>%
    rowwise() %>%
    mutate(text_clean = parse_fn_headers(text_tmp)) %>%
    unnest(text_clean)
}


load('data/claims-raw.RData')


claims_clean_paragraph <- claims_raw %>%
  parse_data()


claims_clean_headers <- claims_raw %>%
  parse_data_headers()


tfidf_paragraph <- nlp_fn(claims_clean_paragraph)
tfidf_headers   <- nlp_fn(claims_clean_headers)


lpcr <- function(tfidf_df, n_comp = 50){
  
  partitions <- initial_split(tfidf_df, prop = 0.8, strata = bclass)
  train_df  <- training(partitions)
  test_df   <- testing(partitions)
  
  X_train <- train_df %>% select(-.id, -bclass) %>% as.matrix()
  X_test  <- test_df  %>% select(-.id, -bclass) %>% as.matrix()
  
  y_train <- as.numeric(train_df$bclass) - 1
  y_test  <- as.numeric(test_df$bclass) - 1
  
  # Drop constant columns
  non_constant <- apply(X_train, 2, var) != 0
  X_train <- X_train[, non_constant, drop = FALSE]
  X_test  <- X_test[,  non_constant, drop = FALSE]
  
  # Scale
  X_train_scaled <- scale(X_train)
  X_test_scaled <- scale(
    X_test,
    center = attr(X_train_scaled, "scaled:center"),
    scale  = attr(X_train_scaled, "scaled:scale")
  )
  
  # PCA
  pca_obj <- prcomp(X_train_scaled)
  Z_train <- pca_obj$x[, 1:n_comp, drop = FALSE]
  Z_test  <- X_test_scaled %*% pca_obj$rotation[, 1:n_comp, drop = FALSE]
  

  ridge_mod <- cv.glmnet(
    Z_train,
    y_train,
    family = "binomial",
    alpha = 0          
  )
  
  prob_hat <- predict(ridge_mod, newx = Z_test, s = "lambda.min", type = "response")
  class_hat <- ifelse(prob_hat > 0.5, 1, 0)
  
  mean(class_hat == y_test)
}




acc_paragraph <- lpcr(tfidf_paragraph, n_comp = 50)
acc_headers   <- lpcr(tfidf_headers,   n_comp = 50)

acc_paragraph
# output: accuracy 0.5443038
acc_headers
# ou5tput: accuracy 0.5309524
