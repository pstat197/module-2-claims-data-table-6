# Required packages

library(tidyverse)
library(tidytext)
library(textstem)
library(rvest)
library(qdapRegex)
library(stopwords)
library(tokenizers)
library(tm)

set.seed(1234)


# Source original preprocessing functions

source('scripts/preprocessing.R')


# 1. Define new parse function including headers

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


# 2. Load raw data

load('data/claims-raw.RData')


# 3. Preprocess datasets

# Paragraph-only (original)
claims_clean_paragraph <- claims_raw %>%
  parse_data()  # uses original parse_fn from preprocessing.R

# Paragraph + headers
claims_clean_headers <- claims_raw %>%
  parse_data() %>%
  rowwise() %>%
  mutate(text_clean = parse_fn_headers(text_tmp)) %>%
  unnest(text_clean)


# 4. Define LPCR function

lpcr_model <- function(cleaned_df, n_pcs = 10){
  
  # Tokenize, remove stopwords, compute TF-IDF
  tfidf_df <- cleaned_df %>%
    unnest_tokens(word, text_clean) %>%
    anti_join(stop_words) %>%
    count(.id, bclass, word) %>%
    bind_tf_idf(word, .id, n)
  
  # Convert to wide matrix
  X_df <- tfidf_df %>%
    select(.id, word, tf_idf) %>%
    pivot_wider(names_from = word, values_from = tf_idf, values_fill = 0)
  
  # Add labels
  lpcr_df <- X_df %>%
    left_join(select(cleaned_df, .id, bclass), by = ".id")
  
  X <- lpcr_df %>% select(-.id, -bclass)
  y <- lpcr_df$bclass
  
  # PCA
  pca <- prcomp(X, center = TRUE, scale. = TRUE)
  
  # Select first n PCs
  pcs <- pca$x[, 1:n_pcs]
  
  # Fit logistic regression
  logit_mod <- glm(y ~ ., data = data.frame(y, pcs), family = binomial)
  
  # Predictions
  probs <- predict(logit_mod, type = "response")
  pred <- ifelse(probs > 0.5, 1, 0)
  
  # Accuracy
  accuracy <- mean(pred == (as.numeric(y) - 1))
  
  return(accuracy)
}


# 5. Compute accuracy for both datasets

accuracy_paragraph <- lpcr_model(claims_clean_paragraph, n_pcs = 10)
accuracy_headers   <- lpcr_model(claims_clean_headers, n_pcs = 10)

cat("LPCR accuracy (paragraph-only):", round(accuracy_paragraph, 3), "\n")
cat("LPCR accuracy (paragraph + headers):", round(accuracy_headers, 3), "\n")




