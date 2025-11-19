library(tidyverse)
library(keras)
library(tensorflow)
source('scripts/preprocessing.R')  

# Load test set
load('data/claims-test.RData')
load('data/claims-clean-example.RData')

# Clean test data
clean_test <- parse_data(claims_test)


# Binary and multiclass RNN models (from primary task)
model_rnn <- load_model_tf("results/primary_task_binary_rnn_model.keras")
model_mc  <- load_model_tf("results/primary_task_multiclass_rnn_model.keras")

# Multiclass levels 
mc_levels <- sort(unique(claims_clean$mclass))


# Binary classification
b_probs <- predict(model_rnn, clean_test$text_clean)
b_preds <- ifelse(b_probs > 0.5, 1, 0)  

# Multiclass classification
mc_probs   <- predict(model_mc, clean_test$text_clean)
mc_int     <- apply(mc_probs, 1, which.max) - 1  # 0-based indices
mclass_pred <- mc_levels[mc_int + 1]


pred_df <- tibble(
  .id         = clean_test$.id,
  bclass.pred = b_preds,
  mclass.pred = mclass_pred
)

save(pred_df, file = "results/preds-group6.RData")
