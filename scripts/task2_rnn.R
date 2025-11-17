############################################################
## 1. LOAD CLEANED DATA  -----------------------------------
############################################################
library(tidyverse)
library(tidymodels)
library(keras)
library(tensorflow)

source("/Users/jiangyibing/Desktop/PSTAT197A/module2/scripts/preprocessing.R")

load("/Users/jiangyibing/Desktop/PSTAT197A/module2/data/claims-clean-example.RData")
load("/Users/jiangyibing/Desktop/PSTAT197A/module2/data/claims-test.RData")
clean_test = parse_data(claims_test)

set.seed(110122)
split <- initial_split(claims_clean, prop = 0.8)
train_df <- training(split)
test_df  <- testing(split)

train_text   <- train_df$text_clean
train_labels <- as.numeric(train_df$bclass) - 1  # 0 / 1
test_text    <- test_df$text_clean
test_labels  <- as.numeric(test_df$bclass) - 1

# integer-encode the multiclass labels 0..K-1
mc_levels  <- sort(unique(claims_clean$mclass))
num_mc     <- length(mc_levels)

train_labels_int <- as.integer(factor(train_df$mclass,
                                      levels = mc_levels)) - 1
test_labels_int  <- as.integer(factor(test_df$mclass,
                                      levels = mc_levels)) - 1

# one-hot matrices for keras
train_mc_labels <- to_categorical(train_labels_int, num_classes = num_mc)
test_mc_labels  <- to_categorical(test_labels_int,  num_classes = num_mc)

############################################################
## 2. TEXT VECTORIZATION LAYER (integer sequences) ---------
############################################################
vocab_max   <- 20000                    # keep 20k most-common tokens
sequence_len <- 750                     # truncate / pad to 750 words
tv_layer <- layer_text_vectorization(
  standardize = NULL,
  split       = "whitespace",
  max_tokens  = vocab_max,
  output_mode = "int",
  output_sequence_length = sequence_len
)
tv_layer %>% adapt(train_text)

############################################################
## 3. RNN ARCHITECTURE -------------------------------------
############################################################
embedding_dim <- 128

model_rnn <- keras_model_sequential() %>%
  tv_layer %>%                                            # (batch, seq_len)
  layer_embedding(input_dim = vocab_max + 1,              # +1 for 0/OOV
                  output_dim = embedding_dim,
                  input_length = sequence_len) %>%
  bidirectional(layer_lstm(units = 64, return_sequences = FALSE)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 1, activation = "sigmoid")

summary(model_rnn)

model_mc <- keras_model_sequential() %>%
  tv_layer %>%
  layer_embedding(input_dim = vocab_max + 1,
                  output_dim = embedding_dim,
                  input_length = sequence_len) %>%
  bidirectional(layer_lstm(units = 64, return_sequences = FALSE)) %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = 64, activation = "relu") %>%
  layer_dropout(rate = 0.3) %>%
  layer_dense(units = num_mc, activation = "softmax")

summary(model_mc)

############################################################
## 4. COMPILE & TRAIN --------------------------------------
############################################################
model_rnn %>% compile(
  loss      = "binary_crossentropy",
  optimizer = optimizer_adam(learning_rate = 1e-3),
  metrics   = c("binary_accuracy", "recall", "precision")
)

early_stop <- callback_early_stopping(
  monitor = "val_loss",
  patience = 3,
  restore_best_weights = TRUE
)

history <- model_rnn %>% fit(
  x               = train_text,
  y               = train_labels,
  validation_split = 0.3,
  epochs           = 5,
  batch_size       = 64,
  callbacks        = list(early_stop),
  verbose          = 2
)

model_mc %>% compile(
  loss      = "categorical_crossentropy",
  optimizer = optimizer_adam(learning_rate = 1e-3),
  metrics   = "accuracy"
)

history_mc <- model_mc %>% fit(
  x               = train_text,
  y               = train_mc_labels,
  validation_split = 0.3,
  epochs           = 10,
  batch_size       = 64,
  verbose          = 2
)

############################################################
## 5. Best Result ------------------------
############################################################
cat("\nBinary model – best accuracy:",
    round(max(history$metrics$binary_accuracy), 4), "\n")

cat("\nMulticlass model – best accuracy:",
    round(max(history_mc$metrics$accuracy), 4), "\n")

############################################################
## 5. Save model ------------------------
############################################################
save_model(model_rnn,
           "/Users/jiangyibing/Desktop/PSTAT197A/module2/results/task2_binary_rnn_model.keras")
save_model(model_mc,
           "/Users/jiangyibing/Desktop/PSTAT197A/module2/results/task2_multiclass_rnn_model.keras")


############################################################
## 6. Prediction ------------------------
############################################################
b_probs <- predict(model_rnn, clean_test$text_clean)
b_preds <- ifelse(b_probs > 0.5, 1, 0)              # or change to original labels

# Multiclass
mc_probs   <- predict(model_mc, clean_test$text_clean)
mc_int     <- apply(mc_probs, 1, which.max) - 1      # 0-based
mclass_pred <- mc_levels[mc_int + 1]

pred_df <- tibble(
  .id         = clean_test$.id,
  bclass.pred = b_preds,
  mclass.pred = mclass_pred
)

save(pred_df, file="/Users/jiangyibing/Desktop/PSTAT197A/module2/results/pred_df.RData")



# Calculate metrics on VALIDATION data (same as used in epochs)
library(caret)

# Binary classification - use VALIDATION data
val_binary_preds <- ifelse(predict(model_rnn, test_text) > 0.5, 1, 0)
binary_cm <- confusionMatrix(factor(val_binary_preds, levels = c(0,1)), 
                             factor(test_labels, levels = c(0,1)))

# Multiclass classification with averages
val_multiclass_probs <- predict(model_mc, test_text)
val_multiclass_preds <- apply(val_multiclass_probs, 1, which.max) - 1

# Create tables
binary_table <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity"),
  Value = round(c(binary_cm$overall["Accuracy"],
                  binary_cm$byClass["Sensitivity"],
                  binary_cm$byClass["Specificity"]), 4)
)

# Create results dataframe
multiclass_results <- data.frame(
  truth = factor(test_labels_int, levels = 0:3, labels = c("A", "B", "C", "D")),
  estimate = factor(val_multiclass_preds, levels = 0:3, labels = c("A", "B", "C", "D"))
)

# Calculate metrics
accuracy <- multiclass_results %>% accuracy(truth, estimate) %>% pull(.estimate)
sensitivity <- multiclass_results %>% sens(truth, estimate) %>% pull(.estimate)
specificity <- multiclass_results %>% spec(truth, estimate) %>% pull(.estimate)

# Create table
multiclass_table <- data.frame(
  Metric = c("Accuracy", "Sensitivity", "Specificity"),
  Value = round(c(accuracy, sensitivity, specificity), 4)
)

# Print results
print("Binary Classification:")
print(binary_table)

print("Multiclass Classification (Averages):")
print(multiclass_table)



