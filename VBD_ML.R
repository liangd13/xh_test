Lasso

# 1. Load necessary packages
library(readxl)
library(dplyr)
library(ggplot2)
library(patchwork) # Combine plots
library(glmnet)    # Lasso
library(randomForest)
library(caret)
library(fastshap)
library(tidyr)     # Data reshaping (required for Lasso path plot)
library(ggrepel)

common_theme <- theme_bw() +
  theme(
    # Overall plot background: solid white
    plot.background = element_rect(fill = "white", color = NA),
    # Panel background: solid white
    panel.background = element_rect(fill = "white", color = NA),
    # Legend background: solid white
    legend.background = element_rect(fill = "white", color = NA),
    # Bold plot title
    plot.title = element_text(face = "bold", size = 12),
    # Axis title size
    axis.title = element_text(size = 10)
  )

# 1. Data Loading and Preprocessing
# Please modify your file path
file_path <- ""
target_var <- "Y" # Name of your dependent variable

# Load data
data_all <- read_xlsx(file_path)
data_all <- as.data.frame(data_all)

# Basic cleaning: Remove missing values
data_all <- na.omit(data_all)

# Automatically convert character variables to factors
for (col in names(data_all)) {
  if (is.character(data_all[[col]])) {
    data_all[[col]] <- as.factor(data_all[[col]])
  }
}

# Ensure Y is a factor (Binary classification)
data_all[[target_var]] <- as.factor(data_all[[target_var]])

# Check and remove zero-variance (single-value) predictors
single_val_cols <- sapply(data_all, function(x) length(unique(x)) <= 1)
if (any(single_val_cols)) {
  bad_cols <- names(data_all)[single_val_cols]
  message("Removed zero-variance variables: ", paste(bad_cols, collapse = ", "))
  data_all <- data_all[, !single_val_cols]
}

message("Data preparation completed. Sample size: ", nrow(data_all))

# 2. Lasso Regression Modeling
set.seed(123)

# Prepare matrix data
y_lasso <- data_all[[target_var]]
x_matrix <- model.matrix(as.formula(paste(target_var, "~ .")), data = data_all)[, -1]

# 2.1 Run Model
# Full model
fit_lasso <- glmnet(x_matrix, y_lasso, family = "binomial", alpha = 1)

# Cross-validation model (for selecting lambda)
cv_lasso <- cv.glmnet(x_matrix, y_lasso, family = "binomial", alpha = 1, type.measure = "auc")

se1_lambda <- cv_lasso$lambda.1se
min_lambda <- cv_lasso$lambda.min
log(se1_lambda)

message("Optimal Lasso Lambda: ", round(se1_lambda, 4))

# Extract variables and coefficients selected by Lasso
# 1. Extract sparse coefficient matrix at 1se.Lambda
coef_obj <- coef(cv_lasso, s = "lambda.1se")

# 2. Convert to standard dataframe
# as.matrix converts sparse matrix to regular matrix, then to data.frame
lasso_result_df <- data.frame(
  Feature = rownames(coef_obj),
  Coefficient = as.matrix(coef_obj)[, 1]
)

# 3. Clean data
lasso_result_df <- lasso_result_df %>%
  # Remove intercept
  filter(Feature != "(Intercept)") %>%
  # Keep only non-zero coefficients
  filter(Coefficient != 0) %>%
  # Sort by absolute value of coefficient in descending order
  arrange(desc(abs(Coefficient)))

# 4. Calculate Odds Ratio (OR)
lasso_result_df$OR_Value <- exp(lasso_result_df$Coefficient)

# 5. Output display
message(">> Lasso Selection Results (Sorted by importance):")
print(lasso_result_df)

# 6. If saving to Excel or CSV is needed
# write.csv(lasso_result_df, "Lasso_Selected_Features.csv", row.names = FALSE)

# 3. Plot Lasso Charts (ggplot2 version)
# --- Figure 3.1: Cross-Validation Error Plot (CV Plot) ---
# Extract CV data
cv_data <- data.frame(
  lambda = cv_lasso$lambda,
  cvm = cv_lasso$cvm,   # Mean
  cvup = cv_lasso$cvup, # Upper bound
  cvlo = cv_lasso$cvlo  # Lower bound
)

p_lasso_cv <- ggplot(cv_data, aes(x = log(lambda), y = cvm)) +
  geom_errorbar(aes(ymin = cvlo, ymax = cvup), color = "grey", width = 0.05) +
  geom_point(color = "#D53E4F", size = 2) +
  geom_vline(xintercept = log(se1_lambda), linetype = "dashed", color = "blue") +
  geom_vline(xintercept = log(min_lambda), linetype = "dashed", color = "black") +
  theme_bw() +
  labs(title = "Lasso Cross-Validation (AUC)", x = "Log(Lambda)", y = "Model Error (Deviance/AUC)")+
  common_theme

print(p_lasso_cv)

# --- Figure 3.2: Coefficient Path Plot ---
# Extract coefficient matrix and convert to long format
beta_matrix <- as.matrix(fit_lasso$beta)
path_data <- as.data.frame(t(beta_matrix))
path_data$lambda <- fit_lasso$lambda
path_data$log_lambda <- log(fit_lasso$lambda)

# Convert to long format required by ggplot
path_data_long <- pivot_longer(path_data, cols = -c(lambda, log_lambda),
                               names_to = "Variable", values_to = "Coefficient")

# Select the leftmost point (minimum Log Lambda) as label position
label_data <- path_data_long %>%
  group_by(Variable) %>%
  filter(log_lambda == min(log_lambda)) %>%
  ungroup() %>%
  # Optional: Only label variables with absolute coefficients > 0.001 at the endpoint to prevent text overlapping on the 0 baseline
  filter(abs(Coefficient) > 0.001)

p_lasso_path <- ggplot(path_data_long, aes(x = log_lambda, y = Coefficient, group = Variable, color = Variable)) +
  geom_line(alpha = 0.8) +
  theme_bw() +
  geom_text_repel(
    data = label_data,
    aes(label = Variable),
    size = 3,           # Font size
    direction = "y",    # Adjust mainly in vertical direction to prevent overlap
    hjust = 1,          # Right align
    xlim = c(-Inf, min(path_data_long$log_lambda)), # Restrict labels to the left area
    max.overlaps = 20   # Maximum allowed overlaps
  ) +
  # Expand left space of X-axis to leave room for text
  scale_x_continuous(expand = expansion(mult = c(0.2, 0.05))) +
  theme(legend.position = "none") + # Hide legend since direct labels are used
  labs(title = "Lasso Coefficient Path", x = "Log(Lambda)", y = "Coefficients") +
  common_theme

print(p_lasso_path)

# Combine and display Lasso results
p_lasso_combined <- p_lasso_path + p_lasso_cv
print(p_lasso_combined)
ggsave("1_Lasso_Diagnostics.png", p_lasso_combined, width = 12, height = 5)

# 4. Extract Variables Selected by Lasso
coef_obj <- coef(cv_lasso, s = "lambda.1se")
selected_vars_idx <- which(coef_obj[, 1] != 0)
selected_vars_names <- rownames(coef_obj)[selected_vars_idx]

# Remove intercept
selected_vars_names <- selected_vars_names[selected_vars_names != "(Intercept)"]

# Note: Lasso selects dummy variable names (e.g., SexM), need to revert to original column names (Sex)
# Simple matching restoration here
all_orig_cols <- names(data_all)
final_features <- c()

for (orig_col in all_orig_cols) {
  # If original column name appears in Lasso selected names (partial match)
  # e.g., Lasso selected "AgeGroup60+", original column is "AgeGroup"
  if (any(grepl(orig_col, selected_vars_names))) {
    final_features <- c(final_features, orig_col)
  }
}
final_features <- unique(final_features)

# Exclude Y
final_features <- setdiff(final_features, target_var)

message("Original variables selected by Lasso: ", paste(final_features, collapse = ", "))

if (length(final_features) == 0) stop("Lasso removed all variables! Please check the data or increase lambda.")












ML

# --------------------------------------------------------------
# Multiple Machine Learning Model Construction
# --------------------------------------------------------------
library(tidymodels)
library(tidyverse)
library(doParallel)
library(ranger)
library(xgboost)
library(C50)
library(kernlab)
library(kknn)
library(rpart)
library(nnet)
library(glmnet)
library(pROC)
library(ggplot2)
library(themis)

# Data Import --------------------------------------------------------------------
df_raw <- read.csv("")

# Data Preprocessing -------------------------------------------------------------
df_final <- df_raw

# 1. Split data first
set.seed(123)
data_split <- initial_split(df_final, prop = 0.7, strata = LPA_Class)
train_raw  <- training(data_split)
test_raw   <- testing(data_split)

# Modified Recipe
rec <- recipe(LPA_Class ~ ., data = train_raw) %>%
  # Handle missing values first
  step_impute_median(all_numeric_predictors()) %>%
  step_impute_mode(all_nominal_predictors()) %>%
  # Convert nominal variables to dummy variables
  step_dummy(all_nominal_predictors())

# Reprocess data
rec_prepped <- prep(rec)
train_data  <- bake(rec_prepped, new_data = NULL)      # Training set is balanced here 
test_data   <- bake(rec_prepped, new_data = test_raw)  # Testing set remains original, no balancing applied

# Parallel acceleration
cl <- makePSOCKcluster(detectCores() - 1)
registerDoParallel(cl)

# Machine Learning Modeling ------------------------------------------------------
# Parameter settings
specs <- list(
  rf      = rand_forest(mtry = floor(sqrt(ncol(train_data))), trees = 600, min_n = 70) %>%
    set_engine("ranger", importance = "impurity") %>% set_mode("classification"),
  
  xgboost = boost_tree(trees = 300, tree_depth = 4, learn_rate = 0.05, min_n = 25) %>%
    set_engine("xgboost") %>% set_mode("classification"),
  
  knn     = nearest_neighbor(neighbors = 40, weight_func = "rectangular") %>%
    set_engine("kknn") %>% set_mode("classification"),
  
  c50     = boost_tree(trees = 50, min_n = 10) %>% set_engine("C5.0") %>% set_mode("classification"),
  
  svm     = svm_rbf(cost = 1, rbf_sigma = 0.05) %>% set_engine("kernlab") %>% set_mode("classification"),
  
  rpart   = decision_tree(cost_complexity = 0.005, tree_depth = 10) %>% set_engine("rpart") %>% set_mode("classification"),
  
  nnet    = mlp(hidden_units = 5, penalty = 0.1, epochs = 500) %>% set_engine("nnet", trace = FALSE) %>% set_mode("classification"),
  
  logreg  = logistic_reg(penalty = 0.015, mixture = 1) %>% set_engine("glmnet") %>% set_mode("classification")
)

# Train models
cat("=== Training 8 models... ===\n")
fitted_models <- list()

for(name in names(specs)) {
  wflow <- workflow() %>% add_model(specs[[name]]) %>% add_formula(LPA_Class ~ .)
  fitted_models[[name]] <- fit(wflow, data = train_data)
}
# stopCluster(cl)

evaluate_dataset <- function(models_list, dataset, dataset_name) {
  cat(sprintf(">>> Generating evaluation table for: %s <<<\n", dataset_name))
  
  res_list <- list()
  roc_objs <- list()
  
  # Specify positive class as "1"
  positive_class <- "1"
  prob_col <- paste0(".pred_", positive_class)
  
  for(name in names(models_list)) {
    model <- models_list[[name]]
    
    # Predict classes and probabilities
    pred_class <- predict(model, dataset, type = "class")
    pred_prob  <- predict(model, dataset, type = "prob")
    
    res <- bind_cols(dataset %>% dplyr::select(LPA_Class), pred_class, pred_prob)
    
    # Calculate metrics (using yardstick)
    acc <- accuracy(res, truth = LPA_Class, estimate = .pred_class)$.estimate
    mcc_val <- mcc(res, truth = LPA_Class, estimate = .pred_class)$.estimate
    
    # Confusion matrix metrics (Note: LPA_Class is a factor, with "1" coming first)
    # caret::confusionMatrix defaults to treating the first level as positive
    cm <- caret::confusionMatrix(res$.pred_class, res$LPA_Class, positive = positive_class)
    
    # ROC calculation (pROC)
    # Note: levels order is c(Negative class, Positive class)
    roc_obj <- roc(response = res$LPA_Class, predictor = res[[prob_col]], 
                   levels = c("0", "1"), direction = "<", quiet = TRUE)
    
    auc_val <- as.numeric(auc(roc_obj))
    ci_val  <- ci.auc(roc_obj)
    roc_objs[[name]] <- roc_obj
    
    res_list[[name]] <- tibble(
      Model = name,
      AUC = auc_val,
      AUC_Low = ci_val[1],
      AUC_High = ci_val[3],
      Accuracy = acc,
      MCC = mcc_val,
      Sens = cm$byClass["Sensitivity"],
      Spec = cm$byClass["Specificity"],
      PPV  = cm$byClass["Pos Pred Value"],
      NPV  = cm$byClass["Neg Pred Value"]
    )
  }
  
  # Sort
  final_df <- bind_rows(res_list) %>% arrange(desc(AUC))
  
  # DeLong's test
  if("logreg" %in% names(roc_objs)) {
    ref_roc <- roc_objs[["logreg"]]
    p_values <- sapply(final_df$Model, function(m) {
      if(m == "logreg") return(NA)
      tryCatch({
        test <- roc.test(ref_roc, roc_objs[[m]], method = "delong")
        test$p.value
      }, error = function(e) return(NA))
    })
    
    final_df <- final_df %>%
      mutate(P_Value = p_values) %>%
      mutate(P_Value = case_when(
        is.na(P_Value) ~ "Ref",
        P_Value < 0.001 ~ "<0.001",
        TRUE ~ as.character(round(P_Value, 3))
      ))
  }
  
  # Format output table
  final_df <- final_df %>%
    mutate(across(where(is.numeric), ~ round(., 3))) %>%
    mutate(`AUC (95% CI)` = paste0(AUC, " (", AUC_Low, "-", AUC_High, ")")) %>%
    dplyr::select(Model, `AUC (95% CI)`, P_Value, Accuracy, MCC, Sens, Spec, PPV, NPV)
  
  return(final_df)
}

# Call function to generate results
train_results <- evaluate_dataset(fitted_models, train_data, "Training Set")
test_results  <- evaluate_dataset(fitted_models, test_data, "Testing Set")

# Output tables
table_1 <- evaluate_dataset(fitted_models, train_data, "Training Set")
table_2 <- evaluate_dataset(fitted_models, test_data, "Testing Set")

print(as.data.frame(table_1))
print(as.data.frame(table_2))

# base_path <- "C:/Users/无敌懒大王/Desktop/Project/R/Charls_001/Final_result/table/"
# write.csv(x = table_1, 
#           "C:/Users/无敌懒大王/Desktop/Project/R/Charls_001/Final_result/table/Training_Set_table.csv", 
#           row.names = FALSE)
# write.csv(x = table_2, 
#           "C:/Users/无敌懒大王/Desktop/Project/R/Charls_001/Final_result/table/Testing_Set_table.csv", 
#           row.names = FALSE)

# Plot Color Configuration -------------------------------------------------------
library(ggplot2)
library(pROC)
library(dplyr)
library(tidyr)
library(dcurves)

# --- 1. Define global color palette ---
model_colors <- c(
  "rf"      = "#E64B35",  # Red
  "xgboost" = "#4DBBD5",  # Blue
  "c50"     = "#00A087",  # Green
  "rpart"   = "#3C5488",  # Dark Blue
  "logreg"  = "#F39B7F",  # Orange
  "knn"     = "#8491B4",  # Grey
  "svm"     = "#91D1C2",  # Light Green
  "nnet"    = "#DC0000",  # Bright Red
  "All"     = "#000000",  # For DCA: Intervene all
  "None"    = "#808080"   # For DCA: Intervene none
)

# Define clean publication theme function
theme_sci_pub <- function() {
  theme_bw() +
    theme(
      panel.grid.major = element_blank(),
      panel.grid.minor = element_blank(),
      axis.text = element_text(color = "black", size = 10),
      axis.title = element_text(face = "bold"),
      plot.title = element_text(hjust = 0.5, face = "bold", size = 14),
      legend.background = element_blank(),
      legend.key = element_blank()
    )
}

# ROC Plotting -------------------------------------------------------------------
plot_roc_sci <- function(models_list, dataset, title_name) {
  roc_data <- data.frame()
  legend_labels <- c()
  
  # Only plot the specified models or all
  target_names <- intersect(names(models_list), names(model_colors))
  
  for (name in target_names) {
    probs <- predict(models_list[[name]], dataset, type = "prob")
    roc_obj <- roc(dataset$LPA_Class, probs$.pred_1, levels = c("0", "1"), direction = "<", quiet = TRUE)
    ci_val <- ci.auc(roc_obj)
    
    # Construct legend display text with CI
    full_label <- paste0(name, ": AUC = ", 
                         sprintf("%.3f", as.numeric(auc(roc_obj))),
                         " (", sprintf("%.3f", ci_val[1]), "-", sprintf("%.3f", ci_val[3]), ")")
    legend_labels[name] <- full_label
    
    df_temp <- data.frame(
      FPR = 1 - roc_obj$specificities,
      TPR = roc_obj$sensitivities,
      Model = name
    )
    roc_data <- rbind(roc_data, df_temp)
  }
  
  ggplot(roc_data, aes(x = FPR, y = TPR, color = Model)) +
    geom_path(linewidth = 1) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "grey50") +
    scale_color_manual(values = model_colors, labels = legend_labels) +
    theme_sci_pub() +
    coord_fixed() +
    labs(title = paste("ROC Comparison:", title_name),
         x = "1 - Specificity", y = "Sensitivity",
         color = "Model Performance") +
    theme(legend.position = c(0.80, 0.15), legend.text = element_text(size = 12))
}

# Call plotting function
p_roc_train <- plot_roc_sci(fitted_models, train_data, "Training Set")
p_roc_test  <- plot_roc_sci(fitted_models, test_data, "Testing Set")

print(p_roc_train)
print(p_roc_test)

# DCA Curves ---------------------------------------------------------------------
plot_dca_sci <- function(models_list, dataset, title) {
  dca_data <- dataset %>% mutate(y = as.numeric(as.character(LPA_Class))) %>% select(y)
  target_names <- intersect(names(models_list), names(model_colors))
  
  for (name in target_names) {
    dca_data[[name]] <- predict(models_list[[name]], dataset, type = "prob")$.pred_1
  }
  
  dca_res <- dca(y ~ ., data = dca_data)
  
  as_tibble(dca_res) %>%
    mutate(label = case_when(variable == "all" ~ "All", variable == "none" ~ "None", TRUE ~ label)) %>%
    ggplot(aes(x = threshold, y = net_benefit, color = label)) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = model_colors) +
    coord_cartesian(ylim = c(-0.01, NA)) + theme_sci_pub() +
    labs(title = title, x = "Threshold Probability", y = "Net Benefit", color = "Model")+
    theme(
      legend.position = c(0.1, 0.4),      # Top right coordinates
      legend.justification = c("right", "top"),
      legend.text = element_text(size = 12),
      legend.title = element_blank()
    )
}

p_dca_train <- plot_dca_sci(fitted_models, train_data, "Training Set")
p_dca_test  <- plot_dca_sci(fitted_models, test_data, "Testing Set")

print(p_dca_train)
print(p_dca_test)

# Calibration Curves -------------------------------------------------------------
plot_cal_sci <- function(models_list, dataset, title) {
  cal_df <- data.frame()
  target_names <- intersect(names(models_list), names(model_colors))
  
  for (name in target_names) {
    prob <- predict(models_list[[name]], dataset, type = "prob")$.pred_1
    obs <- as.numeric(as.character(dataset$LPA_Class))
    
    dat <- data.frame(obs, prob) %>%
      mutate(bin = cut(prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)) %>%
      group_by(bin) %>%
      summarise(y = mean(obs), x = mean(prob), .groups = "drop") %>%
      mutate(Model = name)
    cal_df <- rbind(cal_df, dat)
  }
  
  ggplot(cal_df, aes(x = x, y = y, color = Model)) +
    geom_line(linewidth = 0.8) + geom_point(size = 1.5, shape = 21, fill = "white") +
    geom_abline(lty = 2, color = "grey50") +
    scale_color_manual(values = model_colors) +
    xlim(0, 1) + ylim(0, 1) + theme_sci_pub() +
    labs(title = title, x = "Predicted Probability", y = "Observed Proportion")+
    theme(
      legend.position = c(0.97, 0.03),
      legend.justification = c("right", "bottom"),
      legend.background = element_blank(),
      legend.text = element_text(size = 12),
      legend.title = element_blank()
    )
}

p_cal_train <- plot_cal_sci(fitted_models, train_data, "Training Set")
p_cal_test  <- plot_cal_sci(fitted_models, test_data, "Testing Set")

print(p_cal_train)
print(p_cal_test)

# Combine ROC/DCA/Calibration Curves ---------------------------------------------
library(patchwork)

generate_aligned_panel <- function(models_list, dataset, dataset_name_label) {
  
  # --- 1. Define shared theme logic for subplots ---
  # Set legend at the bottom and force it into 4 rows
  individual_theme <- theme_sci_pub() +
    theme(
      legend.position = "bottom",
      legend.direction = "horizontal",
      legend.box = "vertical",
      legend.text = element_text(size = 7),
      legend.title = element_text(size = 8, face = "bold"),
      legend.spacing.x = unit(0.2, "cm"),
      legend.key.width = unit(0.6, "cm")
    )
  
  # --- A. ROC Curve (with 4-row legend) ---
  roc_df <- data.frame(); labels_map <- c()
  target_names <- intersect(names(models_list), names(model_colors))
  
  for (name in target_names) {
    prob_vals <- predict(models_list[[name]], dataset, type = "prob")$.pred_1
    roc_obj <- roc(dataset$LPA_Class, prob_vals, levels = c("0", "1"), direction = "<", quiet = TRUE)
    ci <- ci.auc(roc_obj)
    lbl <- paste0(name, ": AUC = ", sprintf("%.3f", as.numeric(auc(roc_obj))),
                  " (", sprintf("%.3f", ci[1]), "-", sprintf("%.3f", ci[3]), ")")
    labels_map[name] <- lbl
    roc_df <- rbind(roc_df, data.frame(FPR = 1 - roc_obj$specificities, TPR = roc_obj$sensitivities, Model = name))
  }
  
  p_roc <- ggplot(roc_df, aes(x = FPR, y = TPR, color = Model)) +
    geom_path(linewidth = 0.8) + geom_abline(lty = 2, color = "grey60") +
    scale_color_manual(values = model_colors, labels = labels_map) +
    coord_fixed() +
    labs(title = paste("A. ROC -", dataset_name_label), x = "1-Specificity", y = "Sensitivity", color = "Model AUC (95% CI)") +
    individual_theme +
    guides(color = guide_legend(nrow = 4)) # Force 4 rows
  
  # --- B. Calibration Curve (with 4-row legend) ---
  cal_df <- data.frame()
  for (name in target_names) {
    prob_vals <- predict(models_list[[name]], dataset, type = "prob")$.pred_1
    obs_vals <- as.numeric(as.character(dataset$LPA_Class))
    dat <- data.frame(obs = obs_vals, prob = prob_vals) %>%
      mutate(bin = cut(prob, breaks = seq(0, 1, by = 0.1), include.lowest = TRUE)) %>%
      group_by(bin) %>%
      summarise(y = mean(obs), x = mean(prob), .groups = "drop") %>%
      mutate(Model = name)
    cal_df <- rbind(cal_df, dat)
  }
  
  p_cal <- ggplot(cal_df, aes(x = x, y = y, color = Model)) +
    geom_line(linewidth = 0.8) + geom_point(size = 1.2, shape = 21, fill = "white") +
    geom_abline(lty = 2, color = "grey60") +
    scale_color_manual(values = model_colors) +
    xlim(0, 1) + ylim(0, 1) +
    labs(title = paste("B. Calibration -", dataset_name_label), x = "Predicted Prob", y = "Observed Prop", color = "Model") +
    individual_theme +
    guides(color = guide_legend(nrow = 4))
  
  # --- C. DCA Curve (with 6-row legend) ---
  dca_data <- dataset %>% mutate(y = as.numeric(as.character(LPA_Class))) %>% dplyr::select(y)
  for (name in target_names) {
    dca_data[[name]] <- predict(models_list[[name]], dataset, type = "prob")$.pred_1
  }
  dca_res <- dca(y ~ ., data = dca_data)
  
  p_dca <- as_tibble(dca_res) %>%
    mutate(label = case_when(variable == "all" ~ "All", variable == "none" ~ "None", TRUE ~ label)) %>%
    ggplot(aes(x = threshold, y = net_benefit, color = label)) +
    geom_line(linewidth = 0.8) +
    scale_color_manual(values = model_colors) +
    coord_cartesian(ylim = c(-0.01, NA)) +
    labs(title = paste("C. DCA -", dataset_name_label), x = "Threshold Prob", y = "Net Benefit", color = "Model") +
    individual_theme +
    guides(color = guide_legend(nrow = 6))
  
  
  aligned_plot <- (p_roc | p_cal | p_dca) +
    plot_annotation(
      tag_levels = 'A',
      title = paste("Model Performance Evaluation -", dataset_name_label),
      theme = theme(plot.title = element_text(size = 14, face = "bold", hjust = 0.5))
    )
  
  return(aligned_plot)
}

# Call function
final_aligned_test <- generate_aligned_panel(fitted_models, test_data, "Testing Set")
final_aligned_train <- generate_aligned_panel(fitted_models, train_data, "Training Set")

print(final_aligned_test)
print(final_aligned_train)
