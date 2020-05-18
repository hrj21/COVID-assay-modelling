
# Setup and restore R environment -----------------------------------------

#renv::init() # initialize r enironment file in current project
renv::restore() # install all packages listed in the environment lock file

# Load packages -----------------------------------------------------------

library(tidyverse)
library(mlr)
library(parallel)
library(parallelMap)
library(GGally)

renv::snapshot()

# Read the data -----------------------------------------------------------

master <- read_csv("data/Master_data.csv")

master

glimpse(master)

# Split into bead and ELISA -----------------------------------------------

data_by_assay <- master %>%
  pivot_longer(cols = c(-Lab_ref, -Serum_date, -Label), 
               names_to = "Feature",
               values_to = "Value") %>%
  separate(Feature, into = c("Isotype", "Protein_target", "Assay")) %>%
  group_by(Assay) %>%
  group_split()

bead <- data_by_assay[[1]] %>%
  unite(col = "Reactivity", Protein_target, Isotype, sep = "_") %>%
  pivot_wider(id_cols = c(Lab_ref, Label), 
              names_from = Reactivity, 
              values_from = Value)

ELISA <- data_by_assay[[2]] %>%
  unite(col = "Reactivity", Protein_target, Isotype, sep = "_") %>%
  pivot_wider(id_cols = c(Lab_ref, Label), 
              names_from = Reactivity, 
              values_from = Value)

cell <- data_by_assay[[3]] %>%
  unite(col = "Reactivity", Protein_target, Isotype, sep = "_") %>%
  pivot_wider(id_cols = c(Lab_ref, Label), 
              names_from = Reactivity, 
              values_from = Value)

# Plot data ---------------------------------------------------------------

theme_elements <- list(theme_bw(),
                       theme(panel.grid.major.x = element_blank(),
                             panel.grid.minor.y = element_blank(),
                             panel.spacing = unit(0, "lines")),
                       scale_color_brewer(type = "qual", palette = "Set1"),
                       scale_fill_brewer(type = "qual", palette = "Set1"))

cell %>%
  ggplot(aes(IgG_Perc, IgGAM_Perc, fill = Label)) +
  geom_point(size = 2, shape = 21, col = "black") +
  labs(y = "Percentage IgG+ IgA+ IgM+", 
       x = "Percentage IgG+", 
       title = "IgG only reactivity vs. combined IgG + IgA + IgM reactivity",
       subtitle = "Cell assay") +
  theme_elements

ggsave("plots/Cells.png", width = 12, height = 6) 

# Create task -------------------------------------------------------------

cell_for_models <- select(cell, -Lab_ref)

cell_task <- makeClassifTask(data = cell_for_models, target = "Label")

# Make base learners ------------------------------------------------------

lda <- makeLearner("classif.lda")
logreg <- makeLearner("classif.logreg")
svm <- makeLearner("classif.svm")
xgb <- makeLearner("classif.xgboost", par.vals = list("nrounds" = 20))

# Define 3 fold cross-validation -----------------------------------------

k_fold <- makeResampleDesc("CV", iters = 3, stratify = TRUE)
holdout <- makeResampleDesc("Holdout", stratify = TRUE)

# Define hyperparameter search space --------------------------------------

kernels <- c("polynomial", "radial", "sigmoid")

svm_hyper_space <- makeParamSet(
  makeDiscreteParam("kernel", values = kernels),
  makeIntegerParam("degree", lower = 1, upper = 3),
  makeNumericParam("cost", lower = 0.1, upper = 10),
  makeNumericParam("gamma", lower = 0.1, upper = 10)
)

xgb_hyper_space <- makeParamSet(
  makeNumericParam("eta", lower = 0, upper = 1),
  makeNumericParam("gamma", lower = 0, upper = 5),
  makeIntegerParam("max_depth", lower = 1, 5),
  makeNumericParam("min_child_weight", lower = 1, upper = 10),
  makeNumericParam("subsample", lower = 0.5, upper = 1),
  makeNumericParam("colsample_bytree", lower = 0.5, upper = 1),
  makeDiscreteParam("eval_metric", values = c("error", "logloss"))
)

# Define hyperparameter space search procedure ----------------------------

irace <- makeTuneControlIrace(maxExperiments = 200L)

# Make wrapped learners ---------------------------------------------------

feat_sel_control <- makeFeatSelControlSequential(method = "sfs")

wrapped_lda <- makeFeatSelWrapper(learner = lda,
                                  resampling = k_fold,
                                  control = feat_sel_control)

wrapped_logreg <- makeFeatSelWrapper(learner = logreg,
                                     resampling = k_fold,
                                     control = feat_sel_control)

svm_tuning <- makeTuneWrapper(learner = svm,
                              resampling = k_fold,
                              par.set = svm_hyper_space,
                              control = irace)

wrapped_svm <- makeFeatSelWrapper(learner = svm_tuning,
                                  resampling = k_fold,
                                  control = feat_sel_control)

xgb_tuning <- svm_tuning <- makeTuneWrapper(learner = xgb,
                                            resampling = k_fold,
                                            par.set = xgb_hyper_space,
                                            control = irace)

wrapped_xgb <- makeFeatSelWrapper(learner = xgb_tuning,
                                  resampling = k_fold,
                                  control = feat_sel_control)

# Train the models --------------------------------------------------------

learners <- list(wrapped_lda, wrapped_logreg, wrapped_svm, wrapped_xgb)

cell_bench_results <- benchmark(learners, cell_task, holdout, models = TRUE)

saveRDS(cell_bench_results, paste("cell_bench_results", Sys.time(), sep = "_"))

cell_bench_results <- readRDS("models/cell_bench_results_2020-05-18 10:33:03")

# Confusion matrices ------------------------------------------------------

confusion_matrices <- list(
  lda = calculateConfusionMatrix(
    cell_bench_results$results$cell_for_models$classif.lda.featsel$pred
  ),
  logreg = calculateConfusionMatrix(
    cell_bench_results$results$cell_for_models$classif.logreg.featsel$pred
  ),
  svm = calculateConfusionMatrix(
    cell_bench_results$results$cell_for_models$classif.svm.tuned.featsel$pred
  ),
  xgb = calculateConfusionMatrix(
    cell_bench_results$results$cell_for_models$classif.xgboost.tuned.featsel$pred
  )
)

confusion_matrices

# Performance metrics -----------------------------------------------------

performance_metrics <- function(confusion) {
  tibble(
    Accuracy = (confusion$result[1, 1] + confusion$result[2, 2]) /
      sum(confusion$result[1:2, 1:2]),
    # FPR: proportion of all negs yielding positive results
    FPR = confusion$result[1, 2] / sum(confusion$result[1, 1:2]), 
    # FNR: proportion of all pos yielding negative reuslt
    FNR = confusion$result[2, 1] / sum(confusion$result[2, 1:2]), 
    # Precision: proportion of truly neg we detect as neg
    Precision = confusion$result[2, 2] / 
      (confusion$result[2, 2] + confusion$result[1, 2]),
    # Recall/sensitivity: proportion of truly pos we detect as pos
    Recall =  confusion$result[2, 2] / 
      (confusion$result[2, 2] + confusion$result[2, 1]),
    F1 = 2 / ((1 / Precision) + (1 / Recall)),
  )
}

confusion_matrices
metrics <- map_df(confusion_matrices, performance_metrics)
metrics$Model <- names(confusion_matrices)
metrics

# Inspect final models ----------------------------------------------------

cell_models <- getBMRModels(cell_bench_results, drop = TRUE)

getFeatSelResult(cell_models$classif.lda.featsel[[1]])
getFeatSelResult(cell_models$classif.logreg.featsel[[1]])
getFeatSelResult(cell_models$classif.svm.tuned.featsel[[1]])
getFeatSelResult(cell_models$classif.xgboost.tuned.featsel[[1]])

# Visualising decision boundary -------------------------------------------

cell_grid <- expand.grid(seq(0, 50, 0.1), seq(0, 100, 0.2))
names(cell_grid) <- c("IgG_Perc", "IgGAM_Perc")

cell_grid %>%
mutate(LDA = predict(cell_models[[1]][[1]], newdata = .)$data[, 1],
       LogReg = predict(cell_models[[2]][[1]], newdata = .)$data[, 1],
       SVM = predict(cell_models[[3]][[1]], newdata = .)$data[, 1],
       XGB = predict(cell_models[[4]][[1]], newdata = .)$data[, 1]) %>%
  pivot_longer(cols = c(LDA, LogReg, SVM, XGB),
               names_to = "Model", values_to = "Label") %>%
  ggplot(aes(IgG_Perc, IgGAM_Perc, col = Label, fill = Label)) +
  facet_wrap(~ Model) +
  geom_raster(alpha = 0.5) +
  labs(y = "Percentage IgA+ / IgM+", 
       x = "Percentage IgG+", 
       title = "Visualising decision boundaries for each model",
       subtitle = "Cell assay") +
  coord_cartesian(expand = FALSE) +
  geom_point(data = cell, shape = 21, col = "black") +
  theme_elements

ggsave("plots/Cells_decision_boundaries.png", width = 12, height = 6) 
