
# Setup and restore R environment -----------------------------------------

#renv::init() # initialize r enironment file in current project
renv::restore() # install all packages listed in the environment lock file

# Load packages -----------------------------------------------------------

library(tidyverse)
library(mlr)
library(parallel)
library(parallelMap)
library(GGally)
library(patchwork)

# Read the data -----------------------------------------------------------

bead <- read_csv("data/20200512_BEADS_Analysis.csv")

bead

glimpse(bead)

# Restructure for plotting ------------------------------------------------

long <- bead %>%
  pivot_longer(cols = c(-Sample, -Label, -Plate),
               names_to = "Reactivity",
               values_to = "Fold_change") %>%
  
  separate(Reactivity, 
           into = c("Protein_target", "Isotype"), 
           sep = "_") %>%
  
  mutate(Isotype = factor(Isotype,
                          levels = c("IgG", "IgM", "IgA")),
         Protein_target = factor(Protein_target, 
                                 levels = c("S1", "N", "RBD"))) %>%
  
  arrange(Isotype, Protein_target, Fold_change) %>%
  
  mutate(Short_ID = group_indices(., Sample) %>% factor(levels = unique(.)))

# Plot data ---------------------------------------------------------------

theme_elements <- list(theme_bw(),
                       theme(panel.grid.major.x = element_blank(),
                             panel.grid.minor.y = element_blank(),
                             panel.spacing = unit(0, "lines")),
                       scale_color_brewer(type = "qual", palette = "Set1"),
                       scale_fill_brewer(type = "qual", palette = "Set1"))

long %>%
  filter(Label != "SPECIAL") %>%
  ggplot(aes(Short_ID, Fold_change, col = Isotype)) +
  facet_grid(Protein_target ~ Label, scales = "free_x", space = "free_x") +
  geom_point() +
  labs(y = "Fold change from secondary only", 
       x = "Patient", 
       title = "Reactivity for each patient, ordered by S1 IgG reactivity",
       subtitle = "Common y axis") +
  theme_elements + 
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

ggsave("plots/Beads_common_axis.png", width = 12, height = 6)  

long %>%
  filter(Label != "SPECIAL") %>%
  ggplot(aes(Short_ID, Fold_change, col = Isotype)) +
  facet_grid(Protein_target ~ Label, scales = "free", space = "free_x") +
  geom_point() +
  theme_bw() +
  labs(y = "Fold change from secondary only", 
       x = "Patient", 
       title = "Reactivity for each patient, ordered by S1 IgG reactivity",
       subtitle = "Distinct y axis per protein target") +
  theme_elements + 
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

ggsave("plots/Beads_distinct_axes.png", width = 12, height = 6) 

long %>%
  filter(Label != "SPECIAL") %>%
  ggplot(aes(Short_ID, log2(Fold_change), col = Isotype)) +
  facet_grid(Protein_target ~ Label, scales = "free", space = "free_x") +
  geom_point() +
  theme_bw() +
  labs(y = "Log2 fold change from secondary only", 
       x = "Patient", 
       title = "Reactivity for each patient, ordered by S1 IgG reactivity",
       subtitle = "Distinct log2-transformed y axis per protein target") +
  theme_elements + 
  theme(axis.text.x = element_blank(), axis.ticks.x = element_blank())

ggsave("plots/Beads_log2_axis.png", width = 12, height = 6) 

# Create task -------------------------------------------------------------

data_for_models <- filter(bead, Label != "SPECIAL") %>% select(-Sample, -Plate)

bead_task <- makeClassifTask(data = data_for_models, target = "Label")

# Make base learners ------------------------------------------------------

lda <- makeLearner("classif.lda")
logreg <- makeLearner("classif.logreg")
svm <- makeLearner("classif.svm")
xgb <- makeLearner("classif.xgboost", par.vals = list("nrounds" = 20))

# Define 10 fold cross-validation -----------------------------------------

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

resample(wrapped_xgb, bead_task, holdout)

learners <- list(wrapped_lda, wrapped_logreg, wrapped_svm, wrapped_xgb)

bench_results <- benchmark(learners, bead_task, holdout, models = TRUE)

saveRDS(bench_results, paste("models/bench_results", Sys.time(), sep = "_"))

bench_results <- readRDS("models/bench_results_2020-05-16")

# Confusion matrices ------------------------------------------------------

confusion_matrices <- list(
  lda = calculateConfusionMatrix(
    bench_results$results$data_for_models$classif.lda.featsel$pred
  ),
  logreg = calculateConfusionMatrix(
    bench_results$results$data_for_models$classif.logreg.featsel$pred
  ),
  svm = calculateConfusionMatrix(
    bench_results$results$data_for_models$classif.svm.tuned.featsel$pred
  ),
  xgb = calculateConfusionMatrix(
    bench_results$results$data_for_models$classif.xgboost.tuned.featsel$pred
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

metrics <- map_df(confusion_matrices, performance_metrics)
metrics$Model <- names(confusion_matrices)
metrics

# Inspect final models ----------------------------------------------------
    
models <- getBMRModels(bench_results, drop = TRUE)

getFeatSelResult(models$classif.lda.featsel[[1]])
getFeatSelResult(models$classif.logreg.featsel[[1]])
getFeatSelResult(models$classif.svm.tuned.featsel[[1]])
getFeatSelResult(models$classif.xgboost.tuned.featsel[[1]])

# Visualising decision boundary -------------------------------------------

lda_boundary <- expand.grid(seq(0, 60, 0.2), seq(0, 60, 0.2)) %>%
  rename("N_IgG" = Var1, "RBD_IgA" = Var2) %>%
  mutate(Label = predict(models[[1]][[1]], newdata = .)$data[, 1]) %>%
  ggplot(aes(N_IgG, RBD_IgA, fill = Label)) +
  geom_raster(alpha = 0.5) +
  geom_point(data = filter(bead, Label != "SPECIAL"), shape = 21, col = "black") +
  labs(x = "Fold change in N protein IgG reactivity", 
       y = "Fold change in RBD protein IgA reactivity", 
       title = "LDA decision boundary") +
  coord_cartesian(ylim = c(0, 60), xlim = c(0, 60), expand = FALSE) +
  theme_elements

logreg_boundary1 <- expand.grid(seq(0, 1, 1), seq(0, 60, 0.2), seq(0, 60, 0.2)) %>%
  rename("S1_IgM" = Var1, "N_IgG" = Var2, "RBD_IgA" = Var3) %>%
  mutate(Label = predict(models[[2]][[1]], newdata = .)$data[, 1]) %>%
  ggplot(aes(N_IgG, RBD_IgA, fill = Label)) +
  geom_raster(alpha = 0.5) +
  geom_point(data = filter(bead, Label != "SPECIAL"), shape = 21, col = "black") +
  labs(x = "Fold change in N protein IgG reactivity", 
       y = "Fold change in RBD protein IgA reactivity", 
       title = "Logreg decision boundary") +
  coord_cartesian(ylim = c(0, 60), xlim = c(0, 60), expand = FALSE) +
  theme_elements

logreg_boundary2 <- expand.grid(seq(0, 60, 0.2), seq(0, 60, 0.2), seq(0, 1, 1)) %>%
  rename("S1_IgM" = Var1, "N_IgG" = Var2, "RBD_IgA" = Var3) %>%
  mutate(Label = predict(models[[2]][[1]], newdata = .)$data[, 1]) %>%
  ggplot(aes(S1_IgM, N_IgG, fill = Label)) +
  geom_raster(alpha = 0.5) +
  geom_point(data = filter(bead, Label != "SPECIAL"), shape = 21, col = "black") +
  labs(y = "Fold change in N protein IgG reactivity", 
       x = "Fold change in S protein IgM reactivity", 
       title = "Logreg decision boundary") +
  coord_cartesian(ylim = c(0, 60), xlim = c(0, 60), expand = FALSE) +
  theme_elements

svm_boundary <- expand.grid(seq(0, 60, 0.2), seq(0, 60, 0.2)) %>%
  rename("N_IgG" = Var1, "RBD_IgM" = Var2) %>%
  mutate(Label = predict(models[[3]][[1]], newdata = .)$data[, 1]) %>%
  ggplot(aes(N_IgG, RBD_IgM, fill = Label)) +
  geom_raster(alpha = 0.5) +
  geom_point(data = filter(bead, Label != "SPECIAL"), shape = 21, col = "black") +
  labs(x = "Fold change in N protein IgG reactivity", 
       y = "Fold change in RBD protein IgM reactivity", 
       title = "SVM decision boundary") +
  coord_cartesian(ylim = c(0, 60), xlim = c(0, 60), expand = FALSE) +
  theme_elements

xgb_boundary <- expand.grid(seq(0, 60, 0.2), seq(0, 60, 0.2)) %>%
  rename("N_IgG" = Var1, "RBD_IgM" = Var2) %>%
  mutate(Label = predict(models[[4]][[1]], newdata = .)$data[, 1]) %>%
  ggplot(aes(N_IgG, RBD_IgM, fill = Label)) +
  geom_raster(alpha = 0.5) +
  geom_point(data = filter(bead, Label != "SPECIAL"), shape = 21, col = "black") +
  labs(x = "Fold change in N protein IgG reactivity", 
       y = "Fold change in RBD protein IgM reactivity", 
       title = "XGB decision boundary") +
  coord_cartesian(ylim = c(0, 60), xlim = c(0, 60), expand = FALSE) +
  theme_elements

((lda_boundary | (logreg_boundary1 + logreg_boundary2)) / 
  (svm_boundary | xgb_boundary)) + 
  plot_layout(guides = "collect")

ggsave("plots/Beads_decision_boundaries.png", width = 14, height = 7) 

# Reorder data ------------------------------------------------------------

long %>%
  mutate(Isotype = factor(Isotype,
                          levels = c("IgG", "IgM", "IgA")),
         Protein_target = factor(Protein_target, 
                                 levels = c("N", "S1", "RBD"))) %>%
  
  arrange(Isotype, Protein_target, Fold_change) %>%
  
  mutate(Short_ID = factor(Short_ID, levels = unique(Short_ID))) %>%
  
  filter(Label != "SPECIAL") %>%
  ggplot(aes(Short_ID, log2(Fold_change), col = Isotype)) +
  facet_grid(Protein_target ~ Label, scales = "free", space = "free_x") +
  geom_point() +
  labs(y = "Fold change from secondary only", 
       x = "Patient", 
       title = "Reactivity for each patient, ordered by N IgG reactivity",
       subtitle = "Distinct log2-transformed y axis per protein target") +
  theme_elements

ggsave("plots/Beads_ordered_by_N_IgG.png", width = 12, height = 6) 

# Extensively cross-validate XGBOOST model --------------------------------

xgb_selected_features <- filter(bead, Label != "SPECIAL") %>% 
  select(Label, N_IgM, RBD_IgM)

xgb_task <- makeClassifTask(data = xgb_selected_features, target = "Label")

repCV <- makeResampleDesc("RepCV", stratify = TRUE, folds = 10, reps = 100)

tuned_xgb <- makeLearner("classif.xgboost",
                         par.vals = list(
                           "eta" = 0.106,
                           "gamma" = 2.07, 
                           "max_depth" = 1, 
                           "min_child_weight" = 2.86, 
                           "subsample" = 0.743, 
                           "colsample_bytree" = 0.679, 
                           "eval_metric" = "error",
                           "nrounds" = 20
                         ))

xgb_cv <- resample(tuned_xgb, xgb_task, 
                   models = TRUE, 
                   resampling = repCV)

calculateConfusionMatrix(xgb_cv$pred) 

calculateConfusionMatrix(xgb_cv$pred) %>%
  performance_metrics()

# Extensively cross-validate SVM model ------------------------------------

svm_selected_features <- filter(bead, Label != "SPECIAL") %>% 
  select(Label, N_IgG, RBD_IgM)

svm_task <- makeClassifTask(data = svm_selected_features, target = "Label")

tuned_svm <- makeLearner("classif.svm",
                         par.vals = list(
                           "kernel" = "radial",
                           "cost" = 4.68,
                           "gamma" = 8.33
                         ))

svm_cv <- resample(tuned_svm, svm_task, 
                   models = TRUE, 
                   resampling = repCV)

calculateConfusionMatrix(svm_cv$pred) 

calculateConfusionMatrix(svm_cv$pred) %>%
  performance_metrics()

