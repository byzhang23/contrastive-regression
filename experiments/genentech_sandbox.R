library(dplyr)
library(magrittr)
library(HDF5Array)
library(SingleCellExperiment)

DATA_DIR <- "/Users/andrewjones/Documents/beehive/contrastive_regression/contrastive-regression/data/genentech"

## Small intestine data
data <- readRDS(file.path(DATA_DIR, "small_intestine_data2.RDS"))
# Matrix::writeMM(obj=data@assays@data$counts, file=file.path(DATA_DIR, "small_intestine_data2_counts.txt"))
# data <- readRDS("~/Documents/beehive/contrastive_regression/contrastive-regression/data/genentech/alveolarsphere_data2.RDS")

data@assays@data$counts

gene_vars <- sparseMatrixStats::rowVars(data@assays@data$logcounts)
sorted_idx <- order(-gene_vars)

data_subset <- data@assays@data$logcounts[sorted_idx[1:500],] %>% 
  t()

cell_types <-  data@colData$cell_type

old_idx <- which(data@colData$Age == "Old")
young_idx <- which(data@colData$Age == "Young")

data_old <- data_subset[old_idx,]
data_young <- data_subset[young_idx,]

cell_types_old = cell_types[old_idx]
cell_types_young = cell_types[young_idx]

data_old %>% as.matrix() %>% write.csv(file = file.path(DATA_DIR, "small_intestine_old_counts.csv"))
data_young %>% as.matrix() %>% write.csv(file = file.path(DATA_DIR, "small_intestine_young_counts.csv"))

cell_types_old %>% write.csv(file = file.path(DATA_DIR, "small_intestine_old_cell_types.csv"))
cell_types_young %>% write.csv(file = file.path(DATA_DIR, "small_intestine_young_cell_types.csv"))


## Alveolarsphere data
data <- readRDS(file.path(DATA_DIR, "alveolarsphere_data2.RDS"))

gene_vars <- sparseMatrixStats::rowVars(data@assays@data$logcounts)
sorted_idx <- order(-gene_vars)

data_subset <- data@assays@data$logcounts[sorted_idx[1:500],] %>% 
  t()

cell_types <-  data@colData$cell_type

old_idx <- which(data@colData$Age == "Old")
young_idx <- which(data@colData$Age == "Young")

data_old <- data_subset[old_idx,]
data_young <- data_subset[young_idx,]

cell_types_old = cell_types[old_idx]
cell_types_young = cell_types[young_idx]

data_old %>% as.matrix() %>% write.csv(file = file.path(DATA_DIR, "alveolarsphere_old_counts.csv"))
data_young %>% as.matrix() %>% write.csv(file = file.path(DATA_DIR, "alveolarsphere_young_counts.csv"))

cell_types_old %>% write.csv(file = file.path(DATA_DIR, "alveolarsphere_old_cell_types.csv"))
cell_types_young %>% write.csv(file = file.path(DATA_DIR, "alveolarsphere_young_cell_types.csv"))
