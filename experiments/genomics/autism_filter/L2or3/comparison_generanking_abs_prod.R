library(dplyr)
library(pheatmap)
library(ggplot2)
library(tidyr)
library(ggpubr)
library(ggrepel)
library(gridExtra)
library(GGally)

setwd('./result/autism_prefilter/L2or3/')
ct <-  'L2or3' # cell type
score <- 'scoreA' # score

output_dir <- paste0('./',score,'/')

score <- readRDS(paste0('../',score,'_pfc.rds')) %>% mutate(group = ifelse(diagnosis == 'ASD', 1, 0))
w <- readRDS(paste0(output_dir,'W.rds')) %>% as.matrix
rownames(w) <- paste0('Dim',1:nrow(w))
t <- readRDS(paste0(output_dir,'t.rds')) %>% as.matrix
rownames(t) <- score$sample[score$diagnosis=='ASD']
s <- readRDS(paste0(output_dir,'S.rds')) %>% as.matrix
rownames(s) <- paste0('Dim',1:nrow(w))
beta <- readRDS(paste0(output_dir,'beta.rds')) %>% unlist %>% as.numeric
names(beta) <- paste0('Dim',1:nrow(w))
pb <- readRDS(paste0('../pb/',ct,'.rds'))


beta <- as.matrix(beta)

prod <- t(w) %*% beta
colnames(prod) <- 'product'

dt.prod <- prod %>% data.frame %>% mutate(gene = rownames(prod)) %>%
  arrange(-product) %>% mutate(rank=1:nrow(prod))
write.csv(dt.prod, paste0(output_dir,'gene_rank_prod_beta_w.csv'),row.names = F)

dt <- read.csv(paste0(output_dir,'gene_rank_max_abs_beta.csv')) %>% mutate(method = 'abs')
dt.prod$method <- 'prod'

dt.all <- inner_join(dt %>% select(gene, rank) %>% dplyr::rename(abs_rank = rank), 
                     dt.prod %>% select(gene, rank) %>% dplyr::rename(prod_rank = rank))

ggplot(dt.all, aes(x=abs_rank, y=prod_rank)) +
  geom_point() + 
  geom_point(data = head(dt.all, 10), color = 'blue') +
  labs(x = 'Gene ranking (Current procedure)', y = 'Gene ranking (W^Tbeta)') +
  theme_classic(base_size = 12)
ggsave(paste0(output_dir,'gene_rank_prod_beta_w.png'), height = 3, width = 3)


