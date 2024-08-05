# comparison with cplvm + reg
library(dplyr)
library(ggplot2)

ct <-  'L2or3' # cell type
score <- 'scoreA' # score
output_dir <- paste0('./result/autism_filter/L2or3/',score,'_cplvm/')

score <- readRDS(paste0('./result/autism_filter/',score,'_pfc.rds')) %>% mutate(group = ifelse(diagnosis == 'ASD', 1, 0))
cplvm.w <- read.csv('./result/autism_filter/L2or3/scoreA_cplvm/W_fg_loadings.csv')
rownames(cplvm.w) <- cplvm.w[,1]
cplvm.w <- cplvm.w[,-1]
colnames(cplvm.w) <- paste0('Dim',1:ncol(cplvm.w))
cplvm.t <- read.csv('./result/autism_filter/L2or3/scoreA_cplvm/t_fg_factors.csv')[,-1]
colnames(cplvm.t) <- gsub('^X','',colnames(cplvm.t))
rownames(cplvm.t) <- paste0('Dim',1:ncol(cplvm.w))
cplvm.t <- t(cplvm.t)

t_annot <- suppressMessages(inner_join(data.frame(sample = rownames(cplvm.t)),
                                           score %>% filter(diagnosis == 'ASD') %>% select(-c(individual))))
rownames(t_annot) <- t_annot$sample


if(identical(rownames(t_annot), rownames(cplvm.t))){
  dt <- cbind(t_annot %>% select(sample,zscore),cplvm.t)
}
cplvm.fit <- lm(zscore~., data = dt[,-1])
summary(cplvm.fit) # R^2=0.5
coef(cplvm.fit)
which.max(abs(coef(cplvm.fit))) # Dim3 (max of absolute value of beta)
id_max <- names(which.max(abs(coef(cplvm.fit))))

max.rank <- data.frame(gene = rownames(cplvm.w), w = cplvm.w[,id_max]) %>% arrange(-w) %>% mutate(cplvm_rank = 1:1000)
clr.rank <- read.csv('./result/autism_filter/L2or3/scoreA/gene_rank_max_abs_beta.csv') %>% rename(clr_rank = rank)
comp <- inner_join(max.rank %>% select(gene, cplvm_rank),
                   clr.rank %>% select(gene, clr_rank))

png('./result/autism_filter/L2or3/scoreA_cplvm/compare_clr_cplvm.png', width = 5, height = 5, res = 300, units = 'in')
ggplot(comp, aes(x = clr_rank, y=cplvm_rank)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, lwd = 1, color = 'red', linetype = 'dashed') +
  theme_classic(base_size = 15) +
  labs(x = 'Gene ranking (CLR)', y = 'Gene ranking (CPLVM + Regression)', title = 'Correlation = 0.21') +
  theme(plot.title = element_text(hjust = 0.5))
dev.off()

cor(comp$cplvm_rank,comp$clr_rank, method = 's') # 0.21
