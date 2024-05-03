library(dplyr)
library(ggplot2)
library(ggrepel)

contrast <- read.csv('./result/autism_prefilter/L2or3/scoreA/gene_rank_max_abs_beta.csv')
pcs <- readRDS('./result/autism_prefilter/L2or3/scoreA/PCA.rds')
rownames(pcs) <- paste0('PC',1:8)

pc1 <- t(pcs)[,1,drop = F] %>% data.frame 
pc1$gene <- rownames(pc1)
pc1$abs_PC1 <- abs(pc1$PC1)

pc1 <- pc1 %>% arrange(desc(PC1)) %>% mutate(pc_rank = 1:nrow(pc1))
pc1 <- pc1 %>% arrange(desc(abs_PC1)) %>% mutate(pc_abs_rank = 1:nrow(pc1))


ttl <- inner_join(contrast, pc1)
ttl_sub <- ttl %>% filter(rank %in% c(1:10))
ggplot(ttl, aes(x=rank, y=pc_rank)) +
  geom_point() +
  geom_text_repel(data = ttl_sub, aes(label = gene), col = 'blue') + 
  labs(x = 'contrastive_rank', y='pc_rank') +
  theme_classic()


ggplot(ttl, aes(x=rank, y=pc_abs_rank)) +
  geom_point() +
  geom_text_repel(data = ttl_sub, aes(label = gene), col = 'blue') + 
  labs(x = 'contrastive_rank', y='abs(pc)_rank') +
  theme_classic()
