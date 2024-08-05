## compare random seed
library(dplyr)
library(ggplot2)

d1 <- read.csv('./result/autism_filter/L2or3_random_seed/scoreA/seed100_gene_rank_max_abs_beta.csv') %>% mutate(rank_seed100 = rank)
d2 <- read.csv('./result/autism_filter/L2or3/scoreA/gene_rank_max_abs_beta.csv')

d <- inner_join(d1 %>% select(gene, rank_seed100), d2 %>% select(gene, rank))



ggplot(d, aes(x = rank,  y= rank_seed100)) +
  geom_point() +
  geom_abline(intercept = 0, slope = 1, color = 'red', lwd = 1.5, linetype = 'dashed') +
  theme_classic(base_size = 15) +
  labs(x = 'Gene ranking (seed 10)',
       y = 'Gene ranking (seed 100)')



