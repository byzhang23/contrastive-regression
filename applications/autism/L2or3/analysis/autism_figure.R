library(dplyr)
library(pheatmap)
library(ggplot2)
library(tidyr)
library(ggpubr)
library(ggrepel)
library(gridExtra)
library(GGally)


### Read in contrastive regression results
ct <-  'L2or3' # cell type
score <- 'scoreA' # score

output_dir <- paste0('./result/autism_filter/', ct, '/',score,'/')

score <- readRDS(paste0('./result/autism_filter/',score,'_pfc.rds'))
w <- readRDS(paste0(output_dir,'W.rds')) %>% as.matrix
rownames(w) <- paste0('Dim',1:nrow(w))
t <- readRDS(paste0(output_dir,'t.rds')) %>% as.matrix
rownames(t) <- score$sample[score$diagnosis=='ASD']
s <- readRDS(paste0(output_dir,'S.rds')) %>% as.matrix
rownames(s) <- paste0('Dim',1:nrow(w))
beta <- readRDS(paste0(output_dir,'beta.rds')) %>% unlist %>% as.numeric
names(beta) <- paste0('Dim',1:nrow(w))
pb <- readRDS(paste0('./result/autism_filter/pb/',ct,'.rds'))

message(paste0('Pick the largest absolute value of beta: ', names(sort(abs(beta)))[length(beta)]))
sort(abs(beta))
t_row_annot <- suppressMessages(inner_join(data.frame(sample = rownames(t)),
                                           score %>% filter(diagnosis == 'ASD') %>% select(-c(individual))))
rownames(t_row_annot) <- t_row_annot$sample

### Heatmap: visualizing latent facor t and loading W

# latent factor t
pheatmap(t,annotation_row = t_row_annot[,-1])
# loading W
pheatmap(w,fontsize_row=10,fontsize_col = 2)

### Top 20 genes
# choose a dimension of interest (largest absolute value of beta coef)
target.dim <- names(beta)[which.max(abs(beta))]
print(target.dim)
dt <- w[target.dim,] %>% sort(decreasing = T) %>% data.frame # decreasing
dt[,2] <- rownames(dt)
dt[,3] <- 1:nrow(dt)
colnames(dt) <- c(target.dim, 'gene', 'rank')
write.csv(dt,paste0(output_dir,'gene_rank_max_abs_beta.csv'),row.names=F)


dt <- read.csv(paste0(output_dir,'gene_rank_max_abs_beta.csv'))
target.genes <- dt$gene[1:20]
head(dt,20)


# Annotate with AutDB:
  
  ```{r anontation}
# annotate with AutDB
db <- read.csv('./raw_data/autism-gene-dataset/gene-summary.csv') %>%
  select(Gene.Symbol, Molecular.Function, Support.for.Autism) %>%
  dplyr::rename(gene = Gene.Symbol)
annot <- left_join(dt,db)
head(annot %>% select(gene, rank, Molecular.Function, Support.for.Autism),20)
```

<!-- More refined search top10 genes: [https://docs.google.com/spreadsheets/d/1QGlFPwMUxXsmoiO-iwRVEdoRhw1yKqzCXILksrC4VpY/edit?usp=sharing](https://docs.google.com/spreadsheets/d/1QGlFPwMUxXsmoiO-iwRVEdoRhw1yKqzCXILksrC4VpY/edit?usp=sharing) -->
  
  ### Visualization for selected genes
  
  #### Contrastive expression (residuals)
  
  We can obtain 'contrastive expression' by minimizing the reconstruction error:
  
  ```{r}
expr <- pb[dt$gene,score$sample]
foreground_meta <- score[score$diagnosis == 'ASD',]
foreground_expr <- expr[,foreground_meta$sample]
identical(colnames(foreground_expr), foreground_meta$sample)

# PCA
pca <- prcomp(expr, scale = T)$rotation[,1:2]
if(identical(rownames(pca), score$sample)){
  ggplot(cbind(score, pca), aes(x=PC1, y=PC2, color = diagnosis)) +
    geom_point() +
    geom_text_repel(aes(label = sample)) +
    scale_color_manual(breaks = c('ASD','Control'),
                       values = c('#AF46B4','#4BB446')) +
    theme_classic()
  ggsave(paste0(output_dir, 'pca.png'), height = 5, width = 5.5)
  
}

z_foreground <- solve(s %*% t(s)) %*% s %*% (foreground_expr - t(w) %*% t(t))
background_expr <- expr[,score$diagnosis == 'Control']
z_background <- solve(s %*% t(s)) %*% s %*% (background_expr)

auxiliary_foreground <- t(s) %*% z_foreground
auxiliary_background <- t(s) %*% z_background
contrastive_foreground <- foreground_expr - auxiliary_foreground
contrastive_background <- background_expr - auxiliary_background

contrastive_comb <- cbind(contrastive_foreground,contrastive_background)
diagnosis_col <- c('#AF46B4','#4BB446'); names(diagnosis_col) <- c('ASD','Control')
identical(score$sample, colnames(contrastive_comb))
annotation_col <- score[,c('diagnosis'),drop = F]
rownames(annotation_col) <- score$sample
```

#### Heatmap

# raw expr
pheatmap(expr[target.genes,],
         annotation_col = annotation_col,
         cluster_cols = F, scale = 'row',
         annotation_colors = list(diagnosis = diagnosis_col),
         main = 'Pseudobulk expression')

# residuals
pheatmap(contrastive_comb[target.genes,],
         annotation_col = annotation_col,
         cluster_cols = F, scale = 'row',
         annotation_colors = list(diagnosis = diagnosis_col),
         main = 'Contrastive expression')


#### Scatter plot

# impute NA for plotting purpose
score.plot <- score %>% mutate(zscore_new = ifelse(is.na(zscore), min(score$zscore,na.rm = T) -1, zscore))


pb.long <- gather(t(expr[target.genes,]) %>% 
                    data.frame %>% 
                    mutate(sample = colnames(expr)),gene,expr, -c(sample)) %>% 
  inner_join(score.plot) %>% 
  # mutate(gene = factor(gene, levels = target.genes, labels = target.genes)) %>% 
  mutate(type = 'pseudobulk') 

contrastive.long <- gather(t(contrastive_comb[target.genes,]) %>% 
                             data.frame %>% 
                             mutate(sample = colnames(expr)),gene,expr, -c(sample)) %>% 
  inner_join(score.plot) %>% 
  # mutate(gene = factor(gene, levels = target.genes, labels = target.genes)) %>% 
  mutate(type = 'contrastive')

## scatter plot
p1 <- ggplot(pb.long, aes(x = zscore_new, y = expr, color = diagnosis)) +
  geom_point() +
  geom_smooth(method = 'lm') +
  theme_classic(base_size=14) +
  facet_wrap(~gene, scales = 'free_y') +
  scale_color_manual(breaks = c('ASD','Control'),
                     values = c('#AF46B4','#4BB446')) +
  labs(x = 'Z score', y = 'Pseudobulk expression')
p1
ggsave(paste0(output_dir, 'pb_scatter_top20.png'), height = 8, width = 10)
#p1 + geom_text_repel(aes(label = sample)) + theme_classic(base_size = 11)

p2 <- ggplot(contrastive.long, aes(x = zscore_new, y = expr, color = diagnosis)) +
  geom_point() +
  geom_smooth(method = 'lm') +
  theme_classic(base_size=14) +
  facet_wrap(~gene,scales = 'free_y') +
  scale_color_manual(breaks = c('ASD','Control'),
                     values = c('#AF46B4','#4BB446')) +
  labs(x = 'Z score', y = 'Contrastive expression')
p2
ggsave(paste0(output_dir, 'contrastive_scatter_top20.png'), height = 8, width = 10)
#p2 + geom_text_repel(aes(label = sample)) + theme_classic(base_size = 8)


# selected annotated genes
top20.annot <- annot %>% head(20) %>% filter(!is.na(Molecular.Function))
top20.annot
# annot.genes <- top20.annot$gene
annot.genes <- c('PTPRD','PCDH9','NRXN3','MALAT1')
clr.sub <- contrastive.long %>% filter(gene %in% annot.genes) %>% filter(diagnosis == 'ASD')
text.dt <- sapply(split(clr.sub, factor(clr.sub$gene)), function(d) round(cor(d$expr, d$zscore, method = 's'),2)) %>% data.frame
colnames(text.dt) <- 'cor'
text.dt$gene <- rownames(text.dt)
text.dt$cor <- paste0('Cor=', text.dt$cor)
text.dt


ggplot(contrastive.long %>% filter(gene %in% annot.genes), aes(x = zscore_new, y = expr, color = diagnosis)) +
  geom_point() +
  geom_smooth(method = 'lm') +
  # geom_text(data = text.dt, aes(x = 1, y = -1,label = cor, color = 'black')) +
  theme_classic(base_size=12) +
  facet_wrap(~gene ,scales = 'free_y')+
  scale_color_manual(breaks = c('ASD','Control'),
                     values = c('#AF46B4','#4BB446')) +
  labs(x = 'Z score', y = 'Contrastive expression') +
  geom_text(data = text.dt, aes(Inf, -Inf, label = cor), 
            col = "black", 
            hjust = 1, 
            vjust = -1)
ggsave(paste0(output_dir, 'contrastive_scatter_annotated.png'), height = 5, width = 6)


comb.long <- rbind(pb.long, contrastive.long)
single_gene <- 'CADM2'
illustration <- comb.long %>% filter(gene == single_gene)

ggplot(illustration, 
       aes(x = zscore_new, y = expr, shape = diagnosis, color = type)) +
  geom_smooth(method = 'lm', se = F) +
  geom_line(aes(group = sample),
            color="grey",
            #linetype = 'dashed',
            arrow = arrow(length=unit(0.075, "inches"))) +
  geom_point(aes(color = type), size = 2) +
  scale_color_manual(breaks = c('contrastive','pseudobulk'),
                     values = c('#56B4E9','#E69F00')) +
  theme_classic(base_size=14) +
  # facet_wrap(~gene) +
  labs(x = 'Z score', y = paste0(single_gene, ' expression'))
ggsave(paste0(output_dir, 'illustrate_contrast.png'), height = 5, width = 6)



# gene ranking
ggplot(dt, aes(x = rank, y = Dim2)) +
  geom_point() +
  theme_classic(base_size=14) +
  geom_text_repel(data = dt %>% head(10), aes(label = gene), color = 'blue', max.overlaps=Inf) +
  labs(x = 'Gene Index', y = 'Loading' )
ggsave(paste0(output_dir, 'loading_generank.png'), height = 3, width = 6)


### Explore Z

# (alike PCA plot, exclude from other cell types)

z_auxiliary <- t(cbind(z_foreground, z_background))
if(identical(rownames(z_auxiliary),score$sample)){
  z_auxiliary <- cbind(score, z_auxiliary)
}

gpairs_lower <- function(g){
  g$plots <- g$plots[-(1:g$nrow)]
  g$yAxisLabels <- g$yAxisLabels[-1]
  g$nrow <- g$nrow -1
  
  g$plots <- g$plots[-(seq(g$ncol, length(g$plots), by = g$ncol))]
  g$xAxisLabels <- g$xAxisLabels[-g$ncol]
  g$ncol <- g$ncol - 1
  
  g
}

(ggpairs(z_auxiliary, columns = 6:13,
         mapping=ggplot2::aes(color = diagnosis),
         lower  = list(continuous = "points"),
         upper  = list(continuous = "blank"),
         diag  = list(continuous = "blankDiag")) +
    scale_color_manual(breaks = c('ASD','Control'),
                       values = c('#AF46B4','#4BB446')) + 
    theme_bw(base_size = 14)) %>% gpairs_lower()
ggsave(paste0(output_dir, 'auxiliary_z_pairs.png'), height = 8, width = 11)
