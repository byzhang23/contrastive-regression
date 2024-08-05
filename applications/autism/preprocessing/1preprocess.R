library(dplyr)
library(data.table)
library(ggplot2)
library(RColorBrewer)
library(Matrix)
library(Seurat)
setwd('')

meta <- fread('./raw_data/autism/meta.tsv')
coord <- fread('./raw_data/autism/tMinusSNE.coords.tsv')
colnames(coord) <- c('cell','tSNE_1','tSNE_2')
cell_meta <- inner_join(meta,coord) %>% data.frame
rownames(cell_meta) <- cell_meta$cell
colnames(cell_meta)
saveRDS(cell_meta, './proc_data/autism/cell_meta.rds')
nclu <- length(unique(cell_meta$cluster))

sample_meta <- cell_meta %>%
  select(sample, individual, region, age, sex, diagnosis, Capbatch, Seqbatch) %>% unique
rownames(sample_meta) <- NULL
saveRDS(sample_meta, './proc_data/autism/sample_meta.rds')

# umap
png('./result/autism/umap.png', width = 7,height = 5,units = 'in',res = 300)
ggplot(cell_meta,aes(x=tSNE_1,y=tSNE_2, color = cluster)) +
  geom_point(shape = '.', alpha = 0.8) +
  scale_color_manual(values = rainbow(17)) +
  theme_classic() +
  guides(color = guide_legend(override.aes = list(shape = 19)))
dev.off()

# create seurat
count <- Read10X(data.dir = './raw_data/autism/count/')
count <- count[rowSums(count)>0,]
norm <- fread("./raw_data/autism/exprMatrix.tsv.gz")
genes <- gsub(".+[|]", "", norm[,1][[1]])
length(genes)
count <- count[rownames(count) %in% genes,]
dim(count) #36224 104559
identical(colnames(count),meta$cell)

asd <- CreateSeuratObject(counts = count, project = "asd", meta.data = cell_meta)
saveRDS(asd, './proc_data/autism/asd_seurat.rds')

# pseudobulk per cell type
quantile(colSums(count)) # median: 2211 (1e3 magnitude)


construct_pseudobulk <- function(expr, cell_meta, scale.factor = 1e3){
  
  barcode <- cell_meta$cell
  sample_name <- cell_meta$sample
  sample_name <- factor(sample_name,levels = unique(sample_name))
  expr_subset <- expr[,barcode]
  expr_subset <- Matrix::Matrix(expr_subset,sparse = T)
  
  if(length(levels(sample_name))>1){
    mod <- model.matrix(~ 0 + sample_name)
    colnames(mod) <- levels(sample_name)
    ct <- expr_subset %*% mod
  }else{
    ct <- expr_subset %*% matrix(1,nrow = ncol(expr_subset),ncol = 1,dimnames = list(NULL,unique(levels(sample_name))))
  }
  
  ## filter out genes with 0 count (CPM normalization)
  rc <- Matrix::colSums(ct)/scale.factor
  denom0 <- which(rc==0)
  if(length(denom0)>0){
    warning(paste0('There are ', length(denom0),'genes having zero expression across all samples. Proceed with excluding these genes.'))
    ct <- ct[,-denom0,drop=F]
    rc <- rc[-denom0]
  }
  res <- log2(Matrix::t(Matrix::t(ct)/rc + 1)) %>% as.matrix
  
  return(list(pb_count = ct,
              pb_norm = res))
}

asd.count <- asd@assays$RNA@counts
total_clu <- unique(cell_meta$cluster)
pb.count <- pb.norm <- list()

min.prop <- 0.01
for(i in 1:length(total_clu)){
  print('===')
  print(i)
  clu <- total_clu[i]
  sub.cell <- cell_meta %>% filter(cluster == clu)
  sub.count <- asd.count[,sub.cell$cell]
  
  # filter out lowly expressed genes
  sub.count <- sub.count[rowMeans(sub.count > 0) >= min.prop, ]
  print(dim(sub.count))
  
  pb.ls <- construct_pseudobulk(expr = sub.count,
                                cell_meta = sub.cell,
                                scale.factor = 1e3)
  pb.count[[i]] <- pb.ls$pb_count
  saveRDS(pb.count[[i]], paste0('./result/autism_filter/pb/', gsub('\\/','or',total_clu[i]), '_count.rds'))
  pb.norm[[i]] <- pb.ls$pb_norm
  saveRDS(pb.norm[[i]], paste0('./result/autism_filter/pb/', gsub('\\/','or',total_clu[i]), '.rds'))
  
}
# names(pb.count) <- names(pb.norm) <- total_clu
# saveRDS(pb.count, './proc_data/autism_prefilter/pb_count.rds')
# saveRDS(pb.norm, './proc_data/autism_prefilter/pb_norm.rds') # log2 CPthousands normalized

