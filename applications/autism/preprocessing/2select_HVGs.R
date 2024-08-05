library(dplyr)
library(scran)
library(ggplot2)
library(ggrepel)
setwd('./result/autism_filter/')
output_dir <- './hvg1k/'
if(!dir.exists(output_dir)) dir.create(output_dir)

select_HVGs <- function(dt, n = 1000, exclude.mito = T, exclude.ribo = T){
  
  if(exclude.mito){
    dt <- dt[!grepl('^MT-',rownames(dt)),]
  }
  if(exclude.ribo){
    dt <- dt[!grepl('^RP[SL]',rownames(dt)),]
  }
  dec <- modelGeneVar(dt)
  
  # Get the top 1000 genes.
  hvg <- getTopHVGs(dec, n=n)
  return(hvg)
}

files <- list.files('./pb/','rds')
scoreA <- readRDS('scoreA_pfc.rds')
for(f in files){
  print(f)
  ct <- gsub('pb_norm_|\\.rds','',f)
  dat <- readRDS(paste0('./pb/',f))
  if(length(setdiff(scoreA$sample, colnames(dat))) == 0){
    dat <- dat[,scoreA$sample]
    dim(dat)
    hvg <- select_HVGs(dat, n = 1000, exclude.mito = T, exclude.ribo = T)
    print(quantile(rowMeans(dat[hvg,]>0)))
    saveRDS(hvg,paste0(output_dir,ct,'.rds'))
    
    # PCA
    pca <-prcomp(t(dat[hvg,]), scale=T)$x[,1:2]
    dat.pca <- inner_join(scoreA, data.frame(sample = rownames(pca),pca))
    
    p1 <- ggplot(dat.pca, aes(x = PC1, y= PC2, color = diagnosis)) +
      geom_point() +
      theme_classic(base_size = 12) +
      scale_color_manual(breaks = c('ASD','Control'),
                         values = c('#AF46B4','#4BB446')) 
    ggsave(paste0(output_dir,'pca_scoreA_',ct,'.pdf'),p1, width = 5, height = 4)
    
    p2 <- p1 + geom_text_repel(aes(label = sample), size = 2)
    ggsave(paste0(output_dir,'pca_scoreA_',ct,'_annotate.pdf'),p2, width = 5, height = 4)
    
    
  }else{
    print('skip')
  }
  
}


# L2or3: remove outliers
samples.remove <- c('6033_BA9','5976_BA9','5978_BA9')
f <- 'L2or3.rds'
ct <- 'L2or3'
dat <- readRDS(paste0('./pb/',f))
if(length(setdiff(scoreA$sample, colnames(dat))) == 0){
  dat <- dat[,setdiff(scoreA$sample,samples.remove)]
  dim(dat)
  hvg <- select_HVGs(dat, n = 1000, exclude.mito = T, exclude.ribo = T)
  print(quantile(rowMeans(dat[hvg,]>0)))
  saveRDS(hvg,paste0(output_dir,ct,'_rm_outlier.rds'))
  
  # PCA
  pca <-prcomp(t(dat[hvg,]), scale=T)$x[,1:2]
  dat.pca <- inner_join(scoreA, data.frame(sample = rownames(pca),pca))
  
  p1 <- ggplot(dat.pca, aes(x = PC1, y= PC2, color = diagnosis)) +
    geom_point() +
    theme_classic(base_size = 12) +
    scale_color_manual(breaks = c('ASD','Control'),
                       values = c('#AF46B4','#4BB446')) 
  ggsave(paste0(output_dir,'pca_scoreA_',ct,'_rm_outlier.pdf'),p1, width = 5, height = 4)
  
  p2 <- p1 + geom_text_repel(aes(label = sample), size = 2)
  ggsave(paste0(output_dir,'pca_scoreA_',ct,'_rm_outlier_annotate.pdf'),p2, width = 5, height = 4)
  
  
}
