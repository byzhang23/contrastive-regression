##
library(dplyr)
library(ggplot2)
options(stringsAsFactors = F)
setwd('./result/autism_prefilter')

cell_types <- c('Oligodendrocytes', 'OPC','Neu-NRGN-II','AST-PP',
                  'L4','IN-VIP', 'L2or3')
score_type <- 'scoreA'

comb <- Reduce(inner_join, lapply(cell_types,function(x){
  print(x)
  r <- read.csv(paste0('./', x,'/',score_type,'/gene_rank_max_abs_beta.csv'))[,2:3]
  colnames(r)[2] <- x
  r
}))

panel.cor <- function(x, y, digits = 2, prefix = "", cex.cor, ...)
{
  par(usr = c(0, 1, 0, 1))
  r <- (cor(x, y,method = 's'))
  txt <- format(c(r, 0.123456789), digits = digits)[1]
  txt <- paste0(prefix, txt)
  if(missing(cex.cor)) cex.cor <- 0.8/strwidth(txt)
  text(0.5, 0.5, txt)
}
pairs(comb[,-1], lower.panel = panel.cor, upper.panel = panel.smooth,
      gap=1, row1attop=T)

