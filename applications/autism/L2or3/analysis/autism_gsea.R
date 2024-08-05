## GO Pathway analysis
  
suppressMessages(library(clusterProfiler))
suppressMessages(library(org.Hs.eg.db))
suppressMessages(library(dplyr))
library(msigdbr)
library(fgsea)


ct <- 'L2or3'
cutoff <- 0.15
n_top <- 20


### GO over-representation analysis

### 1. Molecular Function

genelist <- read.csv(paste0('./scoreA/gene_rank_max_abs_beta.csv'))[,2]
gene <- genelist[1:n_top]

ego.mf <- enrichGO(gene          = gene,
                   universe      = genelist,
                   keyType = "SYMBOL",
                   OrgDb         = org.Hs.eg.db,
                   ont           = "MF", 
                   pAdjustMethod = "BH",
                   pvalueCutoff  = cutoff,
                   qvalueCutoff  = cutoff,
                   minGSSize = 1,
                   readable      = TRUE)
head(ego.mf)
dotplot(ego.mf)


### 2. Biological Process

ego.bp <- enrichGO(gene          = gene,
                   universe      = genelist,
                   keyType = "SYMBOL",
                   OrgDb         = org.Hs.eg.db,
                   ont           = "BP",
                   pAdjustMethod = "BH",
                   pvalueCutoff  = cutoff,
                   qvalueCutoff  = cutoff,
                   minGSSize = 1,
                   readable      = TRUE)
head(ego.bp,10)
dotplot(ego.bp, font.size = 14)
ggsave('./scoreA/GO_BP.png',height = 4,width = 6)



# Gene-Concept Network
cnetplot(ego.bp,showCategory = 10, colorEdge = T, layout = 'kk', 
         cex_category = 1, cex_gene = 0.5,
         cex_label_category = 0.8, cex_label_gene = 0.7)
ggsave('./scoreA/GO_BP_network.png',height = 4,width = 7)


cnetplot(ego.bp, circular = TRUE, colorEdge = TRUE)


heatplot(ego.bp)

enrichplot::upsetplot(ego.bp)

enrichplot::treeplot(enrichplot::pairwise_termsim(ego.bp))

emapplot(enrichplot::pairwise_termsim(ego.bp))


### 3. Cellular Component

ego.cc <- enrichGO(gene          = gene,
                   universe      = genelist,
                   keyType = "SYMBOL",
                   OrgDb         = org.Hs.eg.db,
                   ont           = "CC",
                   pAdjustMethod = "BH",
                   pvalueCutoff  = cutoff,
                   qvalueCutoff  = cutoff,
                   minGSSize = 1,
                   readable      = TRUE)
head(ego.cc)


### GO Gene Set Enrichment Analysis

rnk <- read.table(paste0('./scoreA/gene_rank.rnk'))
genelist.rnk <- rnk$V2; names(genelist.rnk) <- rnk$V1

go.gsea.mf <- gseGO(geneList     = genelist.rnk,
                    keyType = "SYMBOL",
                    OrgDb        = org.Hs.eg.db,
                    ont          = "MF",
                    minGSSize    = 1,
                    pvalueCutoff = cutoff,
                    verbose      = FALSE)
head(go.gsea.mf)



go.gsea.bp <- gseGO(geneList     = genelist.rnk,
                    keyType = "SYMBOL",
                    OrgDb        = org.Hs.eg.db,
                    ont          = "BP",
                    minGSSize    = 1,
                    pvalueCutoff = cutoff,
                    verbose      = FALSE)
go.gsea.bp[go.gsea.bp$qvalue<0.05,] %>% data.frame %>% dplyr::select(ID, Description, qvalue)



go.gsea.cc <- gseGO(geneList     = genelist.rnk,
                    keyType = "SYMBOL",
                    OrgDb        = org.Hs.eg.db,
                    ont          = "CC",
                    minGSSize    = 1,
                    pvalueCutoff = cutoff,
                    verbose      = FALSE)
head(go.gsea.cc)
go.gsea.cc[go.gsea.cc$qvalue<0.05,] %>% data.frame %>% dplyr::select(ID, Description, qvalue)


## KEGG Pathway analysis

ids <- bitr(genelist, 
            fromType = "SYMBOL", 
            toType = "ENTREZID", 
            OrgDb = org.Hs.eg.db)

kk <- enrichKEGG(gene         = ids$ENTREZID[ids$SYMBOL %in% gene],
                 universe     = ids$ENTREZID,
                 keyType = 'kegg',
                 organism     = 'hsa',
                 minGSSize = 1,
                 pvalueCutoff = cutoff,
                 qvalueCutoff = cutoff)
head(kk)


## GSEA
pathwaysDF <- msigdbr("Homo sapiens", category="C5", subcategory = "BP") #"Hallmark
pathways <- split(as.character(pathwaysDF$human_gene_symbol), pathwaysDF$gs_name)
names(pathways) <- tolower(gsub('GOBP_','',names(pathways)))

input <- read.csv(paste0('./result/autism_filter/',ct,'/scoreA/gene_rank_max_abs_beta.csv'))
gsea_input <- input$Dim2
names(gsea_input) <- input$gene
head(gsea_input)

set.seed(123)
fgseaRes <- fgsea(pathways = pathways, 
                  stats    = gsea_input,
                  eps = 0.0,
                  minSize  = 15,
                  maxSize  = 500)

sign <- fgseaRes %>% filter(padj < 0.01) %>% arrange(-NES)
head(sign,10)

# plotEnrichment(pathways[[sign$pathway[1]]], gsea_input) + labs(title=sign$pathway[1])

# topPathwaysDown <- fgseaRes[ES < 0][head(order(pval), n=10), pathway]
# topPathways <- c(topPathwaysUp, rev(topPathwaysDown))
png(paste0('./result/autism_filter/',ct,'/scoreA/GSEA/top10_gsea.png'), height = 6, width = 10, units = 'in', res = 300)
plotGseaTable(pathways[sign$pathway[1:10]], gsea_input, fgseaRes, gseaParam=0.5)
dev.off()


