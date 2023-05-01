library(Seurat)
library(dplyr)
library(Matrix)

# dataset: PMID35021063
data_dir <- 'Scott_Guilliams_paper/myeloid_cells/rawData_human/countTable_human'
outdir <- 'Scott_Guilliams_paper/data'

expression_matrix <- Read10X(data.dir = data_dir, gene.column=1)
all_liver_atlas_annot = read.table('Scott_Guilliams_paper/myeloid_cells/annot_humanMyeloid.csv', 
                                   header=TRUE, sep=",", row.names=NULL, stringsAsFactors=FALSE, check.names = FALSE)
rownames(all_liver_atlas_annot) = all_liver_atlas_annot$cell
annot = all_liver_atlas_annot[,c('cell','annot','sample','typeSample')]
colnames(annot) = c('Cell','CellType','Sample','Dataset')
annot = annot[annot$Dataset != "citeSeq",]

annot$CellType = paste('Atlas: ',annot$CellType,sep="")
annot$Dataset = paste('Human liver atlas: ',annot$Dataset,sep="")

curr_expr = expression_matrix[,rownames(annot)]

write.table(annot[annot$CellType == "Atlas: cDC2s",], paste(outdir,"cDCs_annotation.csv",sep="/"), sep="\t", quote=FALSE)

df = curr_expr[,annot[annot$CellType == "Atlas: cDC2s",'Cell']]
write.table(df, paste(outdir,"cDCs_expr_counts.csv",sep="/"), sep="\t", quote=FALSE)