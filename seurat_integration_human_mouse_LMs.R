library(Seurat)
library(dplyr)
library(ggplot2)
options(future.globals.maxSize = 4000 * 1024^2)
.libPaths()
library(cowplot)
library(patchwork)

indir = 'Smartseq2_NPCs/LMs'
outdir = 'Smartseq2_NPCs/LMs/results'

hsa_counts = read.table(paste(indir,'human_LM_counts.csv',sep=""), header=TRUE, sep="\t", 
                        row.names=1, stringsAsFactors=FALSE, check.names = FALSE)

mmu_counts = read.table(paste(indir,'mouse_LM_counts.csv',sep=""), header=TRUE, sep="\t", 
                        row.names=1, stringsAsFactors=FALSE, check.names = FALSE)

hsa_annot = read.table(paste(indir,'../human_NPCs_cell_type_annotation_final.csv',sep=""), header=TRUE, 
                       sep="\t", row.names=NULL, stringsAsFactors=FALSE, check.names = FALSE)

mmu_annot = read.table(paste(indir,'../mouse_NPCs_cell_type_annotation_final.csv',sep=""), header=TRUE, 
                       sep="\t", row.names=NULL, stringsAsFactors=FALSE, check.names = FALSE)

rownames(hsa_annot) = hsa_annot[,'sampleName']
rownames(mmu_annot) = mmu_annot[,'sampleName']

hsa_annot = hsa_annot[,c("sampleName","sampleCluster")]
colnames(hsa_annot) = c("Cell","CellType")
mmu_annot = mmu_annot[,c("sampleName","sampleCluster")]
colnames(mmu_annot) = c("Cell","CellType")

hsa_annot$Dataset = "Human"
mmu_annot$Dataset = "Mouse"

hsa_annot = hsa_annot[hsa_annot$CellType %in% c('LM1','LM2-C1','LM3','LM4'),]
mmu_annot = mmu_annot[mmu_annot$CellType %in% c('KC1','KC2','MoMac1','MoMac2'),]

hsa_counts = hsa_counts[,rownames(hsa_annot)]
mmu_counts = mmu_counts[,rownames(mmu_annot)]

# create seurat object
ob.list <- list()

curr_obj <- CreateSeuratObject(counts = hsa_counts, min.cells = 5, min.features = 100, meta.data = hsa_annot)
curr_obj <- NormalizeData(curr_obj, verbose = FALSE)
curr_obj <- FindVariableFeatures(curr_obj, selection.method = "dispersion", nfeatures = 1500, verbose = FALSE)
ob.list[[1]] = curr_obj

curr_obj <- CreateSeuratObject(counts = mmu_counts, min.cells = 5, min.features = 100,  meta.data = mmu_annot)
curr_obj <- NormalizeData(curr_obj, verbose = FALSE)
curr_obj <- FindVariableFeatures(curr_obj, selection.method = "dispersion", nfeatures = 1500, verbose = FALSE)
ob.list[[2]] = curr_obj

ndim = 20

# integrate dataset
reference.list <- ob.list
data.anchors <- FindIntegrationAnchors(object.list = reference.list, dims = 1:ndim, 
                                       anchor.features=1500, verbose=FALSE)

integrated <- IntegrateData(anchorset = data.anchors, dims = 1:ndim, verbose=FALSE)

DefaultAssay(integrated) <- "integrated"
integrated <- ScaleData(integrated, verbose = FALSE)
integrated <- RunPCA(integrated, npcs = ndim, verbose = FALSE)
integrated <- RunUMAP(integrated, reduction = "pca", 
                      dims = 1:ndim, spread=1, seed.use=30, metric = "correlation",#n.neighbors=20,
                      umap.method='umap-learn', 
                      min.dist=0.5,verbose = FALSE)

curr_pca = Embeddings(integrated, reduction = "pca")
curr_umap = Embeddings(integrated, reduction = "umap")
write.table(curr_umap, paste(outdir,"seurat_integrated_umap.csv",sep="/"), sep="\t", quote=FALSE)