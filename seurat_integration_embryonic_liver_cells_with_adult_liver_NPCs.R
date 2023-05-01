library(Seurat)
library(dplyr)
library(ggplot2)
options(future.globals.maxSize = 1000 * 1024^2 * 30)
.libPaths()
library(Matrix)
library(cowplot)
library(patchwork)

indir = "human_fetal_liver/"
outdir = "human_fetal_liver/results"
merged <- readRDS(file=paste(indir, "merged_counts_liver.Rda", sep="/"))
cell_annot = read.table(paste(indir,'human_fetal_liver_cell_type_paper.csv',sep="/"), header=TRUE, sep="\t", row.names=1, stringsAsFactors=FALSE, check.names=FALSE)
merged = merged[rownames(merged),colnames(merged) %in% rownames(cell_annot)]
cell_annot = cell_annot[colnames(merged),]

hsa_counts = read.table(paste(indir,'human_NPCs_counts.csv',sep=""), header=TRUE, sep="\t", row.names=1, stringsAsFactors=FALSE, check.names = FALSE)
hsa_annot = read.table(paste(indir,'human_NPCs_cell_type_annotation_final.csv',sep=""), header=TRUE, sep="\t", row.names=NULL, stringsAsFactors=FALSE, check.names = FALSE)


ob.list <- list()
curr_obj <- CreateSeuratObject(counts = hsa_counts, min.cells = 5, min.features = 200, meta.data = hsa_annot)
curr_obj <- NormalizeData(curr_obj, verbose = FALSE)
curr_obj <- FindVariableFeatures(curr_obj, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
ob.list[[1]] = curr_obj

curr_obj <- CreateSeuratObject(counts = merged, min.cells = 5, min.features = 200,  meta.data = cell_annot)
curr_obj <- NormalizeData(curr_obj, verbose = FALSE)
curr_obj <- FindVariableFeatures(curr_obj, selection.method = "vst", nfeatures = 2000, verbose = FALSE)
ob.list[[2]] = curr_obj

ndim = 20

reference.list <- ob.list
data.anchors <- FindIntegrationAnchors(object.list = reference.list, dims = 1:ndim, 
                                       anchor.features=2000, verbose=FALSE)
                                       
integrated <- IntegrateData(anchorset = data.anchors, dims = 1:ndim, verbose=FALSE)

DefaultAssay(integrated) <- "integrated"

integrated <- ScaleData(integrated, verbose = FALSE)
integrated <- RunPCA(integrated, npcs = ndim, verbose = FALSE)
integrated <- RunUMAP(integrated, reduction = "pca", dims = 1:ndim, 
                      min.dist=0.5,verbose = FALSE)
                      
curr_pca = Embeddings(integrated, reduction = "pca")
curr_umap = Embeddings(integrated, reduction = "umap")

write.table(curr_umap, paste(outdir,"seurat_integrated_umap_paper_annot_fetal_liver_and_yolksac.csv",sep="/"), sep="\t", quote=FALSE)