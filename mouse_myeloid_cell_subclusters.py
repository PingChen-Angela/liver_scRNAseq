import sys
import warnings
warnings.filterwarnings("ignore")
from multi_dim_reduction import *
from file_process import *
from sample_cluster import *
from sample_filter import *
from vis_tools import *
import pandas as pd
import numpy as np
import os
import re

indir = 'mouse_NPC_cell_types'
outdir = 'mouse_NPC_cell_types/results/LMs'
rpkm_df = pd.read_table('%s/rpkms.csv' %(indir), header=0, sep="\t", index_col=0)
count_df = pd.read_table('%s/counts.csv' %(indir), header=0, sep="\t", index_col=0)
cell_type_annot = pd.read_table('%s/results/DEGs_mouse_cell_types/sample_cluster.csv' %(indir), 
                                sep="\t", header=0, index_col=None)

cell_type_annot.index = cell_type_annot['sampleName'].to_list()
flag = cell_type_annot['sampleCluster'].isin(['MoMacs and cDCs','Kupffer cells'])
cells_in_use = list(cell_type_annot.loc[flag].index)
rpkm_df = rpkm_df[cells_in_use]
count_df = count_df[cells_in_use]

sigVarGenes = variable_genes(count_df, None, None, outputDIR=outdir, nTopGenes=1000)

comp = principle_component_analysis(rpkm_df.T, sigVarGenes, n_comp=30, 
                                    annot=None, annoGE=None, 
                                    pcPlot=False,markerPlot=False)

n_comp = choose_dims_N(comp)

comp = principle_component_analysis(rpkm_df.T, sigVarGenes, n_comp=n_comp, 
                                    annot=None, annoGE=None, log=True,
                                    pcPlot=False,pcPlotType='normal',markerPlot=False)

sample_clusters_ap = spectral_clustering(comp['PCmatrix'],2)
cell_cluster_annot = pd.DataFrame.from_dict(sample_clusters_ap)
cell_cluster_annot.index = cell_cluster_annot['sampleName'].to_list()

umap_comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, 
                annot=cell_cluster_annot['sampleCluster'], 
                annoGE=None, init='pca', init_n_comp=n_comp,
                n_neighbors=15, metric='euclidean', min_dist=0.5, 
                initial_embed='spectral',
                log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', 
                pcX=1, pcY=2, prefix='',
                facecolor='white', markerPlotNcol=5, fontsize=10, 
                random_state=0, size=80, with_mean=True,
                with_std=False, figsize=(5,5), 
                outdir=outdir, legend_loc='right',
                filename1='umap_color_by_subclusters')

cell_cluster_annot.to_csv('%s/LM_subclusters_init.csv' %(outdir), sep="\t", index=True)

# identify subclusters in cluster2
cells_in_use = list(cell_cluster_annot.query('sampleCluster=="cluster2"').index)
c2_rpkm_df = rpkm_df[cells_in_use]
c2_count_df = count_df[cells_in_use]

sigVarGenes2 = variable_genes(c2_count_df, None, None, outputDIR='%s/cluster2_subclusters' %outdir, nTopGenes=1000)
comp = principle_component_analysis(c2_rpkm_df.T, sigVarGenes2, n_comp=30, annot=None, annoGE=None, pcPlot=False,markerPlot=False)
n_comp2 = choose_dims_N(comp)

comp = principle_component_analysis(c2_rpkm_df.T, sigVarGenes2, n_comp=n_comp2, 
                                    annot=None, annoGE=None, log=True,
                                    pcPlot=False,pcPlotType='normal',markerPlot=False)

sample_clusters_ap = spectral_clustering(comp['PCmatrix'],4)
cell_cluster_annot2 = pd.DataFrame.from_dict(sample_clusters_ap)
cell_cluster_annot2.index = cell_cluster_annot2['sampleName'].to_list()

umap_comp = umap_emb(c2_rpkm_df.T, sigVarGenes2, n_comp=2, 
                annot=cell_cluster_annot2['sampleCluster'], 
                annoGE=None, init='pca', init_n_comp=n_comp2,
                n_neighbors=15, metric='euclidean', min_dist=0.5, 
                initial_embed='spectral',
                log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', 
                pcX=1, pcY=2, prefix='',
                facecolor='white', markerPlotNcol=5, fontsize=10, 
                random_state=0, size=80, with_mean=True,
                with_std=False, figsize=(5,5), 
                outdir='%s/cluster2_subclusters' %outdir, legend_loc='right',
                filename1='umap_color_by_cluster2_subclusters')

# summary
cell_cluster_annot2['sampleCluster'] = cell_cluster_annot2['sampleCluster'].str.replace('cluster','cluster2-C',regex=True).to_list()
curr_annot = cell_cluster_annot.copy()
curr_annot.loc[cell_cluster_annot2.index, 'sampleCluster'] = cell_cluster_annot2['sampleCluster'].to_list()
curr_annot['sampleCluster'] = curr_annot['sampleCluster'].replace('cluster1','KC1',regex=True)
curr_annot['sampleCluster'] = curr_annot['sampleCluster'].replace('cluster2-C1','MoMac2',regex=True)
curr_annot['sampleCluster'] = curr_annot['sampleCluster'].replace('cluster2-C2','KC2',regex=True)
curr_annot['sampleCluster'] = curr_annot['sampleCluster'].replace('cluster2-C3','MoMac1',regex=True)
curr_annot['sampleCluster'] = curr_annot['sampleCluster'].replace('cluster2-C4','cDCs',regex=True)

umap_comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, 
                annot=curr_annot['sampleCluster'], 
                annoGE=None, init='pca', init_n_comp=n_comp,
                n_neighbors=15, metric='euclidean', min_dist=0.5, 
                initial_embed='spectral',
                log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', 
                pcX=1, pcY=2, prefix='',
                facecolor='white', markerPlotNcol=5, fontsize=10, 
                random_state=0, size=80, with_mean=True,
                with_std=False, figsize=(5,5), 
                outdir=outdir, legend_loc='right',
                filename1='umap_color_by_subclusters_final')

# DEGs
if not os.path.exists('%s/DEGs' %(outdir)): os.makedirs('%s/DEGs' %(outdir))
curr_annot.to_csv('%s/DEGs/sample_cluster.csv' %(outdir), sep="\t", index=False)
rpkm_df.to_csv('%s/DEGs/expr.csv' %(outdir), sep="\t", index=True)

grpList = list(set(curr_annot['sampleCluster'].values))
grpList = ','.join(grpList)
    
diff_expr('%s/DEGs/expr.csv' %(outdir), '%s/DEGs' %(outdir), '%s/DEGs/sample_cluster.csv' %(outdir), 
          grpList, 'kw', 0.01, 5000, True, 3, 'v2', 1, 'conover', False, 'mean', False)

