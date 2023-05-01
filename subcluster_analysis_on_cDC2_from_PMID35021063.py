import sys
import warnings
warnings.filterwarnings("ignore")
from file_process import *
from sample_cluster import *
from sample_filter import *
from multi_dim_reduction import *
from vis_tools import *
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import re
import scanpy as sc
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42


indir = 'Scott_Guilliams_paper/data'
outdir = 'Scott_Guilliams_paper/results'

scott_cdc2_expr = pd.read_table('%s/cDCs_expr_counts.csv' %(indir), header=0, index_col=0, sep="\t")
scott_cdc2_annot = pd.read_table('%s/cDCs_annotation.csv' %(indir), header=0, index_col=0, sep="\t")

from scanpy.pp import filter_genes_dispersion
varGeneRes = filter_genes_dispersion(data=scott_cdc2_expr.T, n_bins=20, n_top_genes=500)
flag = np.array([item[0] for item in varGeneRes])
varGenes = list(scott_cdc2_expr.index[flag])

comp = principle_component_analysis(scott_cdc2_expr.T, varGenes, n_comp=10, 
                                    annot=None, annoGE=None, log=True,
                                    pcPlot=False,pcPlotType='normal',markerPlot=False)

sample_clusters_ap = spectral_clustering(comp['PCmatrix'],4)
cell_cluster_annot = pd.DataFrame.from_dict(sample_clusters_ap)
cell_cluster_annot.index = cell_cluster_annot['sampleName'].to_list()

umap_comp = umap_emb(scott_cdc2_expr.T, varGenes, n_comp=2, 
                     annot=cell_cluster_annot['sampleCluster'], 
                     annoGE=None, 
                     init='pca', init_n_comp=10,
                     n_neighbors=15, metric='euclidean', min_dist=0.5, 
                     initial_embed='spectral',
                     log=True, markerPlot=True, pcPlot=True, pcPlotType='normal', 
                     pcX=1, pcY=2, prefix='',
                     facecolor='white', markerPlotNcol=5, fontsize=10,
                     random_state=0, size=100, with_mean=True,
                     with_std=False, figsize=(5,5), #color_palette=color_panel,
                     outdir=outdir, filename1='scott_cDC2_subclusters', legend_loc='right')


expr_data = scott_cdc2_expr.copy()
sample_cluster=cell_cluster_annot.loc[expr_data.columns]['sampleCluster']

X = expr_data.values
gene_names = list(expr_data.index)
sample_names = list(expr_data.columns)
    
adata = sc.AnnData(X.transpose())
adata.var_names = gene_names
adata.row_names = sample_names

adata.obs['sampleCluster'] = [ctype.replace('cluster','cDC2-C') for ctype in sample_cluster.to_list()]
sc.settings.figdir = outdir

matplotlib.rcdefaults()
plt.rcParams['axes.labelsize'] = 10
rcParams['pdf.fonttype'] = 42

cDCs_marker_names = ['FCER1A','CD1C','CD1E','CLEC10A','IL1R2']
macro_marker_names = ['CD68','CD14','CSF1R','CD163','VSIG4']

marker_genes_dict = {'cDC2': cDCs_marker_names,
                     'macrophage': macro_marker_names}

sc.pl.dotplot(adata, marker_genes_dict, groupby='sampleCluster', swap_axes=True,
              log=True, save='scott_data_cDC2_subclusters.pdf',
              expression_cutoff=1, color_map='RdPu', figsize=(3.5,3),
             )

# visualize cDC2 subclusters in original UMAP from the paper
curr_indir = 'Scott_Guilliams_paper/myeloid_cells/'
published_data_annotation = pd.read_table('%s/annot_humanMyeloid.csv' %(curr_indir), header=0, index_col=None, sep=",")
published_data_annotation.index = [cell.replace('-','.') for cell in published_data_annotation['cell'].values]

published_data_annotation = published_data_annotation.query('typeSample=="scRnaSeq"')
new_published_data_annotation = published_data_annotation.copy()
new_published_data_annotation.loc[cell_cluster_annot.query('sampleCluster=="cluster1"').index, 'annot'] = 'cDC2-C1'
new_published_data_annotation.loc[cell_cluster_annot.query('sampleCluster=="cluster2"').index, 'annot'] = 'cDC2-C2'
new_published_data_annotation.loc[cell_cluster_annot.query('sampleCluster=="cluster3"').index, 'annot'] = 'cDC2-C3'
new_published_data_annotation.loc[cell_cluster_annot.query('sampleCluster=="cluster4"').index, 'annot'] = 'cDC2-C4'

reduce_plot(published_data_annotation[['UMAP_1','UMAP_2']].values, annot=new_published_data_annotation['annot'], 
            prefix='UMAP', size=30, fontsize=10, pcX=1, pcY=2, color_palette=None,
            figsize=(8,8), outdir=outdir, filename='myeloids_umap_cDC2_subclusters', legend_loc='best', legend_ncol=2,
            add_sample_label=False, ordered_sample_labels=None, edgecolors='none')
