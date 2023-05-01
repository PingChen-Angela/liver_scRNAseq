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

indir = 'human_NPC_cell_types'
outdir = 'human_NPC_cell_types/results/LMs'

rpkm_filter = 1
count_filter = 2
ncell = 3
fdr = 0.05

rpkm_df = pd.read_table('%s/rpkms.csv' %(indir), header=0, sep="\t", index_col=0)
count_df = pd.read_table('%s/counts.csv' %(indir), header=0, sep="\t", index_col=0)
metadata = pd.read_table('%s/metadata.csv' %(indir), header=0, sep="\t", index_col=0)
cell_type_annotation = pd.read_table('%s/results/DEGs_human_cell_types/sample_cluster.csv' %(indir), header=0, index_col=None, sep="\t")
cell_type_annotation.index = cell_type_annotation['sampleName'].to_list()

cells_in_use = list(cell_type_annotation.loc[cell_type_annotation['sampleCluster'].isin(['Myeloid 1','Myeloid 2'])].index)
rpkms = rpkm_df[cells_in_use]
counts = count_df[cells_in_use]

flag = rpkms > rpkm_filter
rpkm_df = rpkms[flag[flag].sum(axis=1) > ncell]

flag = counts > count_filter
count_df = counts[flag[flag].sum(axis=1) > ncell]

sigVarGenes = variable_genes(count_df, None, None, fdr, outdir, nTopGenes=500)
comp = principle_component_analysis(rpkm_df.T, sigVarGenes, n_comp=30, 
                                    annot=None, annoGE=None, 
                                    pcPlot=False,markerPlot=False)

n_comp = choose_dims_N(comp)

def kmeans(df, n_cluster):
    X = df.values
    km = KMeans(n_clusters=n_cluster, random_state=1, n_init=500, max_iter=1000, algorithm='full').fit(X)
    cluster_labels = np.array(['cluster%s'%label for label in km.labels_+1])
    sample_names = np.array(df.index)
    return {'sampleName': sample_names, 'sampleCluster': cluster_labels}

sample_clusters_ap = kmeans(comp['PCmatrix'],4)
cell_cluster_annot = pd.DataFrame.from_dict(sample_clusters_ap)
cell_cluster_annot.index = cell_cluster_annot['sampleName'].to_list()

new_cell_type_annot = cell_cluster_annot.copy()
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster1','LM3', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster2','LM4', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster3','LM2', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster4','LM1', regex=True)

comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, 
                annot=metadata.loc[rpkm_df.columns]['Donor'], 
                annoGE=None, init='pca', init_n_comp=n_comp,
                n_neighbors=15, metric='euclidean', min_dist=0.5, 
                initial_embed='spectral',
                log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', 
                pcX=1, pcY=2, prefix='',
                facecolor='lightgrey', markerPlotNcol=5, fontsize=10, 
                random_state=0, size=120, with_mean=True,
                with_std=False, figsize=(6,6),
                outdir=outdir,legend_loc='right',
                filename1='umap_color_by_donor')

comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, 
                annot=new_cell_type_annot.loc[rpkm_df.columns]['sampleCluster'], 
                annoGE=None, init='pca', init_n_comp=n_comp,
                n_neighbors=15, metric='euclidean', min_dist=0.5, 
                initial_embed='spectral',
                log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', 
                pcX=1, pcY=2, prefix='',
                facecolor='lightgrey', markerPlotNcol=5, fontsize=10, 
                random_state=0, size=120, with_mean=True,
                with_std=False, figsize=(6,6),
                outdir=outdir,legend_loc='right',
                filename1='umap_color_by_subclusters')

curr_deg_dir = '%s/DEGs_LM_subclusters' %outdir
if not os.path.exists(curr_deg_dir): os.makedirs(curr_deg_dir)

rpkm_df[new_cell_type_annot.index].to_csv('%s/expr.csv' %(curr_deg_dir), sep="\t", index=True)
new_cell_type_annot.to_csv('%s/sample_cluster.csv' %(curr_deg_dir), sep="\t", index=False)
grpList = list(set(new_cell_type_annot['sampleCluster'].values))
grpList = ','.join(grpList)
    
diff_expr('%s/expr.csv' %(curr_deg_dir), curr_deg_dir, '%s/sample_cluster.csv' %(curr_deg_dir), 
          grpList, 'kw', 0.01, 5000, True, 3, 'v2', 1, 'conover', False, 'mean', False)

# LM2 subclusters
macro_marker_names = ['CD68','CD14','CSF1R','CD163','VSIG4']
macro_gids = [gene for gene in rpkm_df.index if gene.split('|')[0] in macro_marker_names]
curr_cells = list(new_cell_type_annot.loc[new_cell_type_annot['sampleCluster']=="LM2"].index)
curr_annot = new_cell_type_annot.loc[curr_cells]
curr_expr = np.log2(rpkm_df[curr_cells]+1)

select_gexpr = curr_expr.loc[macro_gids]
select_gexpr.index = [gene.split('|')[0] for gene in select_gexpr.index]

sample_clusters_ap = spectral_clustering(select_gexpr.T,2)
LM2_subclusters = pd.DataFrame.from_dict(sample_clusters_ap)

curr_annot.loc[LM2_subclusters.query('sampleCluster=="cluster1"').index,'sampleCluster'] = 'LM2-C1'
curr_annot.loc[LM2_subclusters.query('sampleCluster=="cluster2"').index,'sampleCluster'] = 'LM2-C2'

# summary
final_LM_subcluster_annotation = new_cell_type_annot.copy()
final_LM_subcluster_annotation.loc[curr_annot.index, 'sampleCluster'] = curr_annot['sampleCluster'].to_list()
