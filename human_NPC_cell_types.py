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
outdir = 'human_NPC_cell_types/results'

## Input data
rpkm_df = pd.read_table('%s/rpkms.csv' %(indir), header=0, sep="\t", index_col=0)
count_df = pd.read_table('%s/counts.csv' %(indir), header=0, sep="\t", index_col=0)
metadata = pd.read_table('%s/metadata.csv' %(indir), header=0, sep="\t", index_col=0)

## Cell cluster identfication
sigVarGenes = variable_genes(count_df, None, None, 0.01, outdir, nTopGenes=1000)

comp = principle_component_analysis(rpkm_df.T, sigVarGenes, n_comp=30, 
                                    annot=None, annoGE=None, 
                                    pcPlot=False,markerPlot=False)

n_comp = choose_dims_N(comp)

sample_clusters_ap = spectral_clustering(comp['PCmatrix'],10)
cell_cluster_annot = pd.DataFrame.from_dict(sample_clusters_ap)

umap_comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, 
                annot=cell_cluster_annot['sampleCluster'], 
                annoGE=None, 
                init='pca', init_n_comp=n_comp,
                n_neighbors=15, metric='euclidean', min_dist=0.6, 
                initial_embed='spectral',
                log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', 
                pcX=1, pcY=2, prefix='',
                facecolor='white', markerPlotNcol=5, fontsize=10,
                random_state=0, size=100, with_mean=True,
                with_std=False, figsize=(5,5), 
                outdir=outdir, legend_loc='right',
                filename1='umap_color_by_subclusters')

umap_comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, 
                annot=metadata.loc[rpkm_df.columns]['Donor'], 
                annoGE=None, init='pca', init_n_comp=n_comp,
                n_neighbors=15, metric='euclidean', min_dist=0.6, 
                initial_embed='spectral',
                log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', 
                pcX=1, pcY=2, prefix='',
                facecolor='white', markerPlotNcol=5, fontsize=10, 
                random_state=0, size=120, with_mean=True,
                with_std=False, figsize=(5,5),
                outdir=outdir, legend_loc='right',
                filename1='umap_color_by_donors')

new_cell_type_annot = cell_cluster_annot.copy()
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster10','Proliferating', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster1','B cell', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster2','Tcells', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster3','Myeloid 1', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster4','Resident NK', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster5','Mast cell', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster6','cDC1', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster7','Myeloid 2', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster8','Circulating NK', regex=True)
new_cell_type_annot['sampleCluster'] = new_cell_type_annot['sampleCluster'].replace('cluster9','Endothelial', regex=True)

umap_comp = umap_emb(rpkm_df.T, sigVarGenes, n_comp=2, 
                annot=new_cell_type_annot['sampleCluster'], 
                annoGE=None, init='pca', init_n_comp=n_comp,
                n_neighbors=15, metric='euclidean', min_dist=0.6, 
                initial_embed='spectral',
                log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', 
                pcX=1, pcY=2, prefix='',
                facecolor='white', markerPlotNcol=5, fontsize=10,
                random_state=0, size=100, with_mean=True,
                with_std=False, figsize=(5,5), 
                outdir=outdir, legend_loc='bottom',
                filename1='umap_color_by_cell_types')

# DEGs among cell types
if not os.path.exists('%s/DEGs_human_cell_types' %(outdir)): os.makedirs('%s/DEGs_human_cell_types' %(outdir))
new_cell_type_annot.to_csv('%s/DEGs_human_cell_types/sample_cluster.csv' %(outdir), sep="\t", index=False)

grpList = list(set(new_cell_type_annot['sampleCluster'].values))
grpList = ','.join(grpList)
    
diff_expr('%s/rpkms.csv' %(indir), '%s/DEGs_human_cell_types' %(outdir), '%s/DEGs_human_cell_types/sample_cluster.csv' %(outdir), grpList, 'kw', 0.01, 5000, True, 3, 'v2', 1, 'conover', False, 'mean', False)

## Gene counts
new_cell_type_annot.index = new_cell_type_annot['sampleName'].to_list()
flag = rpkm_df > 1
ngenes = flag[flag].sum(axis=0)
curr_ctypes = pd.concat([new_cell_type_annot.loc[ngenes.index][['sampleCluster','sampleName']], pd.DataFrame(ngenes, columns=['N_genes'])], axis=1)
curr_ctypes.columns = ['Cell types','Cell Name','N detected genes']
curr_ctypes['N detected genes'] = curr_ctypes['N detected genes'].astype('int')
curr_ctype_list = curr_ctypes['Cell types'].drop_duplicates().sort_values().to_list()
curr_col_list = [color_dict[ctype] for ctype in curr_ctype_list]

plt.figure(figsize=(10,3))
plt.rcParams['pdf.fonttype'] = 42
plt.tick_params(axis="both",bottom=True,top=False,left=True,right=False)

ax = sns.violinplot(x='Cell types',y='N detected genes',data=curr_ctypes, scale='width', 
                    cut=0, order=curr_ctype_list, palette=curr_col_list)

ax.set_xticklabels(ax.get_xticklabels(), rotation=30)
sns.despine()
plt.savefig('%s/n_genes_in_cell_types.pdf' %(outdir), dpi=300, bbox_inches='tight')





