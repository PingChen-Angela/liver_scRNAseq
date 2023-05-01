import sys
import warnings
warnings.filterwarnings("ignore")
from file_process import *
from sample_cluster import *
from sample_filter import *
from multi_dim_reduction import *
import pandas as pd
import numpy as np
import os
import matplotlib.pyplot as plt
from matplotlib import colors as mcolors
import re
from trajectory_inference import *
import loompy
import scvelo as scv
scv.settings.set_figure_params('scvelo')
import scanpy as sc
import scipy
from matplotlib import rcParams
rcParams['pdf.fonttype'] = 42

rcParams['image.cmap'] = 'tab20'

loomFile = 'Smartseq2_NPCs/velocyto/emm_human_npcs.loom'
sample_annotFile = 'Smartseq2_NPCs/human_NPCs_cell_type_annotation_final.csv'
metadata = pd.read_table('human_NPC_cell_types/metadata.csv', header=0, index_col=0, sep="\t")
rpkms = pd.read_table('human_NPC_cell_types/rpkms.csv', header=0, index_col=0, sep="\t")

outputDIR = 'Smartseq2_NPCs/velocyto/results'

sample_cluster_df = pd.read_table(sample_annotFile, header=0, index_col=None, sep="\t")
sample_cluster_df.index = sample_cluster_df['sampleName'].to_list()

curr_cells = sample_cluster_df.loc[sample_cluster_df['sampleCluster'].isin(['Proliferating Cell 1','LM2-C1','LM2-C2','LM1'])]

if not os.path.exists('%s/DEGs' %(outputDIR)): os.makedirs('%s/DEGs' %(outputDIR))
curr_expr = rpkms[curr_cells.index]
curr_sample_annot = sample_cluster_df.loc[curr_cells.index]
curr_sample_annot['sampleCluster'] = curr_sample_annot['sampleCluster'].replace(to_replace=r' ',value='_',regex=True)
curr_expr.to_csv('%s/DEGs/expr.csv' %(outputDIR), sep="\t", index=True)
curr_sample_annot[['sampleCluster','sampleName']].to_csv('%s/DEGs/sample_groups.csv' %(outputDIR), sep="\t", index=False)

grpList = 'Proliferating_Cell_1,LM2-C1,LM2-C2,LM1'

diff_expr('%s/DEGs/expr.csv' %(outputDIR), '%s/DEGs' %(outputDIR), '%s/DEGs/sample_groups.csv' %(outputDIR), grpList, 'kw', 0.01, 5000, True, 4, 'v2', 1, 'conover', False, 'mean', False)

degs = pd.read_table('%s/DEGs/rankedGeneStats.csv' %(outputDIR), index_col=0, header=0)

from scanpy.pp import filter_genes_dispersion
varGeneRes = filter_genes_dispersion(data=curr_expr.T, n_bins=20, n_top_genes=800)
flag = np.array([item[0] for item in varGeneRes])
varGenes = list(curr_expr.index[flag])
merged_varGenes = list(set(list(degs.index) + varGenes))

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy as sc
sc.set_figure_params(color_map='viridis')
plt.rcParams['pdf.fonttype'] = 42

expr_data = curr_expr.copy()
sample_cluster=curr_sample_annot['sampleCluster']

X = expr_data.values
gene_names = list(expr_data.index)
sample_names = list(expr_data.columns)
    
adata = sc.AnnData(X.transpose())
adata.var_names = gene_names
adata.row_names = sample_names

sc.settings.figdir = outputDIR
adata = adata[:, merged_varGenes] 
adata.var_names = [gene.split('|')[0] for gene in adata.var_names]
adata.obs['Cell_types'] = [curr_sample_annot.loc[sample]['sampleCluster'].replace('_',' ') for sample in sample_names]
adata.obs['Obese_status'] = metadata.loc[sample_names]['Obese_status_group'].to_list()
adata.uns['Cell_types_colors'] = ['#2c7bb6','violet','blue','lime']

root_cell = list(adata.obs['Cell_types'][adata.obs['Cell_types'] == 'Proliferating Cell 1'].index)[0]
adata.uns['iroot'] = int(root_cell)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=20, n_pcs=10)
adata.obs.loc[adata.obs['Cell_types']=='Proliferating Cell 1','subclusters'] = 'Proliferating Cell 1'
adata.uns['subclusters_colors'] = ['green','orange','blue','red','lime']
adata.uns['Obese_status_colors'] = ['lightblue','salmon']

rcParams['figure.figsize'] = [6,6]

sc.set_figure_params(scanpy=True, dpi=300, dpi_save=300, frameon=True, vector_friendly=False, 
                     fontsize=10, figsize=None, color_map=None, 
                     format='pdf', facecolor=None, transparent=False, ipython_format='png2x')

sc.tl.umap(adata, n_components=2, random_state=2, min_dist=0.5)
sc.pl.umap(adata, color=['Cell_types','Obese_status'],#,'subclusters'],
           legend_loc='on data',projection='2d')

curr_umap_loc = pd.DataFrame(adata.obsm['X_umap'])
curr_umap_loc.index = sample_names
curr_umap_loc.to_csv('%s/proliferating_LM1_LM2_umap_loc.csv' %(outputDIR), index=False, sep="\t")

# velocity
cells_to_use = list(sample_cluster_df.loc[sample_cluster_df['sampleCluster'].isin(['Proliferating Cell 1','LM2-C1','LM2-C2','LM1'])].index)
adata = scv.read(loomFile, cache=False)
cell_ids = adata.obs_names
cell_ids = list(map(lambda x: re.sub('emm_human_npcs:','',x), cell_ids))
cell_ids = list(map(lambda x: re.sub('_unique.bam','',x), cell_ids))
adata.obs_names = np.array(cell_ids)
adata_filtered = adata[cells_to_use,:]
adata_filtered.obsm['X_umap'] = curr_umap_loc.loc[cells_to_use].values
scv.pp.filter_and_normalize(adata_filtered, min_shared_counts=20, n_top_genes=3000)
adata_filtered.obs['clusters'] = sample_cluster_df.loc[adata_filtered.obs_names]['sampleCluster'].values
adata_filtered.obs['disease_states'] =  donor_info.loc[adata_filtered.obs_names]['Obese_status_group'].values

scv.pp.moments(adata_filtered, n_pcs=20, n_neighbors=15)
scv.tl.velocity(adata_filtered, mode='stochastic')
scv.tl.velocity_graph(adata_filtered)

scv.settings.set_figure_params('scvelo', dpi_save=300, dpi=80, transparent=True)

plt.rcParams['pdf.fonttype'] = 42
ax = scv.pl.velocity_embedding_stream(adata_filtered, 
                                      palette=['lightblue','#2c7bb6','violet','lime'],
                                      n_neighbors=15, min_mass=0, smooth=0.95,
                                      legend_fontsize=10, figsize=(6,6), 
                                      alpha=0.8, show=False, title='', size=800,
                                      legend_loc='right margin', save='%s/velocity_proliferate_cells_to_LMs.pdf'%(outputDIR))

ax = scv.pl.velocity_embedding_stream(adata_filtered, color='disease_states',
                                      palette=['lightblue','salmon'],
                                      n_neighbors=15, min_mass=0, smooth=0.95,
                                      legend_fontsize=10, figsize=(6,6), 
                                      alpha=0.8, show=False, title='', size=800,
                                      legend_loc='right margin')
plt.savefig('%s/velocity_proliferate_cells_to_LMs_disease_states.pdf' %(outputDIR), dpi=300, bbox_inches='tight')

# pseudotime
if not os.path.exists('%s/DEGs_prof1_vs_LM2C1' %(outputDIR)): os.makedirs('%s/DEGs_prof1_vs_LM2C1' %(outputDIR))
curr_cells = sample_cluster_df.loc[sample_cluster_df['sampleCluster'].isin(['Proliferating Cell 1','LM2-C1'])]
curr_expr = rpkms[curr_cells.index]
curr_sample_annot = sample_cluster_df.loc[curr_cells.index]
curr_sample_annot['sampleCluster'] = curr_sample_annot['sampleCluster'].replace(to_replace=r' ',value='_',regex=True)
curr_expr.to_csv('%s/DEGs_prof1_vs_LM2C1/expr.csv' %(outputDIR), sep="\t", index=True)
curr_sample_annot[['sampleCluster','sampleName']].to_csv('%s/DEGs_prof1_vs_LM2C1/sample_groups.csv' %(outputDIR), sep="\t", index=False)
grpList = 'Proliferating_Cell_1,LM2-C1'

diff_expr('%s/DEGs_prof1_vs_LM2C1/expr.csv' %(outputDIR), '%s/DEGs_prof1_vs_LM2C1' %(outputDIR), '%s/DEGs_prof1_vs_LM2C1/sample_groups.csv' %(outputDIR), grpList, 'kw', 0.01, 5000, True, 4, 'v2', 1, 'conover', False, 'mean', False)

degs = pd.read_table('%s/DEGs_prof1_vs_LM2C1/rankedGeneStats.csv' %(outputDIR), index_col=0, header=0)

varGeneRes = filter_genes_dispersion(data=curr_expr.T, n_bins=20, n_top_genes=500)
flag = np.array([item[0] for item in varGeneRes])
varGenes = list(curr_expr.index[flag])
merged_varGenes = list(set(list(degs.index) + varGenes))

import numpy as np
import pandas as pd
import matplotlib.pyplot as pl
from matplotlib import rcParams
import scanpy as sc
sc.set_figure_params(color_map='viridis')
plt.rcParams['pdf.fonttype'] = 42

expr_data = curr_expr.copy()
sample_cluster=curr_sample_annot['sampleCluster']

X = expr_data.values
gene_names = list(expr_data.index)
sample_names = list(expr_data.columns)
    
adata = sc.AnnData(X.transpose())
adata.var_names = gene_names
adata.row_names = sample_names

adata = adata[:, merged_varGenes] 
adata.var_names = [gene.split('|')[0] for gene in adata.var_names]
adata.obs['Cell_types'] = [curr_sample_annot.loc[sample]['sampleCluster'].replace('_',' ') for sample in sample_names]
adata.obs['Obese_status'] = donor_info.loc[sample_names]['Obese_status_group'].to_list()

adata.uns['Cell_types_colors'] = ['#2c7bb6','lime']
root_cell = list(adata.obs['Cell_types'][adata.obs['Cell_types'] == 'Proliferating Cell 1'].index)[0]
adata.uns['iroot'] = int(root_cell)
sc.pp.log1p(adata)
sc.tl.pca(adata, svd_solver='arpack')
sc.pp.neighbors(adata, n_neighbors=15, n_pcs=10)
adata.obs.loc[adata.obs['Cell_types']=='Proliferating Cell 1','subclusters'] = 'Proliferating Cell 1'
adata.uns['Obese_status_colors'] = ['lightblue','salmon']

rcParams['figure.figsize'] = [6,6]
sc.set_figure_params(scanpy=True, dpi=300, dpi_save=300, frameon=True, vector_friendly=False, 
                     fontsize=10, figsize=None, color_map=None, 
                     format='pdf', facecolor=None, transparent=False, ipython_format='png2x')

sc.tl.umap(adata, n_components=2, random_state=1)
sc.tl.diffmap(adata)
sc.pp.neighbors(adata, n_neighbors=15, use_rep='X_diffmap')
sc.tl.dpt(adata)
sc.pl.umap(adata, color=['Cell_types','Obese_status','dpt_pseudotime'],projection='2d', legend_loc='on data')

plt.figure(figsize=(3,5))
sns.set_style('ticks')
sns.barplot(x='Obese_status',y='dpt_pseudotime', 
            data=adata.obs.query('Cell_types=="LM2-C1"'), ci=95, palette=['lightblue','salmon'], capsize=0.1)
sns.despine()
plt.xlabel('')
plt.ylabel('LM2-C1 pseudotime')
plt.savefig('%s/LM2C1_lean_obese_pseudotime.pdf' %(outputDIR), dpi=300, bbox_inches='tight')

scipy.stats.mannwhitneyu(adata.obs.query('Cell_types=="LM2-C1"').query('Obese_status=="Lean"')['dpt_pseudotime'].values, 
                         adata.obs.query('Cell_types=="LM2-C1"').query('Obese_status=="Obese"')['dpt_pseudotime'].values)

