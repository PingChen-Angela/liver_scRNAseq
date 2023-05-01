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
from scipy.io import loadmat

# load data
indir = 'dataset/PMID31292543/processed_datasets'
expr = pd.read_table('%s/GSE124395_liver_epithelial_expression_mtx.csv' %(indir), header=0, index_col=0, sep="\t")
cell_type_annot = pd.read_table('%s/GSE124395_liver_epithelial_cell_type_annot.csv' %(indir), header=0, 
                                index_col=None, sep="\t")
outdir = '%s/results' %(indir)
if not os.path.exists(outdir): os.makedirs(outdir)
    
cell_type_annot.columns = ['Cell','CellType']
cell_type_annot.index = cell_type_annot['Cell'].to_list()

# P304
P304_cryopreserved = cell_type_annot.loc[cell_type_annot['Cell'].str.contains('P304_(17|18)_|ASGR1pos304|LSEC304'), "Cell"].to_list()
P304_fresh = list(set(cell_type_annot.loc[cell_type_annot['Cell'].str.contains('304'),"Cell"].values) - set(P304_cryopreserved))
cryo = cell_type_annot.loc[P304_cryopreserved]
fresh = cell_type_annot.loc[P304_fresh]

cryo['Status'] = 'Cryopreserved'
fresh['Status'] = 'Fresh'
P304_cell_annot = pd.concat([cryo, fresh], axis=0)
P304_cell_annot['Donor'] = 'P304'

# P301
P301_cryopreserved = cell_type_annot.loc[cell_type_annot['Cell'].str.contains('P301_(5|6|7|8|17|18)_'), "Cell"].to_list()
P301_fresh = list(set(cell_type_annot.loc[cell_type_annot['Cell'].str.contains('301'),"Cell"].values) - set(P301_cryopreserved))
cryo = cell_type_annot.loc[P301_cryopreserved]
fresh = cell_type_annot.loc[P301_fresh]

cryo['Status'] = 'Cryopreserved'
fresh['Status'] = 'Fresh'
P301_cell_annot = pd.concat([cryo, fresh], axis=0)
P301_cell_annot['Donor'] = 'P301'

# P325
P325_cryopreserved = cell_type_annot.loc[cell_type_annot['Cell'].str.contains('(ASGR1pos325_1|ASGR1pos325_2|ASGR1posCD45neg_10|ASGR1posCD45neg_9|ASGR1posCD45pos_11|ASGR1posCD45pos_12|CD34pos_10|CD34pos_9|CD45pos325_5|CD45pos325_6|CD45pos325_7|CD45pos325_8|EPCAMhigh325_14|EPCAMint325325_13|EPCAMlow325_15|LSEC325_13|LSEC325_14|LSEC325_15|LSEC325_16|Mixed325_1|Mixed325_2|Mixed325_3|Mixed325_4)'), "Cell"].to_list()

P325_fresh = cell_type_annot.loc[cell_type_annot['Cell'].str.contains('(ASGR1_CD45_UNB_P4_5|ASGR1_CD45_UNB_P4_6|ASGR1_CD45_UNB_P4_7|ASGR1_CD45_UNB_P4_8|CD45NEG_ASGR1_P7_14|CD45_NPC_P5_10|CD45_NPC_P5_11|CD45_NPC_P5_12|CD45_NPC_P5_9|ASGR1pos_4|ASGR1posfresh_4|CD34pos325frsh_10|CD34pos325fresh_11|CD34pos325frsh_12|CD34pos325fresh_9|CD45posfresh_5|CD45posfresh_6|CD45posfresh_7|CD45posfresh_8|EPCAMTrop2high_2|EPCAMTrop2int_1|EPCAMTrop2low_3|EPCAMhighfresh_2|EPCAMintfresh_1|EPCAMlowfresh_3|LSECfresh_10|LSECfresh_11|LSECfresh_12|LSECfresh_9|PHH_UNB_1|PHH_UNB_2|PHH_UNB_3|PHH_UNB_4)'),"Cell"].to_list()

cryo = cell_type_annot.loc[P325_cryopreserved]
fresh = cell_type_annot.loc[P325_fresh]

cryo['Status'] = 'Cryopreserved'
fresh['Status'] = 'Fresh'
P325_cell_annot = pd.concat([cryo, fresh], axis=0)
P325_cell_annot['Donor'] = 'P325'

# subclusters
df = pd.concat([P325_cell_annot.query('CellType=="Kupffer Cells"'),
           P304_cell_annot.query('CellType=="Kupffer Cells"'),
           P301_cell_annot.query('CellType=="Kupffer Cells"')], axis=0)

curr_expr = expr[df.index]
curr_expr.shape

from scanpy.pp import filter_genes_dispersion
varGeneRes = filter_genes_dispersion(data=curr_expr.T, n_bins=20, n_top_genes=1000)
flag = np.array([item[0] for item in varGeneRes])
varGenes = list(curr_expr.index[flag])

comp = principle_component_analysis(curr_expr.T, varGenes, n_comp=30, 
                                    annot=None, annoGE=None, 
                                    pcPlot=False,markerPlot=False)

n_comp = choose_dims_N(comp)

comp = principle_component_analysis(curr_expr.T, varGenes, n_comp=n_comp, 
                                    annot=None, annoGE=None, log=True,
                                    pcPlot=False,pcPlotType='normal',markerPlot=False)

sample_clusters_ap = spectral_clustering(comp['PCmatrix'],5)

cell_cluster_annot = pd.DataFrame.from_dict(sample_clusters_ap)
cell_cluster_annot.index = cell_cluster_annot['sampleName'].to_list()
                     
cell_cluster_annot['sampleCluster'] = cell_cluster_annot['sampleCluster'].replace('cluster1','Myeloid 4',regex=True)
cell_cluster_annot['sampleCluster'] = cell_cluster_annot['sampleCluster'].replace('cluster2','Myeloid 2',regex=True)
cell_cluster_annot['sampleCluster'] = cell_cluster_annot['sampleCluster'].replace('cluster3','Myeloid 3',regex=True)
cell_cluster_annot['sampleCluster'] = cell_cluster_annot['sampleCluster'].replace('cluster4','Myeloid 1',regex=True)
cell_cluster_annot['sampleCluster'] = cell_cluster_annot['sampleCluster'].replace('cluster5','B cells',regex=True)

umap_comp = umap_emb(curr_expr.T, varGenes, n_comp=2, 
                annot=cell_cluster_annot.loc[curr_expr.columns]['sampleCluster'], 
                annoGE=None, init='pca', init_n_comp=n_comp,
                n_neighbors=15, metric='euclidean', min_dist=0.5, 
                initial_embed='spectral',
                log=True, markerPlot=False, pcPlot=True, pcPlotType='normal', 
                pcX=1, pcY=2, prefix='',
                facecolor='white', markerPlotNcol=5, fontsize=10, 
                random_state=0, size=120, with_mean=True,
                with_std=False, figsize=(5,5), 
                outdir=outdir, legend_loc='right',
                filename1='umap_color_by_subclusters'')