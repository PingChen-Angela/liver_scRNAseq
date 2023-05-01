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

hsa_annot = pd.read_table('human_LMs_subcluster_annotation.csv', header=0, index_col=0, sep="\t")
mmu_annot = pd.read_table('mouse_LMs_subcluster_annotation.csv', header=0, index_col=0, sep="\t")
hsa_annot.columns = ['Cell','CellType']
mmu_annot.columns = ['Cell','CellType']
outputDIR = 'conservation_results'

# load expression
human_rpkms = pd.read_table('human_LM_rpkms.csv', sep="\t", header=0, index_col=0)
mouse_rpkms = pd.read_table('mouse_LM_rpkms.csv', header=0, index_col=0, sep="\t")

human_rpkms = human_rpkms[hsa_annot.index]
mouse_rpkms = mouse_rpkms[mmu_annot.index]
mouse_rpkms = np.log2(mouse_rpkms+1)
human_rpkms = np.log2(human_rpkms+1)

# homolog genes
homology_annotation = pd.read_table('ensembl_mart/v92/mart_export.txt',header=0, index_col=None, sep=",")
homology_annotation.columns = [cname.replace(' ','_') for cname in homology_annotation.columns]
homology_annotation = homology_annotation.drop_duplicates().query('Mouse_homology_type!="ortholog_many2many"')
mouse_rpkms.index = [gene.replace('.','') if re.search('\\.',gene) else gene for gene in mouse_rpkms.index]

new_homologs = homology_annotation.copy()
new_homologs.index = new_homologs['Mouse_gene_name']
new_homologs.index.name = ''

def correct_homolog_pairs(df, base, target, quantified_genes):
    base_gene = df[base].values[0]
    target_gene_idx = [idx for idx in range(df[target].shape[0]) if df[target].values[idx].upper()==base_gene.upper()]
    if len(target_gene_idx) > 0:
        return df.iloc[target_gene_idx]
    
    target_gname = [gname for gname in quantified_genes if gname.upper()==base_gene.upper()]
    if len(target_gname) > 0:
        new_df = df.copy()
        new_df.iloc[0][target] = target_gname[0]
        return new_df
        
    return df


human_cell_type_genes = pd.read_table('human_DEGs_LM1_LM3_LM4_mean/rankedGeneStats.csv', index_col=0, header=0)
mouse_cell_type_genes = pd.read_table('mouse_DEGs_KC1_MoMac1_MoMac2_mean/rankedGeneStats.csv', index_col=0, header=0)

# human LM1 and mouse KC1
mouse_cells_in_use = list(mmu_annot.query('sampleCluster=="KC1"')['Cell'].values)
human_cells_in_use = list(hsa_annot.query('sampleCluster=="LM1"')['Cell'].values)
curr_human_expr = human_rpkms[human_cells_in_use]
curr_mouse_expr = mouse_rpkms[mouse_cells_in_use]

mouse_mean_expr = curr_mouse_expr.apply(np.mean, axis=1)
human_mean_expr = curr_human_expr.apply(np.mean, axis=1)

mouse_median_expr = curr_mouse_expr.apply(np.median, axis=1)
human_median_expr = curr_human_expr.apply(np.median, axis=1)

mouse_gene_rank = list(mouse_median_expr.sort_values(ascending=False).index)
mouse_gene_rank = pd.DataFrame([mouse_gene_rank, [i+1 for i in range(len(mouse_gene_rank))]], index=['MouseGene','MouseRank']).T
mouse_gene_rank.index = [gene for gene in mouse_gene_rank['MouseGene']]
mouse_gene_rank = pd.concat([mouse_gene_rank,mouse_median_expr.loc[mouse_gene_rank.index],mouse_mean_expr.loc[mouse_gene_rank.index]],axis=1)
mouse_gene_rank.columns = ['MouseGene','MouseRank','MouseMedianExpr','MouseMeanExpr']
mouse_gene_rank.index = [gene.split('|')[0] for gene in mouse_gene_rank.index]

gene_names = pd.DataFrame(list(mouse_gene_rank.index), columns=['Mouse_GeneName'], index=mouse_gene_rank.index)
mouse_gene_rank = pd.concat([gene_names, mouse_gene_rank], axis=1)
mouse_gene_rank = mouse_gene_rank.groupby('Mouse_GeneName').apply(lambda x: x.loc[x['MouseMedianExpr']==np.max(x['MouseMedianExpr'])].iloc[0,:])
mouse_gene_rank = mouse_gene_rank.sort_values(by="MouseRank")

human_gene_rank = list(human_median_expr.sort_values(ascending=False).index)
human_gene_rank = pd.DataFrame([human_gene_rank, [i+1 for i in range(len(human_gene_rank))]], index=['HumanGene','HumanRank']).T
human_gene_rank.index = [gene for gene in human_gene_rank['HumanGene']]
human_gene_rank = pd.concat([human_gene_rank,human_median_expr.loc[human_gene_rank.index],human_mean_expr.loc[human_gene_rank.index]],axis=1)
human_gene_rank.columns = ['HumanGene','HumanRank','HumanMedianExpr','HumanMeanExpr']
human_gene_rank.index = [gene.split('|')[0] for gene in human_gene_rank.index]

gene_names = pd.DataFrame(list(human_gene_rank.index), columns=['Human_GeneName'], index=human_gene_rank.index)
human_gene_rank = pd.concat([gene_names, human_gene_rank], axis=1)
human_gene_rank = human_gene_rank.groupby('Human_GeneName').apply(lambda x: x.loc[x['HumanMedianExpr']==np.max(x['HumanMedianExpr'])].iloc[0,:])
human_gene_rank = human_gene_rank.sort_values(by="HumanRank")

mouse_quantified_genes = list(set([gene.split('|')[0] for gene in mouse_gene_rank.index]))
human_quantified_genes = list(set([gene.split('|')[0] for gene in human_gene_rank.index]))

curr_new_homologs = new_homologs.groupby(by='Mouse_gene_name').apply(correct_homolog_pairs, base='Mouse_gene_name', target='Gene_name', quantified_genes=human_quantified_genes)
curr_new_homologs = curr_new_homologs.groupby(by='Gene_name').apply(correct_homolog_pairs, base='Gene_name', target='Mouse_gene_name', quantified_genes=mouse_quantified_genes)
curr_new_homologs.index = curr_new_homologs['Mouse_gene_name']

comm_genes = list(set(mouse_gene_rank.index.intersection(curr_new_homologs['Mouse_gene_name'])))

mouse_gene_rank = mouse_gene_rank.loc[comm_genes]

mouse_gene_rank.index.name = ''
curr_new_homologs.index.name = ''
human_gene_rank.index.name = ''

mmu_rank_table = mouse_gene_rank.merge(curr_new_homologs, left_on='Mouse_GeneName', right_on='Mouse_gene_name', how='left')
rank = mmu_rank_table.merge(human_gene_rank, left_on='Gene_name', right_on='Human_GeneName', how='inner')
ids = ['%s_%s' %(rank.iloc[i]['Mouse_GeneName'], rank.iloc[i]['Gene_name']) for i in range(rank.shape[0])]
rank.index = ids

rank_filtered = rank.copy()
flag = (rank_filtered['MouseMedianExpr'] < 1) & (rank['HumanMedianExpr'] < 1)
rank_filtered = rank_filtered[~flag]
rank_filtered.shape

mouse_ranks = rank_filtered.iloc[:,:5].drop_duplicates()
mouse_ranks.loc[mouse_ranks.sort_values(by='MouseMedianExpr', ascending=False).index,'Mouse_Rank'] = [i+1 for i in range(mouse_ranks.shape[0])]
mouse_ranks.loc[mouse_ranks['MouseMedianExpr'] == 0,'Mouse_Rank'] = mouse_ranks.loc[mouse_ranks['MouseMedianExpr'] > 0]['Mouse_Rank'].max() + 1
mouse_ranks.index = mouse_ranks['Mouse_GeneName']

human_ranks = rank_filtered[['Human_GeneName', 'HumanGene', 'HumanRank', 'HumanMedianExpr', 'HumanMeanExpr']].drop_duplicates()
human_ranks.loc[human_ranks.sort_values(by='HumanMedianExpr', ascending=False).index,'Human_Rank'] = [i+1 for i in range(human_ranks.shape[0])]
human_ranks.loc[human_ranks['HumanMedianExpr'] == 0,'Human_Rank'] = human_ranks.loc[human_ranks['HumanMedianExpr'] > 0]['Human_Rank'].max() + 1
human_ranks.index = human_ranks['Human_GeneName']

rank_filtered['Mouse_Rank'] = list(mouse_ranks.loc[rank_filtered['Mouse_GeneName'].values]['Mouse_Rank'].values)
rank_filtered['Human_Rank'] = list(human_ranks.loc[rank_filtered['Human_GeneName'].values]['Human_Rank'].values)

new_ranks = rank_filtered.copy()
from scipy.stats.mstats import gmean
rp = [gmean([new_ranks.iloc[ii]['Mouse_Rank'],new_ranks.iloc[ii]['Human_Rank']]) for ii in range(new_ranks.shape[0])]
final_rank = pd.concat([new_ranks, pd.DataFrame(rp, index=new_ranks.index, columns=['RP'])], axis=1)

order_final_rank = final_rank.sort_values(by="RP")
comm_gene_rank = order_final_rank.query('MouseMedianExpr>=4').query('HumanMedianExpr>=4')
comm_gene_rank.loc[comm_gene_rank.sort_values(by="MouseMedianExpr", ascending=False).index, 'Mouse_Rank'] = [i+1 for i in range(comm_gene_rank.shape[0])]
comm_gene_rank.loc[comm_gene_rank.sort_values(by="HumanMedianExpr", ascending=False).index, 'Human_Rank'] = [i+1 for i in range(comm_gene_rank.shape[0])]
rp = [gmean([comm_gene_rank.iloc[ii]['Mouse_Rank'],comm_gene_rank.iloc[ii]['Human_Rank']]) for ii in range(comm_gene_rank.shape[0])]
comm_gene_rank['RP'] = rp
comm_gene_rank = comm_gene_rank.sort_values(by='RP')
comm_gene_rank.to_excel('%s/hLM1_mKC1_comm_gene_rank.xlsx' %(outputDIR), index=True)
top_comm_genes = list(comm_gene_rank.index)

mouse_ctype_genes = list(mouse_cell_type_genes.query('sigCluster=="kc1"')['GeneName'].values)
human_ctype_genes = list(human_cell_type_genes.query('sigCluster=="lm1"')['GeneName'].values)

hLM1_mKC1_comm_ctype_specific = [gene for gene in top_comm_genes if gene.split('_')[0] in mouse_ctype_genes]
hLM1_mKC1_comm_ctype_specific = [gene for gene in hLM1_mKC1_comm_ctype_specific if gene.split('_')[1] in human_ctype_genes]

# human LM3 and mouse MoMac1
mouse_cells_in_use = list(mmu_annot.query('sampleCluster=="MoMac1"')['Cell'].values)
human_cells_in_use = list(hsa_annot.query('sampleCluster=="LM3"')['Cell'].values)

curr_human_expr = human_rpkms[human_cells_in_use]
curr_mouse_expr = mouse_rpkms[mouse_cells_in_use]

mouse_mean_expr = curr_mouse_expr.apply(np.mean, axis=1)
human_mean_expr = curr_human_expr.apply(np.mean, axis=1)

mouse_median_expr = curr_mouse_expr.apply(np.median, axis=1)
human_median_expr = curr_human_expr.apply(np.median, axis=1)

mouse_gene_rank = list(mouse_median_expr.sort_values(ascending=False).index)
mouse_gene_rank = pd.DataFrame([mouse_gene_rank, [i+1 for i in range(len(mouse_gene_rank))]], index=['MouseGene','MouseRank']).T
mouse_gene_rank.index = [gene for gene in mouse_gene_rank['MouseGene']]
mouse_gene_rank = pd.concat([mouse_gene_rank,mouse_median_expr.loc[mouse_gene_rank.index],mouse_mean_expr.loc[mouse_gene_rank.index]],axis=1)
mouse_gene_rank.columns = ['MouseGene','MouseRank','MouseMedianExpr','MouseMeanExpr']
mouse_gene_rank.index = [gene.split('|')[0] for gene in mouse_gene_rank.index]

gene_names = pd.DataFrame(list(mouse_gene_rank.index), columns=['Mouse_GeneName'], index=mouse_gene_rank.index)
mouse_gene_rank = pd.concat([gene_names, mouse_gene_rank], axis=1)
mouse_gene_rank = mouse_gene_rank.groupby('Mouse_GeneName').apply(lambda x: x.loc[x['MouseMedianExpr']==np.max(x['MouseMedianExpr'])].iloc[0,:])
mouse_gene_rank = mouse_gene_rank.sort_values(by="MouseRank")

human_gene_rank = list(human_median_expr.sort_values(ascending=False).index)
human_gene_rank = pd.DataFrame([human_gene_rank, [i+1 for i in range(len(human_gene_rank))]], index=['HumanGene','HumanRank']).T
human_gene_rank.index = [gene for gene in human_gene_rank['HumanGene']]
human_gene_rank = pd.concat([human_gene_rank,human_median_expr.loc[human_gene_rank.index],human_mean_expr.loc[human_gene_rank.index]],axis=1)
human_gene_rank.columns = ['HumanGene','HumanRank','HumanMedianExpr','HumanMeanExpr']
human_gene_rank.index = [gene.split('|')[0] for gene in human_gene_rank.index]

gene_names = pd.DataFrame(list(human_gene_rank.index), columns=['Human_GeneName'], index=human_gene_rank.index)
human_gene_rank = pd.concat([gene_names, human_gene_rank], axis=1)
human_gene_rank = human_gene_rank.groupby('Human_GeneName').apply(lambda x: x.loc[x['HumanMedianExpr']==np.max(x['HumanMedianExpr'])].iloc[0,:])
human_gene_rank = human_gene_rank.sort_values(by="HumanRank")

mouse_quantified_genes = list(set([gene.split('|')[0] for gene in mouse_gene_rank.index]))
human_quantified_genes = list(set([gene.split('|')[0] for gene in human_gene_rank.index]))

curr_new_homologs = new_homologs.groupby(by='Mouse_gene_name').apply(correct_homolog_pairs, base='Mouse_gene_name', target='Gene_name', quantified_genes=human_quantified_genes)
curr_new_homologs = curr_new_homologs.groupby(by='Gene_name').apply(correct_homolog_pairs, base='Gene_name', target='Mouse_gene_name', quantified_genes=mouse_quantified_genes)
curr_new_homologs.index = curr_new_homologs['Mouse_gene_name']
comm_genes = list(set(mouse_gene_rank.index.intersection(curr_new_homologs['Mouse_gene_name'])))

mouse_gene_rank = mouse_gene_rank.loc[comm_genes]
mouse_gene_rank.index.name = ''
curr_new_homologs.index.name = ''
human_gene_rank.index.name = ''

mmu_rank_table = mouse_gene_rank.merge(curr_new_homologs, left_on='Mouse_GeneName', right_on='Mouse_gene_name', how='left')
rank = mmu_rank_table.merge(human_gene_rank, left_on='Gene_name', right_on='Human_GeneName', how='inner')

ids = ['%s_%s' %(rank.iloc[i]['Mouse_GeneName'], rank.iloc[i]['Gene_name']) for i in range(rank.shape[0])]
rank.index = ids

rank_filtered = rank.copy()
flag = (rank_filtered['MouseMedianExpr'] < 1) & (rank['HumanMedianExpr'] < 1)
rank_filtered = rank_filtered[~flag]

mouse_ranks = rank_filtered.iloc[:,:5].drop_duplicates()
mouse_ranks.loc[mouse_ranks.sort_values(by='MouseMedianExpr', ascending=False).index,'Mouse_Rank'] = [i+1 for i in range(mouse_ranks.shape[0])]
mouse_ranks.loc[mouse_ranks['MouseMedianExpr'] == 0,'Mouse_Rank'] = mouse_ranks.loc[mouse_ranks['MouseMedianExpr'] > 0]['Mouse_Rank'].max() + 1
mouse_ranks.index = mouse_ranks['Mouse_GeneName']

human_ranks = rank_filtered[['Human_GeneName', 'HumanGene', 'HumanRank', 'HumanMedianExpr', 'HumanMeanExpr']].drop_duplicates()
human_ranks.loc[human_ranks.sort_values(by='HumanMedianExpr', ascending=False).index,'Human_Rank'] = [i+1 for i in range(human_ranks.shape[0])]
human_ranks.loc[human_ranks['HumanMedianExpr'] == 0,'Human_Rank'] = human_ranks.loc[human_ranks['HumanMedianExpr'] > 0]['Human_Rank'].max() + 1
human_ranks.index = human_ranks['Human_GeneName']

rank_filtered['Mouse_Rank'] = list(mouse_ranks.loc[rank_filtered['Mouse_GeneName'].values]['Mouse_Rank'].values)
rank_filtered['Human_Rank'] = list(human_ranks.loc[rank_filtered['Human_GeneName'].values]['Human_Rank'].values)

new_ranks = rank_filtered.copy()
from scipy.stats.mstats import gmean
rp = [gmean([new_ranks.iloc[ii]['Mouse_Rank'],new_ranks.iloc[ii]['Human_Rank']]) for ii in range(new_ranks.shape[0])]
final_rank = pd.concat([new_ranks, pd.DataFrame(rp, index=new_ranks.index, columns=['RP'])], axis=1)

order_final_rank = final_rank.sort_values(by="RP")
comm_gene_rank = order_final_rank.query('MouseMedianExpr>=4').query('HumanMedianExpr>=4')
comm_gene_rank.loc[comm_gene_rank.sort_values(by="MouseMedianExpr", ascending=False).index, 'Mouse_Rank'] = [i+1 for i in range(comm_gene_rank.shape[0])]
comm_gene_rank.loc[comm_gene_rank.sort_values(by="HumanMedianExpr", ascending=False).index, 'Human_Rank'] = [i+1 for i in range(comm_gene_rank.shape[0])]

rp = [gmean([comm_gene_rank.iloc[ii]['Mouse_Rank'],comm_gene_rank.iloc[ii]['Human_Rank']]) for ii in range(comm_gene_rank.shape[0])]
comm_gene_rank['RP'] = rp
comm_gene_rank = comm_gene_rank.sort_values(by='RP')
top_comm_genes = list(comm_gene_rank.index)

mouse_ctype_genes = list(mouse_cell_type_genes.query('sigCluster=="momac1"')['GeneName'].values)
human_ctype_genes = list(human_cell_type_genes.query('sigCluster=="lm3"')['GeneName'].values)
hLM3_mMoMac1_comm_ctype_specific = [gene for gene in top_comm_genes if gene.split('_')[0] in mouse_ctype_genes]
hLM3_mMoMac1_comm_ctype_specific = [gene for gene in hLM3_mMoMac1_comm_ctype_specific if gene.split('_')[1] in human_ctype_genes]

# human LM4 and mouse MoMac2
mouse_cells_in_use = list(mmu_annot.query('sampleCluster=="MoMac2"')['Cell'].values)
human_cells_in_use = list(hsa_annot.query('sampleCluster=="LM4"')['Cell'].values)

curr_human_expr = human_rpkms[human_cells_in_use]
curr_mouse_expr = mouse_rpkms[mouse_cells_in_use]

mouse_mean_expr = curr_mouse_expr.apply(np.mean, axis=1)
human_mean_expr = curr_human_expr.apply(np.mean, axis=1)

mouse_median_expr = curr_mouse_expr.apply(np.median, axis=1)
human_median_expr = curr_human_expr.apply(np.median, axis=1)

mouse_gene_rank = list(mouse_median_expr.sort_values(ascending=False).index)
mouse_gene_rank = pd.DataFrame([mouse_gene_rank, [i+1 for i in range(len(mouse_gene_rank))]], index=['MouseGene','MouseRank']).T
mouse_gene_rank.index = [gene for gene in mouse_gene_rank['MouseGene']]
mouse_gene_rank = pd.concat([mouse_gene_rank,mouse_median_expr.loc[mouse_gene_rank.index],mouse_mean_expr.loc[mouse_gene_rank.index]],axis=1)
mouse_gene_rank.columns = ['MouseGene','MouseRank','MouseMedianExpr','MouseMeanExpr']
mouse_gene_rank.index = [gene.split('|')[0] for gene in mouse_gene_rank.index]

gene_names = pd.DataFrame(list(mouse_gene_rank.index), columns=['Mouse_GeneName'], index=mouse_gene_rank.index)
mouse_gene_rank = pd.concat([gene_names, mouse_gene_rank], axis=1)
mouse_gene_rank = mouse_gene_rank.groupby('Mouse_GeneName').apply(lambda x: x.loc[x['MouseMedianExpr']==np.max(x['MouseMedianExpr'])].iloc[0,:])
mouse_gene_rank = mouse_gene_rank.sort_values(by="MouseRank")

human_gene_rank = list(human_median_expr.sort_values(ascending=False).index)
human_gene_rank = pd.DataFrame([human_gene_rank, [i+1 for i in range(len(human_gene_rank))]], index=['HumanGene','HumanRank']).T
human_gene_rank.index = [gene for gene in human_gene_rank['HumanGene']]
human_gene_rank = pd.concat([human_gene_rank,human_median_expr.loc[human_gene_rank.index],human_mean_expr.loc[human_gene_rank.index]],axis=1)
human_gene_rank.columns = ['HumanGene','HumanRank','HumanMedianExpr','HumanMeanExpr']
human_gene_rank.index = [gene.split('|')[0] for gene in human_gene_rank.index]

gene_names = pd.DataFrame(list(human_gene_rank.index), columns=['Human_GeneName'], index=human_gene_rank.index)
human_gene_rank = pd.concat([gene_names, human_gene_rank], axis=1)
human_gene_rank = human_gene_rank.groupby('Human_GeneName').apply(lambda x: x.loc[x['HumanMedianExpr']==np.max(x['HumanMedianExpr'])].iloc[0,:])
human_gene_rank = human_gene_rank.sort_values(by="HumanRank")

mouse_quantified_genes = list(set([gene.split('|')[0] for gene in mouse_gene_rank.index]))
human_quantified_genes = list(set([gene.split('|')[0] for gene in human_gene_rank.index]))

curr_new_homologs = new_homologs.groupby(by='Mouse_gene_name').apply(correct_homolog_pairs, base='Mouse_gene_name', target='Gene_name', quantified_genes=human_quantified_genes)
curr_new_homologs = curr_new_homologs.groupby(by='Gene_name').apply(correct_homolog_pairs, base='Gene_name', target='Mouse_gene_name', quantified_genes=mouse_quantified_genes)
curr_new_homologs.index = curr_new_homologs['Mouse_gene_name']

comm_genes = list(set(mouse_gene_rank.index.intersection(curr_new_homologs['Mouse_gene_name'])))
mouse_gene_rank = mouse_gene_rank.loc[comm_genes]
mouse_gene_rank.index.name = ''
curr_new_homologs.index.name = ''
human_gene_rank.index.name = ''

mmu_rank_table = mouse_gene_rank.merge(curr_new_homologs, left_on='Mouse_GeneName', right_on='Mouse_gene_name', how='left')
rank = mmu_rank_table.merge(human_gene_rank, left_on='Gene_name', right_on='Human_GeneName', how='inner')
ids = ['%s_%s' %(rank.iloc[i]['Mouse_GeneName'], rank.iloc[i]['Gene_name']) for i in range(rank.shape[0])]
rank.index = ids

rank_filtered = rank.copy()
flag = (rank_filtered['MouseMedianExpr'] < 1) & (rank['HumanMedianExpr'] < 1)
rank_filtered = rank_filtered[~flag]

mouse_ranks = rank_filtered.iloc[:,:5].drop_duplicates()
mouse_ranks.loc[mouse_ranks.sort_values(by='MouseMedianExpr', ascending=False).index,'Mouse_Rank'] = [i+1 for i in range(mouse_ranks.shape[0])]
mouse_ranks.loc[mouse_ranks['MouseMedianExpr'] == 0,'Mouse_Rank'] = mouse_ranks.loc[mouse_ranks['MouseMedianExpr'] > 0]['Mouse_Rank'].max() + 1
mouse_ranks.index = mouse_ranks['Mouse_GeneName']

human_ranks = rank_filtered[['Human_GeneName', 'HumanGene', 'HumanRank', 'HumanMedianExpr', 'HumanMeanExpr']].drop_duplicates()
human_ranks.loc[human_ranks.sort_values(by='HumanMedianExpr', ascending=False).index,'Human_Rank'] = [i+1 for i in range(human_ranks.shape[0])]
human_ranks.loc[human_ranks['HumanMedianExpr'] == 0,'Human_Rank'] = human_ranks.loc[human_ranks['HumanMedianExpr'] > 0]['Human_Rank'].max() + 1
human_ranks.index = human_ranks['Human_GeneName']

rank_filtered['Mouse_Rank'] = list(mouse_ranks.loc[rank_filtered['Mouse_GeneName'].values]['Mouse_Rank'].values)
rank_filtered['Human_Rank'] = list(human_ranks.loc[rank_filtered['Human_GeneName'].values]['Human_Rank'].values)

new_ranks = rank_filtered.copy()
from scipy.stats.mstats import gmean
rp = [gmean([new_ranks.iloc[ii]['Mouse_Rank'],new_ranks.iloc[ii]['Human_Rank']]) for ii in range(new_ranks.shape[0])]
final_rank = pd.concat([new_ranks, pd.DataFrame(rp, index=new_ranks.index, columns=['RP'])], axis=1)

order_final_rank = final_rank.sort_values(by="RP")
comm_gene_rank = order_final_rank.query('MouseMedianExpr>=4').query('HumanMedianExpr>=4')

comm_gene_rank.loc[comm_gene_rank.sort_values(by="MouseMedianExpr", ascending=False).index, 'Mouse_Rank'] = [i+1 for i in range(comm_gene_rank.shape[0])]
comm_gene_rank.loc[comm_gene_rank.sort_values(by="HumanMedianExpr", ascending=False).index, 'Human_Rank'] = [i+1 for i in range(comm_gene_rank.shape[0])]

rp = [gmean([comm_gene_rank.iloc[ii]['Mouse_Rank'],comm_gene_rank.iloc[ii]['Human_Rank']]) for ii in range(comm_gene_rank.shape[0])]
comm_gene_rank['RP'] = rp
comm_gene_rank = comm_gene_rank.sort_values(by='RP')
top_comm_genes = list(comm_gene_rank.index)

mouse_ctype_genes = list(mouse_cell_type_genes.query('sigCluster=="momac2"')['GeneName'].values)
human_ctype_genes = list(human_cell_type_genes.query('sigCluster=="lm4"')['GeneName'].values)

hLM4_mMoMac2_comm_ctype_specific = [gene for gene in top_comm_genes if gene.split('_')[0] in mouse_ctype_genes]
hLM4_mMoMac2_comm_ctype_specific = [gene for gene in hLM4_mMoMac2_comm_ctype_specific if gene.split('_')[1] in human_ctype_genes]
