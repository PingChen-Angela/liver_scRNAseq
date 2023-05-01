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

# load input data
indir = 'Smartseq2_NPCs/'
outdir = 'Smartseq2_NPCs/results'
human_npc_ctype_annot = pd.read_table('%s/human_NPCs_cell_type_annotation_final.csv' %(indir), header=0, index_col=None, sep="\t")
mouse_npc_ctype_annot = pd.read_table('%s/mouse_NPCs_cell_type_annotation_final.csv' %(indir), header=0, index_col=None, sep="\t")
umap_loc = pd.read_table('%s/seurat_integrated_umap.csv' %(outdir), header=0, index_col=0, sep="\t")

human_npc_ctype_annot.index = human_npc_ctype_annot['sampleName'].to_list()
mouse_npc_ctype_annot.index = mouse_npc_ctype_annot['sampleName'].to_list()

human_npc_ctype_annot['Dataset'] = 'Human NPCs'
mouse_npc_ctype_annot['Dataset'] = 'Mouse NPCs'
merged_cell_annotation = pd.concat([human_npc_ctype_annot, mouse_npc_ctype_annot], axis=0)
merged_cell_annotation.columns = ['CellID','CellType','Dataset']
merged_cell_annotation = merged_cell_annotation.loc[umap_loc.index]

ctype_annot = merged_cell_annotation.copy()

# visualize human NPC cell types
target_cells = list(ctype_annot.query('Dataset=="Human NPCs"').index)
curr_cell_annot = ctype_annot.query('Dataset=="Human NPCs"')
ref_annot = ctype_annot.query('Dataset=="Mouse NPCs"')
ref_annot['CellType'] = 'Mouse NPCs'
curr_cell_annot = pd.concat([curr_cell_annot, ref_annot], axis=0)
color_palette = sns.color_palette("tab20",13)
color_palette = color_palette.as_hex()
color_dict = {ctype: color_palette[ii] for ii, ctype in enumerate(curr_cell_annot['CellType'].drop_duplicates().to_list()) if ctype != "All mouse NPCs"}
color_dict['Mouse NPCs'] = 'gainsboro'

curr_x_r = umap_loc.loc[ctype_annot.index].values
reduce_plot(curr_x_r, annot=curr_cell_annot.loc[ctype_annot.index]['CellType'], title='', 
            prefix='UMAP', size=80, 
            fontsize=10, pcX=1, pcY=2, color_palette=color_dict,
            figsize=(6,6), outdir=outdir, filename='integration_human_NPCs_umap', 
            legend_loc='bottom', legend_ncol=2,
            add_sample_label=False, ordered_sample_labels=None, edgecolors='none')

# visualize mouse NPC cell types
target_cells = list(ctype_annot.query('Dataset=="Mouse NPCs"').index)
curr_cell_annot = ctype_annot.query('Dataset=="Mouse NPCs"')
ref_annot = ctype_annot.query('Dataset=="Human NPCs"')
ref_annot['CellType'] = 'Human NPCs'
curr_cell_annot = pd.concat([curr_cell_annot, ref_annot], axis=0)

color_palette1 = sns.color_palette("Dark2")
color_palette1 = color_palette1.as_hex()
color_palette2 = sns.color_palette("Set3", 10)
color_palette2 = color_palette2.as_hex()
color_palette2 = color_palette2[2:-2]+color_palette2[-1:]
color_palette = color_palette1 + color_palette2
color_dict = {ctype: color_palette[ii] for ii, ctype in enumerate(ctype_annot.query('Dataset=="Mouse NPCs"')['CellType'].drop_duplicates().to_list())}
color_dict['Human NPCs'] = 'gainsboro'

curr_x_r = umap_loc.loc[ctype_annot.index].values
reduce_plot(curr_x_r, annot=curr_cell_annot.loc[ctype_annot.index]['CellType'], title='', 
            prefix='UMAP', size=80, 
            fontsize=10, pcX=1, pcY=2, color_palette=color_dict,
            figsize=(6,6), outdir=outdir, filename='integration_mouse_NPCs_umap', 
            legend_loc='bottom', legend_ncol=2,
            add_sample_label=False, ordered_sample_labels=None, edgecolors='none')
