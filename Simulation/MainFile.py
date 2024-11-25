from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from shutil import copy
import shutil
from numpy import linalg as LA
from shapely.geometry import Polygon
from numpy.linalg import inv
from numpy.linalg import pinv
import math
import os
import json
from classes_and_functions import Vertex, Cell, Tissue

# === Setup and Cleanup ===
directories = ['first_row', 'second_row', 'third_row', 'results']
files_to_delete = ['4-fold.txt', 'T1_perpendicular.txt', 'T1_original.txt']
# Clean up old directories
for directory in directories:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)
# Remove old files
for file in files_to_delete:
    if os.path.exists(file):
        os.remove(file) 

# === Constants and Parameters ===
dt = 0.05
steps_per_save = 100
total_save_times = 130

# Load parameters from the configuration file
with open('config_fig2A.json', 'r') as config_file:
    config = json.load(config_file)
# Access parameters
Th = config['Th']
v0 = config['v0']
ma = config['ma']
md = config['md']
mf = config['mf']
Sigma0 = config['Sigma0']
CrawlingLeader = config['CrawlingLeader']
PrePatterned = config['PrePatterned']

# === Load Initial Tissue Configuration ===
path = 'initial_data'
celdas_data = np.loadtxt(str(path)+'/cell_vertex_id.txt')
celda = [int(item) for item in celdas_data[:,1]]
v_celda = [int(item) for item in celdas_data[:,2]]

vertex_data = np.loadtxt(str(path)+'/vertex_data.txt')
x = vertex_data[:,2]
y = vertex_data[:,3]
z = np.zeros(len(x))

top = np.loadtxt(str(path)+'/adjacents_vertices_cells.txt')
v_top = [int(item) for item in top[:,1]]
adj1 = [int(item) for item in top[:,2]]
adj2 = [int(item) for item in top[:,3]]
adj3 = [int(item) for item in top[:,4]]
cell1 = [int(item) for item in top[:,5]]
cell2 = [int(item) for item in top[:,6]]
cell3 = [int(item) for item in top[:,7]]
adjs_tipo1 = list(zip(zip(adj1, adj2), zip(adj3, adj1), zip(adj2, adj3)))
cells_tipo1 =list(zip(cell1, cell2, cell3))

n_celulas = int(celda[len(celda)-1]+1)
n_vertices = int(len(x))

# Creation of the tissue structure
exec(open("tissue_creation.py").read())

# Pre-patterned simulation: initial nematic condition 
if PrePatterned == 'YES':  
    for c in T1.R1:
        T1.cell_vec_np_update(c)
        rot = np.pi/2
        nqx = np.cos(rot)*T1.cs[c].vec_np[0] + np.sin(rot)*T1.cs[c].vec_np[1]
        nqy = np.cos(rot)*T1.cs[c].vec_np[1] - np.sin(rot)*T1.cs[c].vec_np[0]
        T1.cs[c].vec_nq = np.array([nqx,nqy,0])
        modq = np.linalg.norm(T1.cs[c].vec_nq)
        dosphi = np.arctan2(nqy,nqx)
        if dosphi <0:
            dosphi = dosphi + 2*np.pi
        phi = 2*dosphi
        T1.cs[c].vec_q = np.array([modq * np.cos(phi), modq * np.sin(phi), 0])


T1.cal_polygons_inicial()
T1.data()
copy('./results/data_vertices.txt','./results/0_vertices.txt')
copy('./results/data_celulas.txt','./results/0_celulas.txt')
copy('./results/data_celda.txt','./results/0_celda.txt')
copy('./first_row/data_1st.txt','./first_row/0_data_1st.txt')
copy('./second_row/data_2nd.txt','./second_row/0_data_2nd.txt')
copy('./third_row/data_3rd.txt','./third_row/0_data_3rd.txt')


check = 0
for i in range(int(steps_per_save * total_save_times)-1):
    if len(T1.c_hole.ind) ==5 and check==0 and T1.c_hole.area<0.01:
        vfirst = T1.c_hole.ind[0]
        vsecond = T1.c_hole.ind[1]
        T1.T1_swap_hole(vfirst, vsecond)
        T1.R1 = []
        T1.extrusion(T1.c_hole.n)
        check = check+1

    # Choose evolution method based on simulation type
    # R1: R1 leader crawling model
    # R2: R2 leader crawling model
    if CrawlingLeader == 'R1':
        T1.evol_vertex_R1leader((i+1)*dt)
    elif CrawlingLeader == 'R2':
        T1.evol_vertex((i+1)*dt)

    c = (i + 1) % steps_per_save
    d = int((i + 1) / steps_per_save)
    if c == 0 :
        T1.data()
        copy('./results/data_vertices.txt', './results/'+ str(d) + '_vertices.txt')
        copy('./results/data_celulas.txt', './results/'+str(d) + '_celulas.txt')
        copy('./results/data_celda.txt','./results/'+str(d) + '_celda.txt')
        copy('./first_row/data_1st.txt','./first_row/'+str(d) + '_data_1st.txt')
        copy('./second_row/data_2nd.txt','./second_row/'+str(d) + '_data_2nd.txt')
        copy('./third_row/data_3rd.txt','./third_row/'+str(d) + '_data_3rd.txt')
