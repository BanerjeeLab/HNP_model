from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from shutil import copy
from numpy.linalg import inv
import math
import matplotlib
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import seaborn as sns
import matplotlib.colors as colors
import os

matplotlib.use('Agg')

class Vertex:
    def __init__(self, vert_cell, tipo):
        self.x = vert_cell[0]
        self.y = vert_cell[1]
        self.tipo = tipo

class Cell(Vertex):
    def __init__(self, vertices,centro):
        self.vs = vertices
        self.c = centro

    def draw1(self):
        if self.c[0]>x_min and self.c[0]<x_max and self.c[1]>y_min and self.c[1]<y_max:
            vert1 = []
            for i in range(len(self.vs)-1):
                vert1.append([self.vs[i].x,self.vs[i].y])
            if len(vert1)>0:
                polygon1 = Polygon(vert1, color='AliceBlue')
                patches.append(polygon1)

    def draw2(self):
        if self.c[0]>x_min and self.c[0]<x_max and self.c[1]>y_min and self.c[1]<y_max:
            vert1 = []
            for i in range(len(self.vs)-1):
                vert1.append([self.vs[i].x,self.vs[i].y])
            if len(vert1)>0:
                polygon1 = Polygon(vert1, color='white')
                patches_red.append(polygon1)

class Tissue(Cell):
    def __init__(self, celulas):
        self.cs = celulas


    def draw2(self):
        for i in range(len(self.cs)):
            if i not in [boundary,gap]:
               self.cs[i].draw1()

def truncate_colormap(cmap, minval=0.0, maxval=1.0, n=100):
    new_cmap = colors.LinearSegmentedColormap.from_list(
        'trunc({n},{a:.2f},{b:.2f})'.format(n=cmap.name, a=minval, b=maxval),
        cmap(np.linspace(minval, maxval, n)))
    return new_cmap

def crear_T(texto0,texto1,texto2):
    data = np.loadtxt(texto0)
    celdas = np.loadtxt(texto1)
    celulas = np.loadtxt(texto2)
    x = data[:,1]
    y = data[:,2]
    t = data[:,5]
    celda = celdas[:,0]
    v_celda = celdas[:,1]
    nros_c = data[:,0]
    centro_x = celulas[:,3]
    centro_y = celulas[:,4]
    pos0 = []
    cells0 = []
    tipo = []

    for i in range(int(nros_c[len(nros_c)-1])+1):
        pos0.append((x[i],y[i],0))
        if t[i]==-1:
            tipo.append(3)
        else:
            tipo.append(4)

    bs = [ [] ]

    k=0
    ll= 0
    for i in range(int(celda[len(celda)-1])+1):
        while celda[ll] == k and ll<len(celda)-1:
            bs[k].append(int(v_celda[ll]))
            ll= ll + 1
        k = k+1
        bs.append([])
    bs[len(bs)-2].append(int(v_celda[ll]))

    a = 0
    for i in range(int(celda[len(celda)-1])+1):
        print(i)
        c = []
        for number in bs[i]:
            globals()['v%s' % number] = Vertex(pos0[number],tipo[number])
            c.append(Vertex(pos0[number],tipo[number]))
        c.append(c[0])

        globals()['c%s' % a] = Cell(c,[centro_x[i],centro_y[i]])
        cells0.append(Cell(c,[centro_x[i],centro_y[i]]))
        a = a+1

    T1 = Tissue(cells0)
    ax1.set_xlim([3,L_x-3])
    ax1.set_ylim([5,L_y-5])

    plt.axis('off')
    T1.draw2()
    return T1

# === Setup and Cleanup ===
directories = ['snapshots_nematic']
# Clean up old directories
for directory in directories:
    if os.path.exists(directory):
        shutil.rmtree(directory)
    os.mkdir(directory)
    

plt.close('all')
factor = np.sqrt(2/(3*np.sqrt(3)))
L_x = 51*np.sqrt(3)*factor
L_y = 108*np.sqrt(3)*(np.sqrt(3)/2)*factor
x_min, x_max =0+4,L_x-4
y_min,y_max = 0+4,L_y-4

plt.figure(figsize=(3,5))
fig, ax1 = plt.subplots()
boundary = 1427
gap = 1428
for i in range(130):
    fig, ax1 = plt.subplots()
    patches = []
    patches_red = []
    T1 = crear_T('./results/'+str(i)+'_vertices.txt', './results/'+str(i)+'_celda.txt','./results/'+str(i)+'_celulas.txt')
    plt.axis('off')
    plt.axis('scaled')
    fig.set_size_inches(4, 5)
    texto0 = './results/'+str(i)+'_celulas.txt'
    data0 = np.loadtxt(texto0)
    cx = data0[:,3]
    cy = data0[:,4]
    nx = data0[:,5]
    ny = data0[:,6]
    nx_normalize = []
    ny_normalize = []
    cxn=[]
    cyn=[]
    modulos = []
    for im in range(len(nx)-2):
        cxn.append(cx[im])
        cyn.append(cy[im])
        mm = np.sqrt(nx[im]**2 + ny[im]**2)
        modulos.append(np.sqrt(nx[im]**2 + ny[im]**2))
        if mm<=0.00001:
            nx_normalize.append(0)
            ny_normalize.append(0)
        else:
            nx_normalize.append(nx[im]/mm)
            ny_normalize.append(ny[im]/mm)

    ncmap =plt.get_cmap('plasma_r')
    cmap = truncate_colormap(ncmap, 0.3, 1)
    norm = matplotlib.colors.Normalize(vmin=0, vmax=1)
    cols = []

    for j in range(len(cxn)):
        im1 = plt.quiver(cxn[j],cyn[j],nx_normalize[j],ny_normalize[j],headaxislength=0.,headlength=0,pivot='middle', color='w', scale=100, width=.004,zorder=5)
        cols.append(cmap(norm(modulos[j])))
    pa = PatchCollection(patches, match_original=False,color=cols, edgecolor='k',linewidths=.4, zorder=2)
    ax1.add_collection(pa)
    
    if len(T1.cs)==1429:
        T1.cs[1428].draw2()
    pa2 = PatchCollection(patches_red, match_original=False,color="white", edgecolor='r',linewidths=2, zorder=10,cmap=cmap)
    ax1.add_collection(pa2)

    plt.xlim([5,50])
    plt.ylim([15,90])
    plt.tight_layout(pad=1.6)

    cbar = fig.colorbar(pa2)
    cbar.ax.tick_params(labelsize=10)
    cbar.set_label('Magnitude $Q$ tensor', rotation=90)
    plt.savefig('./snapshots_nematic/'+str(i)+'.png', dpi=400)
    plt.clf()
