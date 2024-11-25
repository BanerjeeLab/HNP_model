from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from shutil import copy
from numpy import linalg as LA
from shapely.geometry import Polygon
from numpy.linalg import inv
from numpy.linalg import pinv
import math

class Vertex:
    def __init__(self, id,vert_cell, celulas, ady):
        self.n = id
        self.r = vert_cell
        self.cells = celulas
        self.ady = ady
        self.vel = np.array([0,0,0])
        self.eje = [-1,-1]
        self.par = [-1]
    
class Cell:
    def __init__(self, numero, indices):
        self.ind = indices
        self.n = numero
        self.area = 0
        self.per = 0
        self.pol = Polygon([ ])
        self.center = np.array([0,0,0])
        self.vel_vec_p = np.array([0,0,0])
        self.vec_p = np.array([0,0,0])
        self.vec_np = np.array([0,0,0])
        self.vel_vec_q = np.array([0,0,0])
        self.vec_q = np.array([0,0,0])
        self.vec_nq = np.array([0,0,0])
        self.vel = np.array([0,0,0])

    def cal_centro_inicial(self):
        cnew = np.array([self.pol.centroid.x,self.pol.centroid.y,0])
        self.center = cnew

    def cal_centro(self, dt):
        if len(self.ind)>0:
            cpast = self.center
            cnew = np.array([self.pol.centroid.x,self.pol.centroid.y,0])
            self.vel = (cnew - cpast) / (dt)
            self.center = cnew
        else:
            self.vel = np.array([0,0,0])
            self.center = np.array([0,0,0])

    def cal_centro_T1(self):
        if len(self.ind)>0:
            cnew = np.array([self.pol.centroid.x,self.pol.centroid.y,0])
            self.center = cnew
        else:
            self.center = np.array([0,0,0])

    def cal_area(self):
        if len(self.ind)>0:
            self.area = self.pol.area
        else:
            self.area = 0

    def cal_per(self):
        if len(self.ind)>0:
            self.per = self.pol.length
        else:
            self.per = 0

class Tissue:
    def __init__(self, vertices, celulas, c_hole, c_boundary, Th,v0,Sigma0,ma,md,mf,dt):
        self.vs = vertices
        self.cs = celulas
        self.R1 = []
        self.R2 = []
        self.R3 = []
        self.c_hole = c_hole
        self.c_boundary = c_boundary
        self.l_T1 = 0.05
        self.A0 = 1
        self.P0 = 3.5
        self.KA = 1
        self.Th = Th
        self.Tout = Th/10
        self.KP = 0.01
        self.v0 = v0
        self.na = 0.01
        self.nd = 0.01
        self.ma = ma
        self.md = md
        self.mf = mf
        self.dt = dt
        self.Sigma0 = Sigma0

    def first_row(self):
        c = []
        for i in range(len(self.c_hole.ind)):
            for j in range(len(self.vs[self.c_hole.ind[i]].cells)):
                if self.vs[self.c_hole.ind[i]].cells[j] not in c and self.vs[self.c_hole.ind[i]].cells[j]!= self.c_hole.n :
                    c.append(self.vs[self.c_hole.ind[i]].cells[j])
        return c

    def second_row(self,c1):
        c = []
        c_first = c1
        for i in range(len(c_first)):
            v = self.cs[c_first[i]].ind
            for j in range(len(v)):
                for k in self.vs[v[j]].cells:
                    if k not in c and k != self.c_hole.n and k not in c_first:
                        c.append(k)
        return c_first, c

    def third_row(self,c1):
        c = []
        data = self.second_row(c1)
        c_first = data[0]
        c_second = data[1]
        for i in range(len(c_second)):
            v = self.cs[c_second[i]].ind
            for j in range(len(v)):
                for k in self.vs[v[j]].cells:
                    if k not in c and k not in c_second and k not in c_first:
                        c.append(k)
        return c_first, c_second, c

    def cell_vel_p_update(self, c):
        cell = self.cs[c]
        if cell.n == self.c_hole.n or cell.n == self.c_boundary.n:
            cell.vel_vec_p = np.array([0,0,0])
        else:
            if cell.n not in self.R1:
                cells_vecinas = []
                for i in range(len(cell.ind)):
                    for j in range(len(self.vs[cell.ind[i]].cells)):
                        if self.vs[cell.ind[i]].cells[j] not in cells_vecinas and self.vs[cell.ind[i]].cells[j]!= cell.n and self.vs[cell.ind[i]].cells[j]!= self.c_boundary.n:
                            cells_vecinas.append(self.vs[cell.ind[i]].cells[j])
                suma = np.array([0,0,0])
                for k in range(len(cells_vecinas)):
                    suma = suma + (self.cs[cells_vecinas[k]].vec_p - cell.vec_p) 
                cell.vel_vec_p = self.na * suma - self.nd * cell.vec_p

    def cell_vec_p_update(self, c):
        cell = self.cs[c]
        if cell.n == self.c_hole.n or cell.n == self.c_boundary.n:
            cell.vec_p = np.array([0,0,0])
        else:
            if cell.n in self.R1:
                cell.vec_p = np.array([0,0,0])
            elif cell.n in self.R2:
                cells_1row = self.R1
                vecinas_1strow=[]
                for i in range(len(cell.ind)):
                    for k in range(len(self.vs[cell.ind[i]].cells)):
                        ck = self.vs[cell.ind[i]].cells[k]
                        if ck in cells_1row and ck not in vecinas_1strow:
                            vecinas_1strow.append(ck)
                m = np.array([0,0,0])
                n = 0
                for c in vecinas_1strow:
                    m = m + self.cs[c].vec_np
                    n=n+1
                m_mean = m/n
                if np.linalg.norm(m_mean)==0: 
                    cell.vec_p = np.array([0,0,0])
                else:
                    cell.vec_p = m_mean/np.linalg.norm(m_mean)
            else:
                cell.vec_p = cell.vec_p + cell.vel_vec_p * self.dt
    
    def cell_vec_p_update_R1leader(self, c):
        cell = self.cs[c]
        if cell.n == self.c_hole.n or cell.n == self.c_boundary.n:
            cell.vec_p = np.array([0,0,0])
        else:
            if cell.n in self.R1:
                cell.vec_p = cell.vec_np
            else:
                cell.vec_p = cell.vec_p + cell.vel_vec_p * self.dt

    def ps_update(self):
            rows23 = self.third_row(self.R1)
            self.R2 = rows23[1]
            self.R3 = rows23[2]
            for i in range(len(self.cs)):
                self.cell_vel_p_update(i)
            for i in range(len(self.cs)):
                self.cell_vec_p_update(i)
    
    def ps_update_R1leader(self):
            rows23 = self.third_row(self.R1)
            self.R2 = rows23[1]
            self.R3 = rows23[2]
            for i in range(len(self.cs)):
                self.cell_vel_p_update(i)
            for i in range(len(self.cs)):
                self.cell_vec_p_update_R1leader(i)
    
    def cell_vec_np_update(self, c):
        cell = self.cs[c]
        if cell.n == self.c_hole.n or cell.n == self.c_boundary.n:
            cell.vec_np = np.array([0,0,0])
        else:
            if len(self.c_hole.ind)>3:
                if cell.n in self.R1: 
                    par = []
                    for i in range(len(cell.ind)):
                        for k in range(len(self.vs[cell.ind[i]].cells)):
                            if self.vs[cell.ind[i]].cells[k] == self.c_hole.n and cell.ind[i] not in par:
                                par.append(cell.ind[i])
                    for i in range(len(self.c_hole.ind)-1):
                        if self.c_hole.ind[i] == par[1]:
                            if self.c_hole.ind[i+1]== par[0]:
                                ri = par[0]
                                rim1 = par[1]
                            else:
                                ri = par[1]
                                rim1 = par[0]
                    lim1_i = self.vs[rim1].r - self.vs[ri].r
                    k = np.array([0,0,1])
                    m = np.cross(k, lim1_i / np.linalg.norm(lim1_i))
                    cell.vec_np = m

    def nps_update(self):
        for i in range(len(self.cs)):
            self.cell_vec_np_update(i)
    
    def cell_vec_nq_update(self,c):
        cell = self.cs[c]
        if cell.n == self.c_hole.n or cell.n == self.c_boundary.n:
            cell.vec_nq = np.array([0,0,0])
        else:
            qx = cell.vec_q[0]
            qy = cell.vec_q[1]
            modq = np.linalg.norm(cell.vec_q)
            dosphi = np.arctan2(qy,qx)
            if dosphi <0:
                dosphi = dosphi + 2*np.pi
            phi = dosphi/2
            cell.vec_nq = np.array([modq * np.cos(phi), modq * np.sin(phi), 0])

    def nqs_update(self):
        for i in range(len(self.cs)):
            self.cell_vec_nq_update(i)
    
    def cell_vel_q_update(self, c):
        cell = self.cs[c]
        if cell.n == self.c_hole.n or cell.n == self.c_boundary.n:
            cell.vel_vec_q = np.array([0,0,0]) 
        else:
            cells_vecinas = []
            for i in range(len(cell.ind)):
                for j in range(len(self.vs[cell.ind[i]].cells)):
                    if self.vs[cell.ind[i]].cells[j] not in cells_vecinas and self.vs[cell.ind[i]].cells[j]!= cell.n:
                        cells_vecinas.append(self.vs[cell.ind[i]].cells[j])
                        
            suma = np.array([0,0,0])
            for k in range(len(cells_vecinas)):
                if cells_vecinas[k] != self.c_boundary.n and cells_vecinas[k] != self.c_hole.n:
                    suma = suma + self.cs[cells_vecinas[k]].vec_q - cell.vec_q
            cell.vel_vec_q = self.ma * suma - self.md * cell.vec_q + self.mf * self.cell_vector_stress_dev(cell.n)
        
    def cell_vec_q_update(self, c):
        cell = self.cs[c]
        if cell.n == self.c_hole.n or cell.n == self.c_boundary.n:
            cell.vec_q = np.array([0,0,0])
        else:
            test = cell.vec_q + cell.vel_vec_q * self.dt
            modulo = np.linalg.norm(test)
            if modulo <=1:
                cell.vec_q = test

    def cell_vector_stress_dev(self,c):
        cell = self.cs[c]
        TijVM = self.KP * (cell.per - self.P0)
        v = cell.ind
        posiciones_todas = [ ]
        for i in range(len(v)):
            posiciones_todas.append(self.vs[int(v[i])].r)
        m_v = [0,0]
        for i in range(len(cell.ind)-1):
            vi = cell.ind[i]
            vj = cell.ind[i+1]
            vresta = self.vs[int(vj)].r - self.vs[int(vi)].r
            lij = np.linalg.norm(vresta)
            if vi in self.c_hole.ind and vj in self.c_hole.ind:
                Tijact = self.Th
            elif vi in self.c_boundary.ind and vj in self.c_boundary.ind:
                Tijact = self.Tout
            else:
                Tijact = 0

            m_v[0] = m_v[0] + 0.5*(vresta[0]**2 - vresta[1]**2) * (1/lij) * (TijVM + Tijact)/cell.area
            m_v[1] = m_v[1] + (vresta[0])*(vresta[1])* (1/lij)* (TijVM + Tijact)/cell.area

        return np.array([m_v[0], m_v[1],0])

    def qs_update(self):
        for i in range(len(self.cs)):
            self.cell_vel_q_update(i)

        for i in range(len(self.cs)):
            self.cell_vec_q_update(i)

    def create_4F(self,now):
        flats = []
        for i in range(len(self.vs)):
            if len(self.vs[i].cells)>0:
                flat = [self.vs[i].ady[0][0],self.vs[i].ady[0][1],self.vs[i].ady[1][0]]
                flats.append(flat)
            else:
                flats.append([])

        for i in range(len(self.vs)):
            for j in flats[i]:
                if j>i and j in np.array(self.vs[i].ady).flatten():
                    lij = np.linalg.norm(self.vs[i].r - self.vs[j].r)
                    if lij < self.l_T1:
                        if len(self.vs[i].cells)==3 and len(self.vs[j].cells)==3 and self.previo_T1(i,j)==1:
                            if i in self.c_hole.ind and j in self.c_hole.ind:
                                self.delete_v1_original(i,j)
                                r123=0
                                for c in self.vs[i].cells:
                                    if c in self.R1 or c in self.R2 or c in self.R3:
                                        r123=1
                                file = open('4-fold.txt', 'a')
                                pos_mean = self.vs[i].r
                                if len(self.c_hole.ind)>0:
                                    data_gap = self.pos_gap_closer(pos_mean)
                                    pos_gap = data_gap[0]
                                    d_gap = data_gap[1]
                                    phi_gap = data_gap[2]
                                    centro_gap = self.c_hole.center
                                else:
                                    pos_gap = np.array([0,0,0])
                                    centro_gap = np.array([0,0,0])
                                    d_gap = 0
                                    phi_gap = 0 
                                n1,n2,n3,n4,n5,n6, n7=round(now,3), int(i), int(j), round(pos_mean[0],3),round(pos_mean[1],3),int(r123),1
                                n8,n9,n10,n11,n12,n13= round(pos_gap[0],3), round(pos_gap[1],3),round(centro_gap[0],3), round(centro_gap[1],3), round(d_gap,3), round(phi_gap,3)
                                file.write(str(n1)+' '+str(n2)+' '+str(n3)+' '+str(n4)+' '+str(n5)+' '+str(n6)+' '+str(n7)+' '+str(n8)+' '+str(n9)+' '+str(n10)+' '+str(n11)+' '+str(n12)+' '+str(n13)+ '\n')
                    
                            elif i not in self.c_hole.ind and j not in self.c_hole.ind:
                                self.delete_v1_original(i,j)
                                r123=0
                                for c in self.vs[i].cells:
                                    if c in self.R1 or c in self.R2 or c in self.R3:
                                        r123=1
                                file = open('4-fold.txt', 'a')
                                pos_mean = self.vs[i].r
                                if len(self.c_hole.ind)>0:
                                    data_gap = self.pos_gap_closer(pos_mean)
                                    pos_gap = data_gap[0]
                                    d_gap = data_gap[1]
                                    phi_gap = data_gap[2]
                                    centro_gap = self.c_hole.center
                                else:
                                    pos_gap = np.array([0,0,0])
                                    centro_gap = np.array([0,0,0])
                                    d_gap = 0
                                    phi_gap = 0 
                                n1,n2,n3,n4,n5,n6, n7=round(now,3), int(i), int(j), round(pos_mean[0],3),round(pos_mean[1],3),int(r123), 2
                                n8,n9,n10,n11,n12,n13 = round(pos_gap[0],3), round(pos_gap[1],3),round(centro_gap[0],3), round(centro_gap[1],3), round(d_gap,3), round(phi_gap,3)
                                file.write(str(n1)+' '+str(n2)+' '+str(n3)+' '+str(n4)+' '+str(n5)+' '+str(n6)+' '+str(n7)+' '+str(n8)+' '+str(n9)+' '+str(n10)+' '+str(n11)+' '+str(n12)+' '+str(n13)+ '\n')
                    
    def attempt_solve_4F(self, now):
        n = []
        for i in range(len(self.vs)):
            if len(self.vs[i].cells)==4:
                n.append(i)
        for i in range(len(n)):
            old_1strow = self.R1
            
            vi1 = n[i]
            vi2 = self.vs[n[i]].par[0]
            eje = self.vs[vi1].eje

            r123=0
            for c in self.vs[vi1].cells:
                if c in self.R1 or c in self.R2 or c in self.R3:
                    r123=1
            
            caso_rim = 0 
            if vi1 in self.c_hole.ind:
                caso_rim = 1  
                cell_to_eliminate = -1
                cis = self.vs[vi1].cells
                for c in cis:
                    if c not in eje and c!= self.c_hole.n:
                        cell_to_eliminate = c
                new_1strow = []
                for c in self.R1:
                    if c!= cell_to_eliminate:
                        new_1strow.append(c)
            
            if vi1 in self.c_boundary.ind:
                caso_rim = 2
                    
            if caso_rim==1:
                self.R1 = new_1strow
                producto2= self.create_v1_perpendicular(vi1)

                file = open('T1_perpendicular.txt', 'a')
                self.vs[vi1].par = [-1]
                self.vs[vi1].eje = [-1,-1]
                pos_mean = 0.5*(self.vs[vi1].r+self.vs[vi2].r)

                if len(self.c_hole.ind)>0:
                    data_gap = self.pos_gap_closer(pos_mean)
                    pos_gap = data_gap[0]
                    d_gap = data_gap[1]
                    phi_gap = data_gap[2]
                    centro_gap = self.c_hole.center
                else:
                    pos_gap = np.array([0,0,0])
                    centro_gap = np.array([0,0,0])
                    d_gap = 0
                    phi_gap = 0
                n1,n2,n3,n4,n5,n6,n7=round(now,3), int(vi1), int(vi2), round(pos_mean[0],3), round(pos_mean[1],3), int(caso_rim), int(r123)
                n8,n9,n10,n11,n12,n13 = round(pos_gap[0],3), round(pos_gap[1],3),round(centro_gap[0],3), round(centro_gap[1],3),round(d_gap,3),round(phi_gap,3)
                file.write(str(n1)+' '+str(n2)+' '+str(n3)+' '+str(n4)+' '+str(n5)+' '+str(n6)+' '+str(n7)+' '+str(n8)+' '+str(n9)+' '+str(n10)+' '+str(n11)+' '+str(n12)+' '+str(n13)+ '\n')

            elif caso_rim==2:
                producto2= self.create_v1_perpendicular(vi1)
                file = open('T1_perpendicular.txt', 'a')
                self.vs[vi1].par = [-1]
                self.vs[vi1].eje = [-1,-1]
                pos_mean = 0.5*(self.vs[vi1].r+self.vs[vi2].r)

                if len(self.c_hole.ind)>0:
                    data_gap = self.pos_gap_closer(pos_mean)
                    pos_gap = data_gap[0]
                    d_gap = data_gap[1]
                    phi_gap = data_gap[2]
                    centro_gap = self.c_hole.center
                else:
                    pos_gap = np.array([0,0,0])
                    centro_gap = np.array([0,0,0])
                    d_gap = 0
                    phi_gap = 0
                n1,n2,n3,n4,n5,n6,n7=round(now,3), int(vi1), int(vi2), round(pos_mean[0],3), round(pos_mean[1],3), int(caso_rim), int(r123)
                n8,n9,n10,n11,n12,n13 = round(pos_gap[0],3), round(pos_gap[1],3),round(centro_gap[0],3), round(centro_gap[1],3),round(d_gap,3),round(phi_gap,3)
                file.write(str(n1)+' '+str(n2)+' '+str(n3)+' '+str(n4)+' '+str(n5)+' '+str(n6)+' '+str(n7)+' '+str(n8)+' '+str(n9)+' '+str(n10)+' '+str(n11)+' '+str(n12)+' '+str(n13)+ '\n')


            else:
                producto1= self.create_v1_original(vi1)
                self.delete_v1_original(vi1,vi2)
                producto2= self.create_v1_perpendicular(vi1)

                if producto1<0 and producto2<0:
                    self.delete_v1_perpendicular(vi1,vi2)
                    self.vs[vi1].eje = eje
                else:
                    if producto1>producto2:
                        self.delete_v1_perpendicular(vi1,vi2)
                        self.vs[vi1].eje = eje
                        self.create_v1_original(vi1)
                        self.vs[vi1].par = [-1]
                        self.vs[vi1].eje = [-1,-1]
                        file = open('T1_original.txt', 'a')
                        pos_mean = 0.5*(self.vs[vi1].r+self.vs[vi2].r)
                        if len(self.c_hole.ind)>0:
                            data_gap = self.pos_gap_closer(pos_mean)
                            pos_gap = data_gap[0]
                            d_gap = data_gap[1]
                            phi_gap = data_gap[2]
                            centro_gap = self.c_hole.center
                        else:
                            pos_gap = np.array([0,0,0])
                            centro_gap = np.array([0,0,0])
                            d_gap = 0
                            phi_gap = 0
                        n1,n2,n3,n4,n5,n6,n7=round(now,3), int(vi1), int(vi2), round(pos_mean[0],3), round(pos_mean[1],3), int(caso_rim), int(r123)
                        n8,n9,n10,n11,n12,n13 = round(pos_gap[0],3), round(pos_gap[1],3),round(centro_gap[0],3), round(centro_gap[1],3), round(d_gap,3),round(phi_gap,3)
                        file.write(str(n1)+' '+str(n2)+' '+str(n3)+' '+str(n4)+' '+str(n5)+' '+str(n6)+' '+str(n7)+' '+str(n8)+' '+str(n9)+' '+str(n10)+' '+str(n11)+' '+str(n12)+' '+str(n13)+ '\n')
                    else:
                        file = open('T1_perpendicular.txt', 'a')
                        self.vs[vi1].par = [-1]
                        self.vs[vi1].eje = [-1,-1]
                        pos_mean = 0.5*(self.vs[vi1].r+self.vs[vi2].r)

                        if len(self.c_hole.ind)>0:
                            data_gap = self.pos_gap_closer(pos_mean)
                            pos_gap = data_gap[0]
                            d_gap = data_gap[1]
                            phi_gap = data_gap[2]
                            centro_gap = self.c_hole.center
                        else:
                            pos_gap = np.array([0,0,0])
                            centro_gap = np.array([0,0,0])
                            d_gap = 0
                            phi_gap = 0
                        n1,n2,n3,n4,n5,n6,n7=round(now,3), int(vi1), int(vi2), round(pos_mean[0],3), round(pos_mean[1],3), int(caso_rim), int(r123)
                        n8,n9,n10,n11,n12,n13 = round(pos_gap[0],3), round(pos_gap[1],3),round(centro_gap[0],3), round(centro_gap[1],3),round(d_gap,3),round(phi_gap,3)
                        file.write(str(n1)+' '+str(n2)+' '+str(n3)+' '+str(n4)+' '+str(n5)+' '+str(n6)+' '+str(n7)+' '+str(n8)+' '+str(n9)+' '+str(n10)+' '+str(n11)+' '+str(n12)+' '+str(n13)+ '\n')
    
    def attempt_solve_4F_R1leader(self, now):
        n = []
        for i in range(len(self.vs)):
            if len(self.vs[i].cells)==4:
                n.append(i)
        for i in range(len(n)):
            old_1strow = self.R1
            
            vi1 = n[i]
            vi2 = self.vs[n[i]].par[0]
            eje = self.vs[vi1].eje

            r123=0
            for c in self.vs[vi1].cells:
                if c in self.R1 or c in self.R2 or c in self.R3:
                    r123=1
            
            caso_rim = 0 
            if vi1 in self.c_hole.ind:
                caso_rim = 1  
                cell_to_eliminate = -1
                cis = self.vs[vi1].cells
                for c in cis:
                    if c not in eje and c!= self.c_hole.n:
                        cell_to_eliminate = c
                new_1strow = []
                for c in self.R1:
                    if c!= cell_to_eliminate:
                        new_1strow.append(c)
            
            if vi1 in self.c_boundary.ind:
                caso_rim = 2
                    
            if caso_rim==1:
                self.R1 = new_1strow
                producto2= self.create_v1_perpendicular_R1leader(vi1)

                file = open('T1_perpendicular.txt', 'a')
                self.vs[vi1].par = [-1]
                self.vs[vi1].eje = [-1,-1]
                pos_mean = 0.5*(self.vs[vi1].r+self.vs[vi2].r)

                if len(self.c_hole.ind)>0:
                    data_gap = self.pos_gap_closer(pos_mean)
                    pos_gap = data_gap[0]
                    d_gap = data_gap[1]
                    phi_gap = data_gap[2]
                    centro_gap = self.c_hole.center
                else:
                    pos_gap = np.array([0,0,0])
                    centro_gap = np.array([0,0,0])
                    d_gap = 0
                    phi_gap = 0
                n1,n2,n3,n4,n5,n6,n7=round(now,3), int(vi1), int(vi2), round(pos_mean[0],3), round(pos_mean[1],3), int(caso_rim), int(r123)
                n8,n9,n10,n11,n12,n13 = round(pos_gap[0],3), round(pos_gap[1],3),round(centro_gap[0],3), round(centro_gap[1],3),round(d_gap,3),round(phi_gap,3)
                file.write(str(n1)+' '+str(n2)+' '+str(n3)+' '+str(n4)+' '+str(n5)+' '+str(n6)+' '+str(n7)+' '+str(n8)+' '+str(n9)+' '+str(n10)+' '+str(n11)+' '+str(n12)+' '+str(n13)+ '\n')

            elif caso_rim==2:
                producto2= self.create_v1_perpendicular_R1leader(vi1)
                file = open('T1_perpendicular.txt', 'a')
                self.vs[vi1].par = [-1]
                self.vs[vi1].eje = [-1,-1]
                pos_mean = 0.5*(self.vs[vi1].r+self.vs[vi2].r)

                if len(self.c_hole.ind)>0:
                    data_gap = self.pos_gap_closer(pos_mean)
                    pos_gap = data_gap[0]
                    d_gap = data_gap[1]
                    phi_gap = data_gap[2]
                    centro_gap = self.c_hole.center
                else:
                    pos_gap = np.array([0,0,0])
                    centro_gap = np.array([0,0,0])
                    d_gap = 0
                    phi_gap = 0
                n1,n2,n3,n4,n5,n6,n7=round(now,3), int(vi1), int(vi2), round(pos_mean[0],3), round(pos_mean[1],3), int(caso_rim), int(r123)
                n8,n9,n10,n11,n12,n13 = round(pos_gap[0],3), round(pos_gap[1],3),round(centro_gap[0],3), round(centro_gap[1],3),round(d_gap,3),round(phi_gap,3)
                file.write(str(n1)+' '+str(n2)+' '+str(n3)+' '+str(n4)+' '+str(n5)+' '+str(n6)+' '+str(n7)+' '+str(n8)+' '+str(n9)+' '+str(n10)+' '+str(n11)+' '+str(n12)+' '+str(n13)+ '\n')


            else:
                producto1= self.create_v1_original_R1leader(vi1)
                self.delete_v1_original(vi1,vi2)
                producto2= self.create_v1_perpendicular_R1leader(vi1)

                if producto1<0 and producto2<0:
                    self.delete_v1_perpendicular(vi1,vi2)
                    self.vs[vi1].eje = eje
                else:
                    if producto1>producto2:
                        self.delete_v1_perpendicular(vi1,vi2)
                        self.vs[vi1].eje = eje
                        self.create_v1_original_R1leader(vi1)
                        self.vs[vi1].par = [-1]
                        self.vs[vi1].eje = [-1,-1]
                        file = open('T1_original.txt', 'a')
                        pos_mean = 0.5*(self.vs[vi1].r+self.vs[vi2].r)
                        if len(self.c_hole.ind)>0:
                            data_gap = self.pos_gap_closer(pos_mean)
                            pos_gap = data_gap[0]
                            d_gap = data_gap[1]
                            phi_gap = data_gap[2]
                            centro_gap = self.c_hole.center
                        else:
                            pos_gap = np.array([0,0,0])
                            centro_gap = np.array([0,0,0])
                            d_gap = 0
                            phi_gap = 0
                        n1,n2,n3,n4,n5,n6,n7=round(now,3), int(vi1), int(vi2), round(pos_mean[0],3), round(pos_mean[1],3), int(caso_rim), int(r123)
                        n8,n9,n10,n11,n12,n13 = round(pos_gap[0],3), round(pos_gap[1],3),round(centro_gap[0],3), round(centro_gap[1],3), round(d_gap,3),round(phi_gap,3)
                        file.write(str(n1)+' '+str(n2)+' '+str(n3)+' '+str(n4)+' '+str(n5)+' '+str(n6)+' '+str(n7)+' '+str(n8)+' '+str(n9)+' '+str(n10)+' '+str(n11)+' '+str(n12)+' '+str(n13)+ '\n')
                    else:
                        file = open('T1_perpendicular.txt', 'a')
                        self.vs[vi1].par = [-1]
                        self.vs[vi1].eje = [-1,-1]
                        pos_mean = 0.5*(self.vs[vi1].r+self.vs[vi2].r)

                        if len(self.c_hole.ind)>0:
                            data_gap = self.pos_gap_closer(pos_mean)
                            pos_gap = data_gap[0]
                            d_gap = data_gap[1]
                            phi_gap = data_gap[2]
                            centro_gap = self.c_hole.center
                        else:
                            pos_gap = np.array([0,0,0])
                            centro_gap = np.array([0,0,0])
                            d_gap = 0
                            phi_gap = 0
                        n1,n2,n3,n4,n5,n6,n7=round(now,3), int(vi1), int(vi2), round(pos_mean[0],3), round(pos_mean[1],3), int(caso_rim), int(r123)
                        n8,n9,n10,n11,n12,n13 = round(pos_gap[0],3), round(pos_gap[1],3),round(centro_gap[0],3), round(centro_gap[1],3),round(d_gap,3),round(phi_gap,3)
                        file.write(str(n1)+' '+str(n2)+' '+str(n3)+' '+str(n4)+' '+str(n5)+' '+str(n6)+' '+str(n7)+' '+str(n8)+' '+str(n9)+' '+str(n10)+' '+str(n11)+' '+str(n12)+' '+str(n13)+ '\n')

    def pos_gap_closer(self, pos_mean):
        vertex_0 = self.c_hole.ind[0]
        rgap = self.vs[vertex_0].r
        vec_0 = pos_mean - rgap
        l0 = np.linalg.norm(vec_0)

        for i in range(len(self.c_hole.ind)-1):
            vertex_i = self.c_hole.ind[i]
            vec_ij = pos_mean - self.vs[vertex_i].r
            l_ij = np.linalg.norm(vec_ij)
            if l_ij< l0:
                l0 = l_ij
                rgap = self.vs[vertex_i].r

        relative_position = pos_mean - self.c_hole.center
        theta = np.arctan2(relative_position[1], relative_position[0])
        theta_degrees = np.degrees(theta)
        if theta_degrees < 0:
            theta_degrees += 360

        
        return [rgap,l0,theta_degrees]

    def cal_polygon(self, v):
        id_vs = self.cs[v].ind
        vectors_PBC = []
        for i in range(len(id_vs)):
            vi = self.vs[id_vs[i]].r
            vectors_PBC.append(vi)
        self.cs[v].pol = Polygon(vectors_PBC)
        
    def cal_polygons(self):
        for i in range(len(self.cs)):
            self.cal_polygon(i)
            self.cs[i].cal_centro(self.dt)
            self.cs[i].cal_area()
            self.cs[i].cal_per()

    def cal_polygons_inicial(self):
        for i in range(len(self.cs)):
            self.cal_polygon(i)
            self.cs[i].cal_centro_inicial()
            self.cs[i].cal_area()
            self.cs[i].cal_per()

    def evol_vertex(self,now):
        self.create_4F(now)
        self.attempt_solve_4F(now)
        self.nps_update()
        self.ps_update()
        self.qs_update()
        self.nqs_update()
        self.pos_nuevas_vertex()
        self.cal_polygons()
    
    def evol_vertex_R1leader(self,now):
        self.create_4F(now)
        self.attempt_solve_4F_R1leader(now)
        self.nps_update()
        self.ps_update_R1leader()
        self.qs_update()
        self.nqs_update()
        self.pos_nuevas_vertex_R1leader()
        self.cal_polygons()

    def vec_v_VM(self,v):
        vec = self.vs[v]
        velocity = np.array([0, 0 , 0])
        for j in range(len(vec.cells)):
            c = vec.cells[j]
            vi = int(vec.ady[j][0])
            vd = int(vec.ady[j][1])
            a = self.vs[vi].r - vec.r
            b = self.vs[vd].r - vec.r
            l_ii = np.linalg.norm(a)
            l_id = np.linalg.norm(b)
            if c == self.c_hole.n or c == self.c_boundary.n:
                if c == self.c_hole.n:
                    PER = self.Th
                else:
                    PER = self.Tout
            else:
                PER = self.KP * (self.cs[c].per - self.P0)
            velocity =  velocity +  a * (PER/l_ii) +  b * (PER/l_id)
            rest = self.vs[vd].r - self.vs[vi].r
            k = np.array([0, 0 , 1])
            v_tot = np.cross(rest,k)
            if c == self.c_hole.n or c == self.c_boundary.n:
                AREA = 0
            else:
                AREA = 0.5 * self.KA * (self.cs[c].area - self.A0)
            velocity = velocity +  AREA * v_tot

        return velocity

    def vec_v_crawl(self, v):
        vec = self.vs[v]
        velocity = np.array([0, 0 , 0])
        todos_p = np.array([0, 0 , 0])
        cantidad = 0
        for c in vec.cells:
            if c not in self.R1 and c!= self.c_boundary.n and c!= self.c_hole.n:
                todos_p = todos_p + self.cs[c].vec_p
                cantidad = cantidad + 1
        mean_p = todos_p / cantidad
        velocity = velocity +  mean_p * self.v0
        return velocity
    
    def vec_v_crawl_R1leader(self, v):
        vec = self.vs[v]
        velocity = np.array([0, 0 , 0])
        todos_p = np.array([0, 0 , 0])
        cantidad = 0
        for c in vec.cells:
            if c!= self.c_boundary.n and c!= self.c_hole.n:
                todos_p = todos_p + self.cs[c].vec_p
                cantidad = cantidad + 1
        mean_p = todos_p / cantidad
        velocity = velocity +  mean_p * self.v0
        return velocity

    def vec_v_AS(self,v):
        vec = self.vs[v]
        velocity = np.array([0, 0 , 0])
        for j in range(len(vec.cells)):
            vi1 = int(vec.ady[j][0])
            vd1 = int(vec.ady[j][1])
            a1 = self.vs[vi1].r - vec.r
            b1 = self.vs[vd1].r - vec.r
            c = vec.cells[j]
            k = np.array([0, 0 , 1])
            fza_AS = np.array([0, 0 , 0])
            p_aeff = self.Sigma0 * np.linalg.norm(self.cs[c].vec_nq)
            modulo_n = np.linalg.norm(self.cs[c].vec_nq)
            if modulo_n==0:
                nc = np.array([0, 0 , 0])
            else:
                nc = self.cs[c].vec_nq / modulo_n
            mc = np.cross(nc,k)
            
            eta_c = (nc[0]**2 - mc[0]**2, nc[0]*nc[1]-mc[0]*mc[1],0)
            gamma_c = (nc[0]*nc[1]-mc[0]*mc[1], nc[1]**2-mc[1]**2,0)
            fza_AS_x = -1*(p_aeff*np.dot(eta_c,a1) + p_aeff*np.dot(eta_c,b1))
            fza_AS_y = -1*(p_aeff*np.dot(gamma_c,a1) +p_aeff*np.dot(gamma_c,b1))
            fza_AS = np.array([fza_AS_x,fza_AS_y,0])
            velocity = velocity + fza_AS
        return velocity

    def pos_nuevas_vertex(self):
        velocidades_VM = []
        velocidades_crawl = []
        velocidades_AS = []
        for n in range(len(self.vs)):
            if len(self.vs[n].cells)>0:
                velocidades_VM.append(self.vec_v_VM(n))
                velocidades_AS.append(self.vec_v_AS(n))
                if n in self.c_hole.ind:
                    velocidades_crawl.append(np.array([0,0,0]))
                else:
                    velocidades_crawl.append(self.vec_v_crawl(n))
            else:
                velocidades_VM.append(np.array([0,0,0]))
                velocidades_AS.append(np.array([0,0,0]))
                velocidades_crawl.append(np.array([0,0,0]))

        for n in range(len(self.vs)):
            self.vs[n].vel = velocidades_VM[n]+velocidades_crawl[n]+velocidades_AS[n]
            new_position = self.vs[n].r + (velocidades_VM[n]+velocidades_crawl[n]+velocidades_AS[n]) * self.dt
            self.vs[n].r = new_position
        
        for n in range(len(self.vs)):
            if len(self.vs[n].cells) == 0:
                id_par = self.vs[n].par[0]
                pos_par = self.vs[id_par].r
                self.vs[n].r = pos_par

    def pos_nuevas_vertex_R1leader(self):
        velocidades_VM = []
        velocidades_crawl = []
        velocidades_AS = []
        for n in range(len(self.vs)):
            if len(self.vs[n].cells)>0:
                velocidades_VM.append(self.vec_v_VM(n))
                velocidades_AS.append(self.vec_v_AS(n))
                velocidades_crawl.append(self.vec_v_crawl_R1leader(n))
            else:
                velocidades_VM.append(np.array([0,0,0]))
                velocidades_AS.append(np.array([0,0,0]))
                velocidades_crawl.append(np.array([0,0,0]))

        for n in range(len(self.vs)):
            self.vs[n].vel = velocidades_VM[n]+velocidades_crawl[n]+velocidades_AS[n]
            new_position = self.vs[n].r + (velocidades_VM[n]+velocidades_crawl[n]+velocidades_AS[n]) * self.dt
            self.vs[n].r = new_position
        
        for n in range(len(self.vs)):
            if len(self.vs[n].cells) == 0:
                id_par = self.vs[n].par[0]
                pos_par = self.vs[id_par].r
                self.vs[n].r = pos_par
                  
    def data(self):
        new01 = [] #vertex ID
        pos_x = [] #vertex position, component x
        pos_y = [] #vertex position, component y
        vels_vs_x = [] #vertex velocity, component x
        vels_vs_y = [] #vertex velocity, component y
        pair = [] #merged 3fold-vertex ID
        for i in range(len(self.vs)):
            new01.append(i)
            pos_x.append(self.vs[i].r[0])
            pos_y.append(self.vs[i].r[1])
            vels_vs_x.append(self.vs[i].vel[0])
            vels_vs_y.append(self.vs[i].vel[1])
            pair.append(self.vs[i].par[0])

        new02 = [] #cell ID
        A = [] #cell area
        P = [] #cell perimeter
        Cx = [] #cell center, component x
        Cy = [] #cell center, component y
        Nq_cs_x = [] #cell nematic, component x
        Nq_cs_y = [] #cell nematic, component y
        vels_cs_x = [] #cell velocity, component x
        vels_cs_y = [] #cell velocity, component y

        #network topology
        celda0 = [] #cells
        celda1 = [] #vertices

        for i in range(len(self.cs)):
            new02.append(i)
            A.append(self.cs[i].area)
            P.append(self.cs[i].per)
            Cx.append(self.cs[i].center[0])
            Cy.append(self.cs[i].center[1])
            Nq_cs_x.append(self.cs[i].vec_nq[0])
            Nq_cs_y.append(self.cs[i].vec_nq[1])
            vels_cs_x.append(self.cs[i].vel[0])
            vels_cs_y.append(self.cs[i].vel[1])
            for k in range(len(self.cs[i].ind)-1):
                celda0.append(i)
                celda1.append(self.cs[i].ind[k])

        np.savetxt('./results/data_vertices.txt',  np.c_[new01, pos_x, pos_y, vels_vs_x, vels_vs_y, pair], fmt='%1.10f')
        np.savetxt('./results/data_celulas.txt',  np.c_[new02, A, P, Cx, Cy, Nq_cs_x, Nq_cs_y, vels_cs_x, vels_cs_y], fmt='%1.10f')
        np.savetxt('./results/data_celda.txt',  np.c_[celda0, celda1], fmt='%1.0f')
        np.savetxt('./first_row/data_1st.txt',  np.c_[self.R1], fmt='%1.0f')
        np.savetxt('./second_row/data_2nd.txt',  np.c_[self.R2], fmt='%1.0f')
        np.savetxt('./third_row/data_3rd.txt',  np.c_[self.R3], fmt='%1.0f')
       
    def create_v1_perpendicular(self,vi1):
        cells_i1 = self.vs[vi1].cells #A,B,C,B
        cellA = cells_i1[0]
        cellB = cells_i1[1]
        cellC = cells_i1[2]
        cellD = cells_i1[3]
        vert_A = self.cs[cellA].ind
        vert_B = self.cs[cellB].ind
        vert_C = self.cs[cellC].ind
        vert_D = self.cs[cellD].ind

        for i in vert_A:
            if i in vert_B and i not in vert_C and i not in vert_D:
                vert2 = i
        for i in vert_A:
            if i in vert_D and i not in vert_B and i not in vert_C:
                vert3 = i
        for i in vert_C:
            if i in vert_B and i not in vert_A and i not in vert_D:
                vert6 = i
        for i in vert_C:
            if i in vert_D and i not in vert_A and i not in vert_B:
                vert5 = i
        vert4 = vi1
        vi2 = self.vs[vi1].par[0]
        vert1 = self.vs[vi1].par[0]

        c1y2 = []
        for c in [cellA,cellB,cellC,cellD]:
            if c not in self.vs[vert4].eje:
                c1y2.append(c)
        c1 = c1y2[0]
        c2 = c1y2[1]
        r_T1 = self.cs[c1].center - self.cs[c2].center
        l_r_T1 = np.linalg.norm(r_T1)
        rnorm_T1 = (1/l_r_T1) * r_T1

        #MODIFY ADY
        for i in range(len(self.vs[vert3].ady)):
            for j in range(len(self.vs[vert3].ady[i])):
                if self.vs[vert3].ady[i][j] == vert4:
                    self.vs[vert3].ady[i][j] = vert1
        for i in range(len(self.vs[vert5].ady)):
            for j in range(len(self.vs[vert5].ady[i])):
                if self.vs[vert5].ady[i][j] == vert4:
                    self.vs[vert5].ady[i][j] = vert1

        self.vs[vert4].cells = [cellA, cellB, cellC]
        self.vs[vert4].ady = [[vert2,vert1],[vert6,vert2],[vert1,vert6]]
        self.vs[vert1].cells = [cellC, cellD, cellA]
        self.vs[vert1].ady = [[vert5,vert4],[vert3,vert5],[vert4,vert3]]

        #MODIFY IND cell A
        add_A = []
        q = 0
        for i in range(len(self.cs[cellA].ind)-1):
            if self.cs[cellA].ind[i] == vert4 and q==0:
                add_A.append(vert4)
                add_A.append(vert1)
                q=1
            else:
                add_A.append(self.cs[cellA].ind[i])
        self.cs[cellA].ind = add_A
        if self.cs[cellA].ind[0] != self.cs[cellA].ind[len(self.cs[cellA].ind)-1]:
            self.cs[cellA].ind.append(self.cs[cellA].ind[0])

        #MODIFY IND cell C
        add_C = []
        q = 0
        for i in range(len(self.cs[cellC].ind)-1):
            if self.cs[cellC].ind[i] == vert5 and q==0:
                add_C.append(vert5)
                add_C.append(vert1)
                q=1
            else:
                add_C.append(self.cs[cellC].ind[i])
        self.cs[cellC].ind = add_C
        if self.cs[cellC].ind[0] != self.cs[cellC].ind[len(self.cs[cellC].ind)-1]:
            self.cs[cellC].ind.append(self.cs[cellC].ind[0])

        #MODIFY IND cell D
        add_D = []
        q = 0
        for i in range(len(self.cs[cellD].ind)-1):
            if self.cs[cellD].ind[i] == vert4 and q==0:
                add_D.append(vert1)
                q=1
            else:
                add_D.append(self.cs[cellD].ind[i])
        self.cs[cellD].ind = add_D
        if self.cs[cellD].ind[0] != self.cs[cellD].ind[len(self.cs[cellD].ind)-1]:
            self.cs[cellD].ind.append(self.cs[cellD].ind[0])


        self.vs[vert4].eje = [-1,-1]
        self.vs[vert4].par[0] = -1

        pos_vi1p = self.vs[vert4].r + (3/4)*self.l_T1 * rnorm_T1
        pos_vi2p = self.vs[vert4].r - (3/4)*self.l_T1 * rnorm_T1

        self.vs[vi1].r = pos_vi1p
        self.vs[vi2].r = pos_vi2p

        for c in [cellA,cellB,cellC,cellD]:
            self.cal_polygon(c)
            self.cs[c].cal_area()
            self.cs[c].cal_per()
            self.cs[c].cal_centro(self.dt)

        if vert4 in self.c_hole.ind:
            vel_i = self.vec_v_VM(vert4) + self.vec_v_AS(vert4)
        else:
            vel_i = self.vec_v_VM(vert4) + self.vec_v_AS(vert4)+self.vec_v_crawl(vert4)
        
        if vert1 in self.c_hole.ind:
            vel_j = self.vec_v_VM(vert1) + self.vec_v_AS(vert1)
        else:
            vel_j = self.vec_v_VM(vert1) + self.vec_v_AS(vert1)+self.vec_v_crawl(vert1)

        producto = np.dot((self.vs[vert4].r-self.vs[vert1].r),(vel_i -vel_j))
        return producto
    
    def create_v1_perpendicular_R1leader(self,vi1):
        cells_i1 = self.vs[vi1].cells #A,B,C,B
        cellA = cells_i1[0]
        cellB = cells_i1[1]
        cellC = cells_i1[2]
        cellD = cells_i1[3]
        vert_A = self.cs[cellA].ind
        vert_B = self.cs[cellB].ind
        vert_C = self.cs[cellC].ind
        vert_D = self.cs[cellD].ind

        for i in vert_A:
            if i in vert_B and i not in vert_C and i not in vert_D:
                vert2 = i
        for i in vert_A:
            if i in vert_D and i not in vert_B and i not in vert_C:
                vert3 = i
        for i in vert_C:
            if i in vert_B and i not in vert_A and i not in vert_D:
                vert6 = i
        for i in vert_C:
            if i in vert_D and i not in vert_A and i not in vert_B:
                vert5 = i
        vert4 = vi1
        vi2 = self.vs[vi1].par[0]
        vert1 = self.vs[vi1].par[0]

        c1y2 = []
        for c in [cellA,cellB,cellC,cellD]:
            if c not in self.vs[vert4].eje:
                c1y2.append(c)
        c1 = c1y2[0]
        c2 = c1y2[1]
        r_T1 = self.cs[c1].center - self.cs[c2].center
        l_r_T1 = np.linalg.norm(r_T1)
        rnorm_T1 = (1/l_r_T1) * r_T1

        #MODIFY ADY
        for i in range(len(self.vs[vert3].ady)):
            for j in range(len(self.vs[vert3].ady[i])):
                if self.vs[vert3].ady[i][j] == vert4:
                    self.vs[vert3].ady[i][j] = vert1
        for i in range(len(self.vs[vert5].ady)):
            for j in range(len(self.vs[vert5].ady[i])):
                if self.vs[vert5].ady[i][j] == vert4:
                    self.vs[vert5].ady[i][j] = vert1

        self.vs[vert4].cells = [cellA, cellB, cellC]
        self.vs[vert4].ady = [[vert2,vert1],[vert6,vert2],[vert1,vert6]]
        self.vs[vert1].cells = [cellC, cellD, cellA]
        self.vs[vert1].ady = [[vert5,vert4],[vert3,vert5],[vert4,vert3]]

        #MODIFY IND cell A
        add_A = []
        q = 0
        for i in range(len(self.cs[cellA].ind)-1):
            if self.cs[cellA].ind[i] == vert4 and q==0:
                add_A.append(vert4)
                add_A.append(vert1)
                q=1
            else:
                add_A.append(self.cs[cellA].ind[i])
        self.cs[cellA].ind = add_A
        if self.cs[cellA].ind[0] != self.cs[cellA].ind[len(self.cs[cellA].ind)-1]:
            self.cs[cellA].ind.append(self.cs[cellA].ind[0])

        #MODIFY IND cell C
        add_C = []
        q = 0
        for i in range(len(self.cs[cellC].ind)-1):
            if self.cs[cellC].ind[i] == vert5 and q==0:
                add_C.append(vert5)
                add_C.append(vert1)
                q=1
            else:
                add_C.append(self.cs[cellC].ind[i])
        self.cs[cellC].ind = add_C
        if self.cs[cellC].ind[0] != self.cs[cellC].ind[len(self.cs[cellC].ind)-1]:
            self.cs[cellC].ind.append(self.cs[cellC].ind[0])

        #MODIFY IND cell D
        add_D = []
        q = 0
        for i in range(len(self.cs[cellD].ind)-1):
            if self.cs[cellD].ind[i] == vert4 and q==0:
                add_D.append(vert1)
                q=1
            else:
                add_D.append(self.cs[cellD].ind[i])
        self.cs[cellD].ind = add_D
        if self.cs[cellD].ind[0] != self.cs[cellD].ind[len(self.cs[cellD].ind)-1]:
            self.cs[cellD].ind.append(self.cs[cellD].ind[0])


        self.vs[vert4].eje = [-1,-1]
        self.vs[vert4].par[0] = -1

        pos_vi1p = self.vs[vert4].r + (3/4)*self.l_T1 * rnorm_T1
        pos_vi2p = self.vs[vert4].r - (3/4)*self.l_T1 * rnorm_T1

        self.vs[vi1].r = pos_vi1p
        self.vs[vi2].r = pos_vi2p

        for c in [cellA,cellB,cellC,cellD]:
            self.cal_polygon(c)
            self.cs[c].cal_area()
            self.cs[c].cal_per()
            self.cs[c].cal_centro(self.dt)

        vel_i = self.vec_v_VM(vert4) + self.vec_v_AS(vert4)+self.vec_v_crawl_R1leader(vert4)
        vel_j = self.vec_v_VM(vert1) + self.vec_v_AS(vert1)+self.vec_v_crawl_R1leader(vert1)

        producto = np.dot((self.vs[vert4].r-self.vs[vert1].r),(vel_i -vel_j))
        return producto

    def create_v1_original(self,vi1):
        cells_i1 = self.vs[vi1].cells #A,B,C,B
        cellA = cells_i1[0]
        cellB = cells_i1[1]
        cellC = cells_i1[2]
        cellD = cells_i1[3]
        vert_A = self.cs[cellA].ind
        vert_B = self.cs[cellB].ind
        vert_C = self.cs[cellC].ind
        vert_D = self.cs[cellD].ind

        for i in vert_A:
            if i in vert_B and i not in vert_C and i not in vert_D:
                vert2 = i
        for i in vert_A:
            if i in vert_D and i not in vert_B and i not in vert_C:
                vert3 = i
        for i in vert_C:
            if i in vert_B and i not in vert_A and i not in vert_D:
                vert6 = i
        for i in vert_C:
            if i in vert_D and i not in vert_A and i not in vert_B:
                vert5 = i
        vert4 = vi1
        vi2 = self.vs[vi1].par[0]
        vert1 = self.vs[vi1].par[0]

        c1 = self.vs[vert4].eje[0]
        c2 = self.vs[vert4].eje[1]
        r_T1 = self.cs[c1].center - self.cs[c2].center
        l_r_T1 = np.linalg.norm(r_T1)
        rnorm_T1 = (1/l_r_T1) * r_T1

        #MODIFY ADY
        for i in range(len(self.vs[vert6].ady)):
            for j in range(len(self.vs[vert6].ady[i])):
                if self.vs[vert6].ady[i][j] == vert4:
                    self.vs[vert6].ady[i][j] = vert1
        for i in range(len(self.vs[vert5].ady)):
            for j in range(len(self.vs[vert5].ady[i])):
                if self.vs[vert5].ady[i][j] == vert4:
                    self.vs[vert5].ady[i][j] = vert1

        self.vs[vert4].cells = [cellA, cellB, cellD]
        self.vs[vert4].ady = [[vert2,vert3],[vert1,vert2],[vert3,vert1]]
        self.vs[vert1].cells = [cellC, cellD, cellB]
        self.vs[vert1].ady = [[vert5,vert6],[vert4,vert5],[vert6,vert4]]

        #MODIFY IND cell B
        add_B = []
        q = 0
        for i in range(len(self.cs[cellB].ind)-1):
            if self.cs[cellB].ind[i] == vert6 and q==0:
                add_B.append(vert6)
                add_B.append(vert1)
                q=1
            else:
                add_B.append(self.cs[cellB].ind[i])
        self.cs[cellB].ind = add_B
        if self.cs[cellB].ind[0] != self.cs[cellB].ind[len(self.cs[cellB].ind)-1]:
            self.cs[cellB].ind.append(self.cs[cellB].ind[0])

        #MODIFY IND cell C
        add_C = []
        q = 0
        for i in range(len(self.cs[cellC].ind)-1):
            if self.cs[cellC].ind[i] == vi1 and q==0:
                add_C.append(vert1)
                q=1
            else:
                add_C.append(self.cs[cellC].ind[i])
        self.cs[cellC].ind = add_C
        if self.cs[cellC].ind[0] != self.cs[cellC].ind[len(self.cs[cellC].ind)-1]:
            self.cs[cellC].ind.append(self.cs[cellC].ind[0])

        #MODIFY IND cell D
        add_D = []
        q = 0
        for i in range(len(self.cs[cellD].ind)-1):
            if self.cs[cellD].ind[i] == vi1 and q==0:
                add_D.append(vi1)
                add_D.append(vert1)
                q=1
            else:
                add_D.append(self.cs[cellD].ind[i])
        self.cs[cellD].ind = add_D
        if self.cs[cellD].ind[0] != self.cs[cellD].ind[len(self.cs[cellD].ind)-1]:
            self.cs[cellD].ind.append(self.cs[cellD].ind[0])


        self.vs[vert4].eje = [-1,-1]
        self.vs[vert4].par[0] = -1

        pos_vi1p = self.vs[vert4].r + (3/4)*self.l_T1 * rnorm_T1
        pos_vi2p = self.vs[vert4].r - (3/4)*self.l_T1 * rnorm_T1

        self.vs[vi1].r = pos_vi1p
        self.vs[vi2].r = pos_vi2p

        for c in [cellA,cellB,cellC,cellD]:
            self.cal_polygon(c)
            self.cs[c].cal_area()
            self.cs[c].cal_per()
            self.cs[c].cal_centro(self.dt)

        if vert4 in self.c_hole.ind:
            vel_i = self.vec_v_VM(vert4) + self.vec_v_AS(vert4)
        else:
            vel_i = self.vec_v_VM(vert4) + self.vec_v_AS(vert4)+self.vec_v_crawl(vert4)
        
        if vert1 in self.c_hole.ind:
            vel_j = self.vec_v_VM(vert1) + self.vec_v_AS(vert1)
        else:
            vel_j = self.vec_v_VM(vert1) + self.vec_v_AS(vert1)+self.vec_v_crawl(vert1)

        producto = np.dot((self.vs[vert4].r-self.vs[vert1].r), (vel_i -vel_j))
        return producto
 
    def create_v1_original_R1leader(self,vi1):
        cells_i1 = self.vs[vi1].cells #A,B,C,B
        cellA = cells_i1[0]
        cellB = cells_i1[1]
        cellC = cells_i1[2]
        cellD = cells_i1[3]
        vert_A = self.cs[cellA].ind
        vert_B = self.cs[cellB].ind
        vert_C = self.cs[cellC].ind
        vert_D = self.cs[cellD].ind

        for i in vert_A:
            if i in vert_B and i not in vert_C and i not in vert_D:
                vert2 = i
        for i in vert_A:
            if i in vert_D and i not in vert_B and i not in vert_C:
                vert3 = i
        for i in vert_C:
            if i in vert_B and i not in vert_A and i not in vert_D:
                vert6 = i
        for i in vert_C:
            if i in vert_D and i not in vert_A and i not in vert_B:
                vert5 = i
        vert4 = vi1
        vi2 = self.vs[vi1].par[0]
        vert1 = self.vs[vi1].par[0]

        c1 = self.vs[vert4].eje[0]
        c2 = self.vs[vert4].eje[1]
        r_T1 = self.cs[c1].center - self.cs[c2].center
        l_r_T1 = np.linalg.norm(r_T1)
        rnorm_T1 = (1/l_r_T1) * r_T1

        #MODIFY ADY
        for i in range(len(self.vs[vert6].ady)):
            for j in range(len(self.vs[vert6].ady[i])):
                if self.vs[vert6].ady[i][j] == vert4:
                    self.vs[vert6].ady[i][j] = vert1
        for i in range(len(self.vs[vert5].ady)):
            for j in range(len(self.vs[vert5].ady[i])):
                if self.vs[vert5].ady[i][j] == vert4:
                    self.vs[vert5].ady[i][j] = vert1

        self.vs[vert4].cells = [cellA, cellB, cellD]
        self.vs[vert4].ady = [[vert2,vert3],[vert1,vert2],[vert3,vert1]]
        self.vs[vert1].cells = [cellC, cellD, cellB]
        self.vs[vert1].ady = [[vert5,vert6],[vert4,vert5],[vert6,vert4]]

        #MODIFY IND cell B
        add_B = []
        q = 0
        for i in range(len(self.cs[cellB].ind)-1):
            if self.cs[cellB].ind[i] == vert6 and q==0:
                add_B.append(vert6)
                add_B.append(vert1)
                q=1
            else:
                add_B.append(self.cs[cellB].ind[i])
        self.cs[cellB].ind = add_B
        if self.cs[cellB].ind[0] != self.cs[cellB].ind[len(self.cs[cellB].ind)-1]:
            self.cs[cellB].ind.append(self.cs[cellB].ind[0])

        #MODIFY IND cell C
        add_C = []
        q = 0
        for i in range(len(self.cs[cellC].ind)-1):
            if self.cs[cellC].ind[i] == vi1 and q==0:
                add_C.append(vert1)
                q=1
            else:
                add_C.append(self.cs[cellC].ind[i])
        self.cs[cellC].ind = add_C
        if self.cs[cellC].ind[0] != self.cs[cellC].ind[len(self.cs[cellC].ind)-1]:
            self.cs[cellC].ind.append(self.cs[cellC].ind[0])

        #MODIFY IND cell D
        add_D = []
        q = 0
        for i in range(len(self.cs[cellD].ind)-1):
            if self.cs[cellD].ind[i] == vi1 and q==0:
                add_D.append(vi1)
                add_D.append(vert1)
                q=1
            else:
                add_D.append(self.cs[cellD].ind[i])
        self.cs[cellD].ind = add_D
        if self.cs[cellD].ind[0] != self.cs[cellD].ind[len(self.cs[cellD].ind)-1]:
            self.cs[cellD].ind.append(self.cs[cellD].ind[0])


        self.vs[vert4].eje = [-1,-1]
        self.vs[vert4].par[0] = -1

        pos_vi1p = self.vs[vert4].r + (3/4)*self.l_T1 * rnorm_T1
        pos_vi2p = self.vs[vert4].r - (3/4)*self.l_T1 * rnorm_T1

        self.vs[vi1].r = pos_vi1p
        self.vs[vi2].r = pos_vi2p

        for c in [cellA,cellB,cellC,cellD]:
            self.cal_polygon(c)
            self.cs[c].cal_area()
            self.cs[c].cal_per()
            self.cs[c].cal_centro(self.dt)

        vel_i = self.vec_v_VM(vert4) + self.vec_v_AS(vert4)+self.vec_v_crawl_R1leader(vert4)
        vel_j = self.vec_v_VM(vert1) + self.vec_v_AS(vert1)+self.vec_v_crawl_R1leader(vert1)

        producto = np.dot((self.vs[vert4].r-self.vs[vert1].r), (vel_i -vel_j))
        return producto

    def delete_v1_original(self,vi1,vi2):
        not_repeated = []
        cells_i1 = self.vs[vi1].cells #A,B,D    o    B,D,A    o    D,A,B
        cells_i2 = self.vs[vi2].cells #B,C,D    o    C,D,B    o    D,B,C
        for i in range(len(cells_i1)):
            if cells_i1[i] in cells_i2:
                qq=0
            else:
                not_repeated.append(cells_i1[i]) #A
        for i in range(len(cells_i2)):
            if cells_i2[i] in cells_i1:
                qq=0
            else:
                not_repeated.append(cells_i2[i]) #C
        if cells_i1[0]== not_repeated[0]: #A,B,D
            cells_i1_new = [cells_i1[0],cells_i1[1],not_repeated[1]]
        elif cells_i1[1]== not_repeated[0]: #D,A,B
            cells_i1_new = [cells_i1[1],cells_i1[2],not_repeated[1]]
        else:  #B,D,A
            cells_i1_new = [cells_i1[2],cells_i1[0],not_repeated[1]]
        if cells_i2[0]== not_repeated[1]: #C,D,B
            cells_i2_new = [cells_i2[0],cells_i2[1],not_repeated[0]]
        elif cells_i2[1]== not_repeated[1]: #B,C,D
            cells_i2_new = [cells_i2[1],cells_i2[2],not_repeated[0]]
        else: #D,B,C
            cells_i2_new = [cells_i2[2],cells_i2[0],not_repeated[0]]

        cellA = cells_i1_new[0]
        cellB = cells_i1_new[1]
        cellC = cells_i1_new[2]
        cellD = cells_i2_new[1]
        vert_A = self.cs[cellA].ind[:len(self.cs[cellA].ind)-1]
        vert_B = self.cs[cellB].ind[:len(self.cs[cellB].ind)-1]
        vert_C = self.cs[cellC].ind[:len(self.cs[cellC].ind)-1]
        vert_D = self.cs[cellD].ind[:len(self.cs[cellD].ind)-1]

        for i in vert_A:
            if i in vert_B and i not in vert_C and i not in vert_D:
                vert2 = i
        for i in vert_A:
            if i in vert_D and i not in vert_B and i not in vert_C:
                vert3 = i
        for i in vert_C:
            if i in vert_B and i not in vert_A and i not in vert_D:
                vert6 = i
        for i in vert_C:
            if i in vert_D and i not in vert_A and i not in vert_B:
                vert5 = i
        vert4 = vi1
        vert1 = vi2

        #MODIFY ADY
        for i in range(len(self.vs[vert6].ady)):
            for j in range(len(self.vs[vert6].ady[i])):
                if self.vs[vert6].ady[i][j] == vert1:
                    self.vs[vert6].ady[i][j] = vert4

        for i in range(len(self.vs[vert5].ady)):
            for j in range(len(self.vs[vert5].ady[i])):
                if self.vs[vert5].ady[i][j] == vert1:
                    self.vs[vert5].ady[i][j] = vert4
        self.vs[vert4].cells = [cellA, cellB, cellC, cellD]
        self.vs[vert4].ady = [[vert2,vert3],[vert6,vert2],[vert5,vert6],[vert3,vert5]]
        self.vs[vert1].cells = []
        self.vs[vert1].ady = []

        #MODIFY IND cell B
        for i in self.cs[cellB].ind:
            if i == vi2:
                self.cs[cellB].ind.remove(vi2)
        if self.cs[cellB].ind[0] != self.cs[cellB].ind[len(self.cs[cellB].ind)-1]:
            self.cs[cellB].ind.append(self.cs[cellB].ind[0])

        #MODIFY IND cell C
        add_C = []
        q = 0
        for i in range(len(self.cs[cellC].ind)-1):
            if self.cs[cellC].ind[i] == vi2 and q==0:
                add_C.append(vi1)
                q=1
            else:
                add_C.append(self.cs[cellC].ind[i])
        self.cs[cellC].ind = add_C
        if self.cs[cellC].ind[0] != self.cs[cellC].ind[len(self.cs[cellC].ind)-1]:
            self.cs[cellC].ind.append(self.cs[cellC].ind[0])

        #MODIFY IND cell D
        for i in self.cs[cellD].ind:
            if i == vi2:
                self.cs[cellD].ind.remove(vi2)
        if self.cs[cellD].ind[0] != self.cs[cellD].ind[len(self.cs[cellD].ind)-1]:
            self.cs[cellD].ind.append(self.cs[cellD].ind[0])

        self.vs[vert4].eje = [cellA,cellC]
        self.vs[vert4].par[0] = vert1

        mid = 0.5 * (self.vs[vert4].r + self.vs[vert1].r) 

        self.vs[vi1].r = mid
        self.vs[vi2].r = mid

        for c in [cellA,cellB,cellC,cellD]:
            self.cal_polygon(c)
            self.cs[c].cal_area()
            self.cs[c].cal_per()
            self.cs[c].cal_centro(self.dt)

    def delete_v1_perpendicular(self,vi1,vi2): 
        not_repeated = []
        cells_i1 = self.vs[vi1].cells #A,B,D    o    B,D,A    o    D,A,B
        cells_i2 = self.vs[vi2].cells #B,C,D    o    C,D,B    o    D,B,C
        for i in range(len(cells_i1)):
            if cells_i1[i] in cells_i2:
                qq=0
            else:
                not_repeated.append(cells_i1[i]) #B
        for i in range(len(cells_i2)):
            if cells_i2[i] in cells_i1:
                qq=0
            else:
                not_repeated.append(cells_i2[i]) #D
        if cells_i1[0]== not_repeated[0]: #A,B,D
            cells_i1_new = [cells_i1[0],cells_i1[1],not_repeated[1]]
        elif cells_i1[1]== not_repeated[0]: #D,A,B
            cells_i1_new = [cells_i1[1],cells_i1[2],not_repeated[1]]
        else:  #B,D,A
            cells_i1_new = [cells_i1[2],cells_i1[0],not_repeated[1]]
        if cells_i2[0]== not_repeated[1]: #C,D,B
            cells_i2_new = [cells_i2[0],cells_i2[1],not_repeated[0]]
        elif cells_i2[1]== not_repeated[1]: #B,C,D
            cells_i2_new = [cells_i2[1],cells_i2[2],not_repeated[0]]
        else: #D,B,C
            cells_i2_new = [cells_i2[2],cells_i2[0],not_repeated[0]]

        cellB = cells_i1_new[0]
        cellC = cells_i1_new[1]
        cellD = cells_i1_new[2]
        cellA = cells_i2_new[1]
        vert_A = self.cs[cellA].ind[:len(self.cs[cellA].ind)-1]
        vert_B = self.cs[cellB].ind[:len(self.cs[cellB].ind)-1]
        vert_C = self.cs[cellC].ind[:len(self.cs[cellC].ind)-1]
        vert_D = self.cs[cellD].ind[:len(self.cs[cellD].ind)-1]

        for i in vert_A:
            if i in vert_B and i not in vert_C and i not in vert_D:
                vert2 = i
        for i in vert_A:
            if i in vert_D and i not in vert_B and i not in vert_C:
                vert3 = i
        for i in vert_C:
            if i in vert_B and i not in vert_A and i not in vert_D:
                vert6 = i
        for i in vert_C:
            if i in vert_D and i not in vert_A and i not in vert_B:
                vert5 = i
        vert4 = vi1
        vert1 = vi2

        #MODIFY ADY
        for i in range(len(self.vs[vert3].ady)):
            for j in range(len(self.vs[vert3].ady[i])):
                if self.vs[vert3].ady[i][j] == vert1:
                    self.vs[vert3].ady[i][j] = vert4

        for i in range(len(self.vs[vert5].ady)):
            for j in range(len(self.vs[vert5].ady[i])):
                if self.vs[vert5].ady[i][j] == vert1:
                    self.vs[vert5].ady[i][j] = vert4

        self.vs[vert4].cells = [cellA, cellB, cellC, cellD]
        self.vs[vert4].ady = [[vert2,vert3],[vert6,vert2],[vert5,vert6],[vert3,vert5]]
        self.vs[vert1].cells = []
        self.vs[vert1].ady = []

        #MODIFY IND cell C
        for i in self.cs[cellC].ind:
            if i == vi2:
                self.cs[cellC].ind.remove(vi2)
        if self.cs[cellC].ind[0] != self.cs[cellC].ind[len(self.cs[cellC].ind)-1]:
            self.cs[cellC].ind.append(self.cs[cellC].ind[0])

        #MODIFY IND cell D
        add_D = []
        q = 0
        for i in range(len(self.cs[cellD].ind)-1):
            if self.cs[cellD].ind[i] == vi2 and q==0:
                add_D.append(vi1)
                q=1
            else:
                add_D.append(self.cs[cellD].ind[i])
        self.cs[cellD].ind = add_D
        if self.cs[cellD].ind[0] != self.cs[cellD].ind[len(self.cs[cellD].ind)-1]:
            self.cs[cellD].ind.append(self.cs[cellD].ind[0])

        #MODIFY IND cell A
        for i in self.cs[cellA].ind:
            if i == vi2:
                self.cs[cellA].ind.remove(vi2)
        if self.cs[cellA].ind[0] != self.cs[cellA].ind[len(self.cs[cellA].ind)-1]:
            self.cs[cellA].ind.append(self.cs[cellA].ind[0])

        self.vs[vert4].eje = [cellA,cellC]
        self.vs[vert4].par[0] = vert1

        mid = 0.5 * (self.vs[vert4].r + self.vs[vert1].r)

        self.vs[vi1].r = mid
        self.vs[vi2].r = mid

        for c in [cellA,cellB,cellC,cellD]:
            self.cal_polygon(c)
            self.cs[c].cal_area()
            self.cs[c].cal_per()
            self.cs[c].cal_centro(self.dt)

    def previo_T1(self,vi1,vi2):
        not_repeated = []
        cells_i1 = self.vs[vi1].cells #A,B,D    o    B,D,A    o    D,A,B
        cells_i2 = self.vs[vi2].cells #B,C,D    o    C,D,B    o    D,B,C
        for i in range(len(cells_i1)):
            if cells_i1[i] in cells_i2:
                qq=0
            else:
                not_repeated.append(cells_i1[i]) #A
        for i in range(len(cells_i2)):
            if cells_i2[i] in cells_i1:
                qq=0
            else:
                not_repeated.append(cells_i2[i]) #C
        if cells_i1[0]== not_repeated[0]: #A,B,D
            cells_i1_new = [cells_i1[0],cells_i1[1],not_repeated[1]]
        elif cells_i1[1]== not_repeated[0]: #D,A,B
            cells_i1_new = [cells_i1[1],cells_i1[2],not_repeated[1]]
        else:  #B,D,A
            cells_i1_new = [cells_i1[2],cells_i1[0],not_repeated[1]]
        if cells_i2[0]== not_repeated[1]: #C,D,B
            cells_i2_new = [cells_i2[0],cells_i2[1],not_repeated[0]]
        elif cells_i2[1]== not_repeated[1]: #B,C,D
            cells_i2_new = [cells_i2[1],cells_i2[2],not_repeated[0]]
        else: #D,B,C
            cells_i2_new = [cells_i2[2],cells_i2[0],not_repeated[0]]

        cellA = cells_i1_new[0]
        cellB = cells_i1_new[1]
        cellC = cells_i1_new[2]
        cellD = cells_i2_new[1]
        vert_A = self.cs[cellA].ind[:len(self.cs[cellA].ind)-1]
        vert_B = self.cs[cellB].ind[:len(self.cs[cellB].ind)-1]
        vert_C = self.cs[cellC].ind[:len(self.cs[cellC].ind)-1]
        vert_D = self.cs[cellD].ind[:len(self.cs[cellD].ind)-1]

        if len(vert_A)>=4 and len(vert_C)>=4 and len(vert_B)>=5 and len(vert_D)>=5:
            return 1
        else:
            return -1

    def extrusion(self,ce):
        vert1 = self.cs[ce].ind[0]
        vert2 = self.cs[ce].ind[1]
        vert3 = self.cs[ce].ind[2]

        cells_vert1 = self.vs[vert1].cells #A,D,C ...
        cells_vert2 = self.vs[vert2].cells #A,D,B ...
        cells_vert3 = self.vs[vert3].cells #B,D,C ...

        cellD = ce
        for c in cells_vert1:
            if c in cells_vert2 and c!=cellD:
                cellA = c

        for c in cells_vert2:
            if c in cells_vert3 and c!=cellD:
                cellB = c

        for c in cells_vert3:
            if c in cells_vert1 and c!=cellD:
                cellC = c


        vert_A = self.cs[cellA].ind
        vert_B = self.cs[cellB].ind
        vert_C = self.cs[cellC].ind
        vert_D = self.cs[cellD].ind

        for i in vert_A:
            if i in vert_C and i not in vert_D:
                vert4 = i
        for i in vert_A:
            if i in vert_B and i not in vert_D:
                vert5 = i
        for i in vert_B:
            if i in vert_C and i not in vert_D:
                vert6 = i

        cells_vert4 = self.vs[vert4].cells #E,A,C ...
        cells_vert5 = self.vs[vert5].cells #A,B,F ...
        cells_vert6 = self.vs[vert6].cells #B,C,G ...

        for c in cells_vert4:
            if c!=cellA:
                if c!=cellC:
                    cellE = c

        for c in cells_vert5:
            if c!=cellA:
                if c!=cellB:
                    cellF = c

        for c in cells_vert6:
            if c!=cellB:
                if c!=cellC:
                    cellG = c

        #MODIFY IND1 cell A
        for i in self.cs[cellA].ind:
            if i == vert2:
                self.cs[cellA].ind.remove(vert2)
        if self.cs[cellA].ind[0] != self.cs[cellA].ind[len(self.cs[cellA].ind)-1]:
            self.cs[cellA].ind.append(self.cs[cellA].ind[0])


        #MODIFY IND1 cell B
        for i in self.cs[cellB].ind:
            if i == vert3:
                self.cs[cellB].ind.remove(vert3)
        if self.cs[cellB].ind[0] != self.cs[cellB].ind[len(self.cs[cellB].ind)-1]:
            self.cs[cellB].ind.append(self.cs[cellB].ind[0])

        for i in self.cs[cellB].ind:
            if i == vert2:
                self.cs[cellB].ind.remove(vert2)
        if self.cs[cellB].ind[0] != self.cs[cellB].ind[len(self.cs[cellB].ind)-1]:
            self.cs[cellB].ind.append(self.cs[cellB].ind[0])

        add_B = []
        q = 0
        for i in range(len(self.cs[cellB].ind)):
            if self.cs[cellB].ind[i] == vert6 and q==0:
                add_B.append(vert6)
                add_B.append(vert1)
                q = 1
            else:
                add_B.append(self.cs[cellB].ind[i])
        self.cs[cellB].ind = add_B
        if self.cs[cellB].ind[0] != self.cs[cellB].ind[len(self.cs[cellB].ind)-1]:
            self.cs[cellB].ind.append(self.cs[cellB].ind[0])

        #MODIFY IND1 cell A
        for i in self.cs[cellC].ind:
            if i == vert3:
                self.cs[cellC].ind.remove(vert3)
        if self.cs[cellC].ind[0] != self.cs[cellC].ind[len(self.cs[cellC].ind)-1]:
            self.cs[cellC].ind.append(self.cs[cellC].ind[0])

        #MODIFY ADY vert5
        for i in range(len(self.vs[vert5].ady)):
            for j in range(len(self.vs[vert5].ady[i])):
                if self.vs[vert5].ady[i][j] == vert2:
                    self.vs[vert5].ady[i][j] = vert1

        #MODIFY ADY vert6
        for i in range(len(self.vs[vert6].ady)):
            for j in range(len(self.vs[vert6].ady[i])):
                if self.vs[vert6].ady[i][j] == vert3:
                    self.vs[vert6].ady[i][j] = vert1

        self.vs[vert1].ady = np.array([[vert5,vert4],[vert6,vert5],[vert4,vert6]])
        self.vs[vert1].cells = np.array([cellA, cellB, cellC])
        self.vs[vert1].r = (1/3)* (self.vs[vert1].r + self.vs[vert2].r + self.vs[vert3].r)

        self.vs[vert2].tipo = 6
        self.vs[vert2].r = np.array([0,0,0])
        self.vs[vert2].ady = np.array([])
        self.vs[vert2].cells = np.array([])

        self.vs[vert3].tipo = 6
        self.vs[vert3].r = np.array([0,0,0])
        self.vs[vert3].ady = np.array([])
        self.vs[vert3].cells = np.array([])

        self.cs[cellD].ind = np.array([])

    def T1_swap_hole(self,vi1, vi2): 
        not_repeated = []
        cells_i1 = self.vs[vi1].cells #A,B,D    o    B,D,A    o    D,A,B
        cells_i2 = self.vs[vi2].cells #B,C,D    o    C,D,B    o    D,B,C

        for i in range(len(cells_i1)):
            if cells_i1[i] in cells_i2:
                qq=0
            else:
                not_repeated.append(cells_i1[i]) #A
        for i in range(len(cells_i2)):
            if cells_i2[i] in cells_i1:
                qq=0
            else:
                not_repeated.append(cells_i2[i]) #C

        if cells_i1[0]== not_repeated[0]: #A,B,D
            cells_i1_new = [cells_i1[0],cells_i1[1],not_repeated[1]]
        elif cells_i1[1]== not_repeated[0]: #D,A,B
            cells_i1_new = [cells_i1[1],cells_i1[2],not_repeated[1]]
        else:  #B,D,A
            cells_i1_new = [cells_i1[2],cells_i1[0],not_repeated[1]]

        if cells_i2[0]== not_repeated[1]: #C,D,B
            cells_i2_new = [cells_i2[0],cells_i2[1],not_repeated[0]]
        elif cells_i2[1]== not_repeated[1]: #B,C,D
            cells_i2_new = [cells_i2[1],cells_i2[2],not_repeated[0]]
        else: #D,B,C
            cells_i2_new = [cells_i2[2],cells_i2[0],not_repeated[0]]


        if len(self.cs[cells_i1_new[1]].ind)>=5 and len(self.cs[cells_i2_new[1]].ind)>=5: #squares -> triangles
            self.vs[vi1].cells = cells_i1_new #A,B,C
            self.vs[vi2].cells = cells_i2_new #C,D,A

            vert_A = self.cs[cells_i1_new[0]].ind
            vert_B = self.cs[cells_i1_new[1]].ind
            vert_C = self.cs[cells_i1_new[2]].ind
            vert_D = self.cs[cells_i2_new[1]].ind

            for i in vert_A:
                if i in vert_B and i!=vi1 :#i not in vert_C and i not in vert_D:
                    vert2 = i
            for i in vert_A:
                if i in vert_D and i!=vi1 :#and i not in vert_B and i not in vert_C:
                    vert3 = i
            for i in vert_C:
                if i in vert_B and i!=vi2 :#and i not in vert_A and i not in vert_D:
                    vert6 = i
            for i in vert_C:
                if i in vert_D and i!=vi2 :#and i not in vert_A and i not in vert_B:
                    vert5 = i
            vert4 = vi1
            vert1 = vi2

            #MODIFY ADY1
            for i in range(len(self.vs[vert3].ady)):
                for j in range(len(self.vs[vert3].ady[i])):
                    if self.vs[vert3].ady[i][j] == vert4:
                        self.vs[vert3].ady[i][j] = vert1

            for i in range(len(self.vs[vert6].ady)):
                for j in range(len(self.vs[vert6].ady[i])):
                    if self.vs[vert6].ady[i][j] == vert1:
                        self.vs[vert6].ady[i][j] = vert4

            self.vs[vert4].ady = [[vert2,vert1],[vert6,vert2],[vert1,vert6]]
            self.vs[vert1].ady = [[vert5,vert4],[vert3,vert5],[vert4,vert3]]


            #MODIFY IND1 cell A
            add_A = []
            q = 0
            for i in range(len(self.cs[cells_i1_new[0]].ind)): #cellA.ind
                if self.cs[cells_i1_new[0]].ind[i] == vi1 and q==0:
                    add_A.append(vi1)
                    add_A.append(vi2)
                    q = 1
                else:
                    add_A.append(self.cs[cells_i1_new[0]].ind[i])
            self.cs[cells_i1_new[0]].ind = add_A
            if self.cs[cells_i1_new[0]].ind[0] != self.cs[cells_i1_new[0]].ind[len(self.cs[cells_i1_new[0]].ind)-1]:
                self.cs[cells_i1_new[0]].ind.append(self.cs[cells_i1_new[0]].ind[0])

            #MODIFY IND1 cell B
            for i in self.cs[cells_i1_new[1]].ind:
                if i == vi2:
                    self.cs[cells_i1_new[1]].ind.remove(vi2)
            if self.cs[cells_i1_new[1]].ind[0] != self.cs[cells_i1_new[1]].ind[len(self.cs[cells_i1_new[1]].ind)-1]:
                self.cs[cells_i1_new[1]].ind.append(self.cs[cells_i1_new[1]].ind[0])

            #MODIFY IND1 cell C
            add_C = []
            q = 0
            for i in range(len(self.cs[cells_i2_new[0]].ind)):
                if self.cs[cells_i2_new[0]].ind[i] == vi2 and q==0:
                    add_C.append(vi2)
                    add_C.append(vi1)
                    q=1
                else:
                    add_C.append(self.cs[cells_i2_new[0]].ind[i])

            self.cs[cells_i2_new[0]].ind = add_C
            if self.cs[cells_i1_new[2]].ind[0] != self.cs[cells_i1_new[2]].ind[len(self.cs[cells_i1_new[2]].ind)-1]:
                self.cs[cells_i1_new[2]].ind.append(self.cs[cells_i1_new[2]].ind[0])

            #MODIFY IND1 cell D
            for i in self.cs[cells_i2_new[1]].ind:
                if i == vi1:
                    self.cs[cells_i2_new[1]].ind.remove(vi1)
            if self.cs[cells_i2_new[1]].ind[0] != self.cs[cells_i2_new[1]].ind[len(self.cs[cells_i2_new[1]].ind)-1]:
                self.cs[cells_i2_new[1]].ind.append(self.cs[cells_i2_new[1]].ind[0])

            mid_pos =0.5 * (self.vs[vi2].r + self.vs[vi1].r)
            mid_k = np.array([0,0,1])

            l1mid = self.vs[vi1].r - mid_pos
            l2mid = self.vs[vi2].r - mid_pos

            self.vs[vi1].r = np.cross(l1mid,mid_k) + mid_pos

            self.vs[vi2].r = np.cross(l2mid,mid_k) + mid_pos

            for c in cells_i1_new:
                self.cal_polygon(c)
                self.cs[c].cal_centro_T1()
                self.cs[c].cal_area()
                self.cs[c].cal_per()
            for c in cells_i2_new:
                self.cal_polygon(c)
                self.cs[c].cal_centro_T1()
                self.cs[c].cal_area()
                self.cs[c].cal_per()
