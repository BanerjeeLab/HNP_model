pos0 = []
cells0 = []
t = []
for i in range(n_vertices):
    pos0.append(np.array([x[i],y[i],0]))
    cells_i_horario = cells_tipo1[i]
    ady_i_horario = adjs_tipo1[i]
    t.append(Vertex(i,pos0[i],np.array(cells_i_horario), np.array(ady_i_horario)))

#Creations of cells
celulas = []
for i in range(n_celulas):
    b = []
    for j in range(len(celda)):
        if celda[j] == i:
            b.append(int(v_celda[j]))
    celulas.append(Cell(i,b))   
    b.append(b[0])

#Creation of tissue
T1 = Tissue(t, celulas, celulas[len(celulas)-1], celulas[len(celulas)-2],Th,v0,Sigma0,ma,md,mf,dt)

T1.R1 = T1.first_row()
T1.R2 = T1.third_row(T1.R1)[1]
T1.R3 = T1.third_row(T1.R1)[2]
