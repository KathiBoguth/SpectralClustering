import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
#matplotlib.use("GTKAgg")
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg, NavigationToolbar2Tk
from matplotlib.backend_bases import NavigationToolbar2
from matplotlib.figure import Figure
import matplotlib.colors as colors
from matplotlib.mlab import bivariate_normal


#from sklearn.cluster import SpectralClustering
import seaborn as sns
import statistics
import math
from scipy.sparse import csgraph, csr_matrix
import scipy.linalg as la
from scipy.sparse.linalg import eigsh
from scipy.spatial.distance import pdist, squareform, minkowski
from scipy.sparse.csgraph import laplacian as csgraph_laplacian
import networkx.linalg.laplacianmatrix as nx
from networkx import from_numpy_matrix, to_numpy_matrix
import networkx.algorithms.community.quality as quality

import time

import tkinter as tk
from tkinter import ttk
from tkinter import filedialog
import os

import csv
import ast
import pandas
import random

##for Clustering
import warnings
from pathlib import Path

import numpy as np
from operator import itemgetter

from PIL import Image, ImageTk

from sklearn.base import BaseEstimator, ClusterMixin
from sklearn.utils import check_random_state, as_float_array
from sklearn.utils.validation import check_array
from sklearn.metrics.pairwise import pairwise_kernels
from sklearn.metrics import normalized_mutual_info_score, adjusted_rand_score
from sklearn.manifold import spectral_embedding_
from sklearn.cluster.k_means_ import k_means
import sklearn.datasets as ds
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph, NearestNeighbors, RadiusNeighborsClassifier
import sklearn.metrics.pairwise as pairwise
import sklearn.decomposition as dec
#from sklearn.cluster import spectral


LARGE_FONT = ("Verdana", 12)

#global variables
samples = 50
k_neighbors = 6
gamma = 0.5
epsilon = 1.0
eMax=10
numb_eig = 3
n_clusters = 3
lineLastID = 1

X = None
Xorig= None
y = None
admatrix = None
linesMatrix = None
laplacian = None
fig = None
fig2 = None
subplotGraph = None
eigvalues=None

markedDataPoint = -1

labels = None
zoomEigsActive=False



#click_radius = statistics.mean([max(X[:,0]) - min(X[:,0]), max(X[0,:]) -
#min(X[0,:])]) * 0.015

#keepCustom

#style: https://coolors.co/01061e-e1dce0-d8d813-7a9b76-c1292e
#style2: https://coolors.co/242038-33658a-fbf7e9-6b0f1a-426b69
color_lines = '#2F5C7E'      #'#426B69'      #'#909090'   #'#6e53d1'
color_dots =   '#8AB0AB'     #'#01061E'
color_marked = '#D72638'     #'#6B0F1A'     #'#D8D813'
color_custom =  '#8AB0AB'    #'#33658A'     #'#353535'    #'#2dc62d'
#color_bg='#E1DCE0'
color_bg = "#242038"
color_text = '#FBF7E9'
#color clusters:
colors_cluster = ["blue", "green", "yellow", "magenta"]


class SeaofBTCapp(tk.Tk):

    def __init__(self, *args, **kwargs):
        
        tk.Tk.__init__(self, *args, **kwargs)

        #tk.Tk.iconbitmap(self, default="clienticon.ico")
        tk.Tk.wm_title(self, "Spectral Clustering") 
        
        container = tk.Frame(self)#, width=550, height=450
        #container.pack(side="top", fill="both", expand = True)
        container.grid(row=0, column=0, sticky=tk.NSEW)
        container.grid_rowconfigure(0, weight=1)
        container.grid_columnconfigure(0, weight=1)

        self.frames = {}
        

        F = (PageThree)

        frame = F(container, self)

        self.frames[F] = frame
        frame.config(bg=color_bg)

        frame.grid(row=0, column=0, sticky="nsew")
        #frame.grid_rowconfigure(0, weight=1)
        #frame.grid_columnconfigure(0, weight=1)

        #self.show_frame(StartPage)

    def show_frame(self, cont):

        frame = self.frames[cont]
        frame.tkraise()




class PageThree(tk.Frame):
    mainframe = None
    markedPoint = None
    #hoveredPoint = None

    #subplotGraph = None
    scatter = None
    customLines = []
    deletedLines = []
    lines = []
    keepCustom = None
    canvas = None
    canvas2 = None
    tab1 = None
    tab2 = None
    tab3 = None
    frameHeatmap = None
    indicatorLine = None
    vecMat= None
    nextNeighbors = None
    distances = None
    gaussianDistances=None
    lastResults = []
    chooseMatrix=None
    chooseGraph=None
    vert=None
    hor=None
    oldResults=[]
    oldResultsX=[]
    dimension1= None
    dimension2=None
    variableDisplay=None
    labDim=None
    

    

    def __init__(self, parent, controller):


        global mainframe, fig, fig2, subplotGraph, admatrix, laplacian, linesMatrix, lineLastID, keepCustom, n_clusters, X, y, k_neighbors, eigvalues, canvas, eMax, zoomEigsActive  #subplotgraph

        zoom = NavigationToolbar2.zoom

        def new_zoom(self, *args, **kwargs):
            global zoomEigsActive
            zoomEigsActive = not zoomEigsActive
            zoom(self, *args, **kwargs)
            

        NavigationToolbar2.zoom = new_zoom

        ###################Elements##############################################################
        #self.geometry('500x300+0+0')
        #parent.grid_columnconfigure(0, weight=1)
        

        ####Tabs
        mainframe = tk.Frame.__init__(self, parent)
        #self.grid_columnconfigure(0, weight=1)
        #self.grid_rowconfigure(0, weight=1)
        


        style = ttk.Style(mainframe)
        #tk.Tk.style = ttk.Style()
        #('winnative', 'clam', 'alt', 'default', 'classic', 'vista', 
        #'xpnative')
        #tk.Tk.style.theme_use("vista")
        #style.theme_use("clam")
        style.configure("My.TFrame", background=color_bg, foreground=color_bg)
        style.configure("My.TNotebook", background=color_bg, foreground=color_bg)
        style.configure("TNotebook.Tab", background=color_bg, foreground=color_bg)
        style.configure("My.TRadiobutton", background=color_bg, foreground=color_text)



        tabControl = ttk.Notebook(self, style="My.TNotebook") 
        self.tab1 = ttk.Frame(tabControl, style="My.TFrame")            
        tabControl.add(self.tab1, text='Graph')      
     
        self.tab2 = ttk.Frame(tabControl, style="My.TFrame")            
        tabControl.add(self.tab2, text='Eigenvectors')  

        self.tab3 = ttk.Frame(tabControl, style="My.TFrame")            
        tabControl.add(self.tab3, text='PCA') 
        
        
        tabControl.grid(row=0, column=0, rowspan=7, padx=5, pady=5, sticky=tk.NSEW) 
        tabControl.grid_columnconfigure(0, weight=1)
        tabControl.grid_rowconfigure(0, weight=1)

        ####Entry for Vectors



        tk.Label(self.tab2, text="vertical:", bg=color_bg, fg=color_text).grid(row=0, column=1, sticky=tk.W)
        tk.Label(self.tab2, text="horizontal:", bg=color_bg, fg=color_text).grid(row=0, column=1, sticky=tk.SW)
        self.vert = tk.StringVar()
        self.vert.set(0)
        self.vert.trace("w", self.change_vert)
        self.hor = tk.StringVar()
        self.hor.set(1)
        self.hor.trace("w", self.change_hor)
        tk.Entry(self.tab2, width=3, textvariable=self.vert).grid(row=0, column=2, sticky=tk.E)
        tk.Entry(self.tab2, width=3, textvariable=self.hor).grid(row=0, column=2, sticky=tk.SE)


        ####Heatmap
        self.frameHeatmap = ttk.Frame(self, style="My.TFrame")
        self.frameHeatmap.grid(row=2, column=2)
        lab3=tk.Label(self.frameHeatmap, text="Type of Laplacian Matrix:", bg=color_bg, fg=color_text)
        lab3.grid(row=0, column=0, padx=5)
        
        X, y = ds.make_blobs(n_samples=samples)
        fig = plt.Figure(figsize=(6,5.5), dpi=100)


        #####Eigenvalues
        self.frameEigenG = ttk.Frame(self, style="My.TFrame")
        self.frameEigenG.grid(row=4, column=2)

        self.figEigVal = Figure(figsize=(3.5,1.5), dpi=100)
        self.figEigVal.patch.set_facecolor(color_bg)

        params = {"ytick.color" : color_text,
          "xtick.color" : color_text,
          "axes.labelcolor" : color_text,
          "axes.edgecolor" : color_text}
        plt.rcParams.update(params)

        ##Setup
        self.setup()

        ###Load Data Button
        def click_LoadData(*args):
            #tk.TK.withdraw()
            self.newWindow = tk.Toplevel()  #self.master
            bb = LoadData(self.newWindow, self)


        buttonLoad = ttk.Button(self.tab1, text="Load Data",
                            command=click_LoadData)
        buttonLoad.grid(row=5, column=4, padx=5, sticky=tk.E)
        
        ####Toolbar
        #toolbarFrame = ttk.Frame(self.tab1, style="My.TFrame")
        #toolbarFrame.grid(row=1,column=0, columnspan=2, sticky=tk.W, padx=10, pady=10)
        #toolbar = NavigationToolbar2Tk(fig.canvas, toolbarFrame)    #Agg
        #toolbar.config(background=color_bg)
        #toolbar._message_label.config(background=color_bg, foreground=color_text)
        ##toolbar.update()

        ####Dropdown Graph

        #####Slider k, e and gaussian
        frameSimGraph = ttk.Frame(self, style="My.TFrame")
        frameSimGraph.grid(row=0, column=2, pady=10)
        self.chooseGraph = tk.StringVar(frameSimGraph)
        choices = { 'kNN','epsilon','complete', 'gaussian'}
        #chooseGraph.set('kNN')
        popupMenuGraph = ttk.OptionMenu(frameSimGraph, self.chooseGraph, 'kNN', *choices)
        popupMenuGraph.config(width=10)

        #tk.Label(mainframe, text="Graph").grid(row = 1, column = 1)

        
        def change_chooseGraph(*args):
            global admatrix, samples
            if self.chooseGraph.get() == 'epsilon':
                self.updateData(self.chooseGraph.get(), self.sliderE.get())
                self.sliderK.grid_remove()
                self.sliderG.grid_remove()
                self.sliderE.grid(row=1, column=1, padx=5, sticky=tk.E)
            if self.chooseGraph.get() == 'kNN':
                self.updateData('k_neighbors', self.sliderK.get())
                self.sliderE.grid_remove()
                self.sliderG.grid_remove()
                self.sliderK.grid(row=1, column=1, padx=5, sticky=tk.E)
            if self.chooseGraph.get() == 'complete':
                self.updateData(self.chooseGraph.get())
                self.sliderK.grid_remove()
                self.sliderE.grid_remove()
                self.sliderG.grid_remove()
            if self.chooseGraph.get() == 'gaussian':
                self.updateData(self.chooseGraph.get(), self.sliderG.get())
                self.sliderK.grid_remove()
                self.sliderE.grid_remove()
                self.sliderG.grid(row=1, column=1,  padx=5, sticky=tk.E)
            #redraw()
            self.calcLaplacian()
            self.update_Lines()
            #self.recluster()


        popupMenuGraph.grid(row = 0, column =1, padx=5, sticky=tk.E)
        self.chooseGraph.trace('w', change_chooseGraph)


        lab1=tk.Label(frameSimGraph, text="Generate Similarity Graph:", bg=color_bg, fg=color_text)
        lab1.grid(row=0, column=0, padx=5, pady=5)


        


        #tk.Label(self, text="Graph").grid(row = 0, column = 3)
        def change_sliderK(event):
            global k_neighbors, admatrix
            self.updateData("k_neighbors", self.sliderK.get())
            #redraw()
            self.update_Lines()
            #recluster()
            self.calcLaplacian()

        def change_sliderE(event):
            global epsilon, admatrix
            self.updateData("epsilon", self.sliderE.get())
            #redraw()
            self.update_Lines()
            #recluster()
            self.calcLaplacian()



        def change_sliderG(event):
            global gamma, admatrix
            self.updateData("gaussian", self.sliderG.get())
            #redraw()
            self.update_Lines()
            #recluster()
            self.calcLaplacian()



        style.configure("My.TScale", background=color_bg, foreground=color_text)
        self.sliderK = tk.Scale(frameSimGraph, from_=1, to=samples,  bg=color_bg, highlightthickness=0, fg=color_text, orient=tk.HORIZONTAL) # style="Horizontal.TScale" label='k',
        self.sliderK.bind("<ButtonRelease-1>", change_sliderK)
        self.sliderK.set(k_neighbors)
        self.sliderK.grid(row=1, column=1, padx=5, sticky=tk.E)

        self.sliderE = tk.Scale(frameSimGraph, from_=0, to=eMax,  resolution=0.1, bg=color_bg, fg=color_text, highlightthickness=0, orient=tk.HORIZONTAL) #label='e',
        self.sliderE.bind("<ButtonRelease-1>", change_sliderE)
        self.sliderE.set(epsilon)
        #sliderE.grid(row=0, column=4, sticky=tk.N, padx=5)
        #sliderE.grid_remove()

        self.sliderG = tk.Scale(frameSimGraph, from_=0, to=eMax-1, resolution=0.5, bg=color_bg, fg=color_text, highlightthickness=0, orient=tk.HORIZONTAL) # style="Horizontal.TScale" label='gamma',
        self.sliderG.bind("<ButtonRelease-1>", change_sliderG)
        self.sliderG.set(gamma)
        #sliderG.grid(row=0, column=4, sticky=tk.N, padx=5)
        #sliderG.grid_remove()


        ######Checkbox custom Lines
        style.configure("My.TCheckbutton", background=color_bg, foreground=color_text)
        keepCustom = tk.BooleanVar() 
        checkBox = ttk.Checkbutton(frameSimGraph, text="Keep custom Lines",  variable=keepCustom, style="My.TCheckbutton")
        checkBox.grid(row=1, column=0, sticky=tk.W, padx=5)
        #checkBox.trace('w', change_checkBox)

        ####Seperators
        style.configure("TSeperator", background=color_text, foreground=color_text)
        sep1 = ttk.Separator(self, orient="horizontal")
        sep1.grid(row=1, column=2, pady=10, sticky="we")
        sep2 = ttk.Separator(self, orient="horizontal")
        sep2.grid(row=3, column=2, pady=10, sticky="we")
        sep3 = ttk.Separator(self, orient="horizontal")
        sep3.grid(row=5, column=2, pady=10, sticky="we")
        sep4 = ttk.Separator(self, orient="vertical")
        sep4.grid(row=0, column=1, rowspan=7, pady=10, sticky="ns")
        sep5 = ttk.Separator(self, orient="vertical")
        sep5.grid(row=0, column=3, rowspan=7, pady=10, sticky="ns")
        
       

        #######Choose Matrix
        def change_chooseMatrix(*args):
            global admatrix, laplacian
            self.calcLaplacian()
            self.calcEigVecs()
            self.change_vert()
            self.change_hor()
            self.redraw_heatmap()
            self.recluster()

        self.chooseMatrix = tk.StringVar(self.frameHeatmap)
        choices = { 'Laplacian','Lsym','Lrw', 'alt Laplacian'}
        popupMenuMatrix = ttk.OptionMenu(self.frameHeatmap, self.chooseMatrix, 'Laplacian', *choices)
        popupMenuMatrix.config(width=10)

        
        popupMenuMatrix.grid(row = 0, column =1, padx=5)
        self.chooseMatrix.trace('w', change_chooseMatrix)



        #chooseMatrix.set('Laplacian') # set the default option


        #######choose Algorithm
        style.configure("TMenubutton", background=color_text)
        frameBottom = ttk.Frame(self, style="My.TFrame")
        frameBottom.grid(row=6, column=2, pady=10)
        chooseAlgo = tk.StringVar(frameBottom)
        choices = { 'kmeans','b','c'}
        chooseAlgo.set('kmeans') # set the default option
        popupMenuAlgo = ttk.OptionMenu(frameBottom, chooseAlgo, 'kmeans', *choices)
        popupMenuAlgo.config(width=10)
        #tk.Label(mainframe, text="Graph").grid(row = 1, column = 1)
        lab4=tk.Label(frameBottom, text="Clustering Algorithm", bg=color_bg, fg=color_text)
        lab4.grid(row=0, column=0)
 
        def change_chooseAlgo(*args):
            print(chooseAlgo.get())

        popupMenuAlgo.grid(row = 0, column =1, padx=10, pady=20)
        chooseAlgo.trace('w', change_chooseAlgo)

        

        ######number of Clusters
        def change_sliderClusters(event):
            global n_clusters
            n_clusters = self.sliderCluster.get()
            self.recluster()

        self.sliderCluster = tk.Scale(frameBottom, from_=1, to=samples / 2, bg=color_bg, fg=color_text, highlightthickness=0, orient=tk.HORIZONTAL) # style="Horizontal.TScale" label='|C|',
        self.sliderCluster.bind("<ButtonRelease-1>", change_sliderClusters)
        self.sliderCluster.set(n_clusters)
        self.sliderCluster.grid(row=0, column=2, sticky=tk.N, padx=10, pady=5)




        #######Save Button
        def click_Save(*args):
            global subplotGraph, fig
            frame=ttk.Frame(frameHistory.viewPort, style="My.TFrame")
            nb_clusters, _, _ = eigenDecomposition()
            print("k Clusters: " + str(nb_clusters))
            part=[[] for x in range(np.amax(self.clustering.labels_)+1)]
            for i in range(0, len(X)):
                part[self.clustering.labels_[i]].append(i)
            q=quality.modularity(from_numpy_matrix(admatrix), part)
            cond=quality.coverage(from_numpy_matrix(admatrix), part)
            temp = np.c_[X, self.clustering.labels_]
            f = filedialog.asksaveasfile(mode='w', defaultextension=".csv")
            if f is None: 
                return
            #text2save = str(X.get(1.0, END))
            #f.write(text2save)
            header= ","*len(X[0])+"class\n"
            f.write(header) 
            np.savetxt(f, temp, fmt='%.3f', delimiter=",")
            
            tk.Label(frame, text=Path(f.name).stem+":", bg=color_bg, fg=color_text).grid(row=0, column=0)

            #path="C:\\Users\\Saturn\\Documents\\Uni\\bachelorarbeit\\img.png"
            path=os.path.dirname(f.name)+"/prev.png"
            fig.savefig(path, format="png", transparent=True)
            load = Image.open(path)
            render = ImageTk.PhotoImage(load.resize((165, 155)))
            img = tk.Label(frame, image=render, bg=color_bg, fg=color_bg, borderwidth=0, highlightthickness=0)
            img.image = render
            img.grid(row=1, column=0, columnspan=2, rowspan=3)
            #img.place(x=0, y=0)
            tk.Label(frame, text="Modularity: "+str(round(q,3)), bg=color_bg, fg=color_custom).grid(row=3, column=0, sticky=tk.SW)
            tk.Label(frame, text="Conductance: "+str(round(cond,3)), bg=color_bg, fg=color_custom).grid(row=3, column=0, sticky=tk.W)

            f.close() 

            fd = filedialog.asksaveasfile(mode='w', defaultextension=".txt")
            if fd is None: 
                return
            #text2save = str(X.get(1.0, END))
            #f.write(text2save)
            if self.chooseGraph.get() == "kNN":
                sl=str(self.sliderK.get())
            elif self.chooseGraph.get() == "epsilon":
                sl=str(self.sliderE.get())
            elif self.chooseGraph.get() == "gamma":
                sl=str(self.sliderG.get())
            else:
                sl=""

            temp=["chooseGraph, value, chooseMatrix, Eigenvalues, number of clusters, custom lines, deleted lines", str(self.chooseGraph.get()), sl, self.chooseMatrix.get(),
                         str(int(self.eigVLine.get_xdata()[0])+1), n_clusters, self.customLines, self.deletedLines]
            numb=len(self.oldResults)
            np.savetxt(fd, temp, fmt="%s", delimiter=",")
            tk.Button(frame, text="Restore", command=lambda a=numb: 
                                self.restore(i=a)).grid(row=3, column=1, padx=5, sticky=tk.SE)
            self.oldResults.append(temp)
            self.oldResultsX.append(X)

            fd.close() 
            ttk.Separator(frame, orient="horizontal").grid(row=4, column=0, columnspan=2, pady=10, sticky="we")
            self.lastResults.insert(0,frame)
            #if len(self.lastResults)>3:
            #    self.lastResults.pop(3)
            self.grid_propagate(1)
            for i in range(len(self.lastResults)):
                self.lastResults[i].grid(row=i+2, column=0)
            #self.scrollbar.config(command=self.canvasHistory.yview)
            #frameHistory.onFrameConfigure()
            #self.configure(height=self["height"],width=self["width"])
            #self.grid_propagate(0)



            




        buttonSave = ttk.Button(self.tab1, text="Save",
                            command=click_Save, style='Fun.TButton')
        buttonSave.grid(row=5, column=5, padx=5, sticky=tk.E)




        #frame = Frame(root, bd=2, relief=SUNKEN)
        frameHistory=ScrollFrame(self)#ttk.Frame(self, style="My.TFrame")

        frameHistory.grid(row=0, column=4, rowspan=20, sticky=tk.N)


        res=tk.Label(frameHistory.viewPort, text="Your last Results:", bg=color_bg, fg=color_text)
        res.grid(row=0, column=0, padx=10, pady=10, stick="we")
        ttk.Separator(frameHistory.viewPort, orient="horizontal").grid(row=1, column=0, sticky="we")
        #frameHistory.configure(height=640,width=200)
        #frameHistory.grid_propagate(0)



        ########################Logical
       

        ###To find best number k for Clusters
        def eigenDecomposition(coordinates=X, plot=False):
            """
            :param A: Affinity matrix
            :param plot: plots the sorted eigen values for visual inspection
            :return A tuple containing:
            - the optimal number of clusters by eigengap heuristic
            - all eigen values
            - all eigen vectors
    
            This method performs the eigen decomposition on a given affinity matrix,
            following the steps recommended in the paper:
            1. Construct the normalized affinity matrix: L = D−1/2ADˆ −1/2.
            2. Find the eigenvalues and their associated eigen vectors
            3. Identify the maximum gap which corresponds to the number of clusters
            by eigengap heuristic
    
            References:
            https://papers.nips.cc/paper/2619-self-tuning-spectral-clustering.pdf
            http://www.kyb.mpg.de/fileadmin/user_upload/files/publications/attachments/Luxburg07_tutorial_4488%5b0%5d.pdf
            """
            k = k_neighbors
            # calculate euclidian distance matrix
            dists = squareform(pdist(coordinates)) 
    
            # for each row, sort the distances ascendingly and take the index
            # of the
            #k-th position (nearest neighbour)
            knn_distances = np.sort(dists, axis=0)[k]
            knn_distances = knn_distances[np.newaxis].T
    
            # calculate sigma_i * sigma_j
            local_scale = knn_distances.dot(knn_distances.T)

            affinity_matrix = dists * dists
            affinity_matrix = -affinity_matrix / local_scale
            # divide square distance matrix by local scale
            affinity_matrix[np.where(np.isnan(affinity_matrix))] = 0.0
            # apply exponential
            affinity_matrix = np.exp(affinity_matrix)
            np.fill_diagonal(affinity_matrix, 0)

            A = affinity_matrix

            #L = csgraph.laplacian(A, normed=True)
            L = csgraph.laplacian(A, normed=False)
            n_components = A.shape[0]
    
            # LM parameter : Eigenvalues with largest magnitude (eigs, eigsh),
            # that is, largest eigenvalues in
            # the euclidean norm of complex numbers.
            eigenvalues, eigenvectors = eigsh(L, k=n_components, which="LM", sigma=1.0, maxiter=5000)
    
            if plot:
                plt.title('Largest eigen values of input matrix')
                plt.scatter(np.arange(len(eigenvalues)), eigenvalues)
                plt.grid()
        
            # Identify the optimal number of clusters as the index
            # corresponding
            # to the larger gap between eigen values
            index_largest_gap = np.argmax(np.diff(eigenvalues))
            nb_clusters = index_largest_gap + 1
        
            return nb_clusters, eigenvalues, eigenvectors

        nb_clusters, _, _ = eigenDecomposition()
        print("k Clusters: " + str(nb_clusters))

            
        self.configure(height=640,width=1245)
        self.grid_propagate(0)

   
        ####End of init

    def restore(self, i, file=None):
        global k_neighbors, gamma, epsilon, numb_eig, n_clusters, X
        if file is not None:
            with open(file, 'r') as csvfile:
                reader = csv.reader(csvfile, delimiter= '|')
                linecount = 0
                temp = []
                #temp=np.array()
                reader = csv.reader(csvfile, delimiter= '|')
                for row in reader:
                    #rowfloat = [float(i) for i in row]
                    temp.append(row[0])
                    #temp[linecount]=rowfloat
                    #temp[linecount][0]=rowfloat[0]
                    #temp[linecount][1]=rowfloat[1]
                    linecount+=1
                #temp[2]=int(temp[2])
                temp[4]=int(temp[4])
                temp[5]=int(temp[5])
                temp[6]=ast.literal_eval(temp[6])
                temp[7]=ast.literal_eval(temp[7])
        else:
            print(self.oldResults[i])
            temp=self.oldResults[i]
            X=self.oldResultsX[i]
            self.setup()

        temp = np.array(temp)
        self.chooseGraph.set(temp[1])
        if self.chooseGraph.get() == "kNN":
            k_neighbors=int(temp[2])
            self.sliderK.set(k_neighbors)
        elif self.chooseGraph.get() == "epsilon":
            epsilon=float(temp[2])
            self.sliderE.set(epsilon)
        elif self.chooseGraph.get() == "gamma":
            gamma=float(temp[2])
            self.sliderG.set(epsilon)

        #change_sliderK()
        self.eigVLine.set_xdata([int(temp[4]), int(temp[4])])
        numb_eig=int(self.eigVLine.get_xdata()[0])

        self.chooseMatrix.set(temp[3])
        #change_chooseMatrix()
        n_clusters=int(temp[5])
        self.sliderCluster.set(n_clusters)

        self.redraw()
        #self.update_Lines()

        #x=ast.literal_eval(temp[6])
        # x=ast.parse(temp[6], mode='eval')
        x=temp[6]
        for a,b in x:
            dist = minkowski(X[a],X[b],2)     #math.sqrt((X[a,0] - X[b,0]) ** 2 + (X[a,0] - X[b,0]) ** 2)
            admatrix[a][b] = dist
            admatrix[b][a] = dist

        # x=ast.literal_eval(temp[7])
        x=temp[7]
        for a,b in x:
            admatrix[a][b] = 0
            admatrix[b][a] = 0

        self.update_Lines(custom=True)
        self.recluster()


    ################Drawing
    
    #####Setup for new Data model
    def setup(self):
        global fig, fig2, admatrix, linesMatrix, lineLastID, lines, markedDataPoint, X, y, subplotGraph, laplacian, eigvalues, Xorig, eMax

        self.markedEig=3
        n_clusters=3

        dim1=0
        dim2=1
        if self.variableDisplay is not None:
            if self.variableDisplay.get()==2:
                dim1=self.sliderDimension1.get()-1
                dim2=self.sliderDimension2.get()-1

        ###PCA or Dimension
        if len(X[0])>2:
            self.sliderDimension1 = tk.Scale(self.tab1, from_=1, to=len(X[0]),  bg=color_bg, highlightthickness=0, fg=color_text, orient=tk.HORIZONTAL, length=80) # style="Horizontal.TScale" label='k',
            self.sliderDimension1.bind("<ButtonRelease-1>", self.change_sliderDimension1)
            self.sliderDimension2 = tk.Scale(self.tab1, from_=1, to=len(X[0]),  bg=color_bg, highlightthickness=0, fg=color_text, orient=tk.HORIZONTAL, length=80) # style="Horizontal.TScale" label='k',
            self.sliderDimension2.bind("<ButtonRelease-1>", self.change_sliderDimension1)
            self.sliderDimension1.grid(row=1, column=5, padx=5, sticky=tk.N)
            self.sliderDimension2.grid(row=1, column=5, padx=5)
            self.sliderDimension2.set(2)

            #self.variableDisplay = tk.IntVar()indicate
            #self.variableDisplay.set(1)
            #ttk.Radiobutton(self.tab1, text="Show PCA:", variable=self.variableDisplay, value=1, command=self.switchDisplay, style="My.TRadiobutton").grid(row=0, column=5, padx=5)
            #ttk.Radiobutton(self.tab1, text="Show Data:", variable=self.variableDisplay, value=2, command=self.switchDisplay, style="My.TRadiobutton").grid(row=0, column=5, padx=5, sticky=tk.S)
            tk.Label(self.tab1, text="Dimensions:", background=color_bg, foreground=color_text).grid(row=0, column=5, padx=5, sticky=tk.S)

        if labels is not None:
            tk.Label(self.tab1, text="NMI:", background=color_bg, foreground=color_text).grid(row=2, column=5, padx=5, sticky=tk.N)
            label_nmi=tk.Label(self.tab1, text="__", background=color_bg, foreground=color_text)
            label_nmi.grid(row=2, column=5, padx=5, sticky=tk.NE)
            tk.Label(self.tab1, text="ARI:", background=color_bg, foreground=color_text).grid(row=2, column=5, padx=5)
            label_ari=tk.Label(self.tab1, text="__", background=color_bg, foreground=color_text)
            label_ari.grid(row=2, column=5, padx=5, sticky=tk.E)

            def click_recalc():
                nmi=round(normalized_mutual_info_score(labels_true=labels, labels_pred=self.clustering.labels_), 3)
                ari=round(adjusted_rand_score(labels_true=labels, labels_pred=self.clustering.labels_), 3)
                label_nmi.config(text=nmi)
                label_ari.config(text=ari)

            ttk.Button(self.tab1, text="Recalc", command=click_recalc).grid(row=2, column=5, padx=5, sticky=tk.S)
            

        self.nextNeighbors=[[0 for col in range(len(X))] for row in range(len(X))]
        ###precalculate next neighbors
        if len(X)>30:
            kMax=30
        else:
            kMax=len(X)-1
        for i in range(1, kMax):    #int(len(X)/2) TODO: welche Zahl?
            temp = kneighbors_graph(X,i, mode="connectivity").toarray()
            for j in range(len(X)):
                for k in range(len(X)):
                    if self.nextNeighbors[j][k]==0 and temp[j][k]==1:
                        self.nextNeighbors[j][k]=i
        ###precalculate distances
        self.distances=kneighbors_graph(X, len(X)-1, mode="distance").toarray()
        eMax=np.amax(self.distances)
        temp=[[0 for col in range(len(self.distances))] for row in range(len(self.distances))]
        for i in range(len(self.distances)):
            for j in range(len(self.distances[i])):
                temp[i][j]=[self.distances[i][j],j]
        for i in range(len(temp)):
            temp[i]=sorted(temp[i], key=lambda x: x[0], reverse=False)
        self.distances=np.array(temp)

        ###precalculation for gaussian kernel
        #self.gaussianDistances=pairwise.rbf_kernel(X, Y=X)
        #temp=[[0 for col in range(len(self.gaussianDistances))] for row in range(len(self.gaussianDistances))]
        #for i in range(len(self.gaussianDistances)):
        #    for j in range(len(self.gaussianDistances[i])):
        #        temp[i][j]=[self.gaussianDistances[i][j],j]
        #for i in range(len(temp)):
        #    temp[i]=sorted(temp[i], key=lambda x: x[0], reverse=False)


        #self.gaussianDistances=np.array(temp)


        admatrix=[[0 for col in range(len(X))] for row in range(len(X))]
        for i in range(len(X)):
            for j in range(len(X)):
                if self.nextNeighbors[i][j]<=k_neighbors and self.nextNeighbors[i][j]!=0:
                    dist = minkowski(X[i],X[j]) #math.sqrt((X[i,0] - X[j,0]) ** 2 + (X[i,0] - X[j,0]) ** 2)
                    admatrix[i][j] = dist
        admatrix=np.array(admatrix)
        np.around(admatrix, decimals=3)

        #admatrix = kneighbors_graph(X,k_neighbors, mode="distance").toarray()
        
        #self.symmetrize()
        linesMatrix = [[0 for col in range(len(admatrix[0]))] for row in range(len(admatrix))]
        laplacian, diag = csgraph.laplacian(admatrix, normed=False, use_out_degree=True, return_diag=True)
        #print(diag)
        #toolbar.destroy()
        fig.patch.set_facecolor(color_bg)
        fig.tight_layout()
        subplotGraph = fig.add_subplot(111)
        subplotGraph.set_facecolor(color_bg)
        

        
        for i in range(0, len(admatrix)):
            for j in range(0, len(admatrix[i])):
                if admatrix[i][j] != 0:
                    if linesMatrix[j][i] == 0:
                        tup = (X[i,dim1], X[j,dim1])
                        tup2 = (X[i,dim2], X[j,dim2])
                        self.lines.append(subplotGraph.plot(tup,tup2, color_lines, zorder=0, gid=lineLastID).pop(0))
                        linesMatrix[i][j] = lineLastID
                        lineLastID+=1

        fig.canvas = FigureCanvasTkAgg(fig, self.tab1)  #master=root)
        fig.canvas._tkcanvas.grid(row=0,column=0, columnspan=5, rowspan=5, padx=5)
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)

        fig2 = plt.figure(figsize=(3.33,2))
        fig2.patch.set_facecolor(color_bg)
        fig2.tight_layout()
        self.canvas2 = FigureCanvasTkAgg(fig2, self.frameHeatmap)  #master=root)
        self.canvas2._tkcanvas.grid(row=1, column=0, columnspan=2, sticky=tk.S, padx=5)

        ###############Clustering

        self.clustering = SpectralClustering(n_clusters=3, affinity="precomputed", assign_labels="kmeans", random_state=0, n_components=numb_eig,  l=laplacian).fit(admatrix) #n_components=numb_eig, assign_labels="discretize"
        self.scatter = subplotGraph.scatter(X[:,dim1], X[:,dim2], c=self.clustering.labels_, cmap='brg', zorder=3) 

        #update_Lines(1)
        #scatter = subplotGraph.scatter(X[:,0], X[:,1], color=color_dots,
        #zorder=3)
        if markedDataPoint != -1:
            subplotGraph.scatter(X[markedDataPoint][dim1], X[markedDataPoint][dim2], c=color_marked, zorder=4)


        ####create Heatmap

        ax = sns.heatmap(np.real(laplacian), linewidth=0.0, linecolor="white", cmap="RdYlBu", robust=True, cbar_kws={"orientation": "horizontal"}) #YlGnBu
        ax.patch.set_facecolor(color_bg)
        plt.tight_layout()

         ###Toolbar
        toolbarFrame = ttk.Frame(self.tab1, style="My.TFrame")
        toolbarFrame.grid(row=5,column=0, columnspan=3, sticky=tk.W, padx=10, pady=10)
        toolbar = NavigationToolbar2Tk(fig.canvas, toolbarFrame)    #Agg
        toolbar.config(background=color_bg)
        toolbar._message_label.config(background=color_bg, foreground=color_text)

        toolbar.update()

        #####Events
        #fig.canvas.mpl_connect("motion_notify_event", self.hover)

        ######Eigenvektor Graph
        
        self.subplotEigV = self.figEigVal.add_subplot(111)
        self.subplotEigV.set_facecolor(color_bg)
        

        eigvalues = sorted(np.linalg.eigvals(admatrix), reverse=True)
        eigvals, eigvecs = la.eig(admatrix)   #sorted, key= lambda x: x[0], reverse=True)

        
        self.subplotEigV.scatter(range(0, len(eigvalues)), eigvalues, s=3, color=color_dots, zorder=3)
        #markedEig= subplotEigV.scatter(range(0,numb_eig+1),
        #eigvalues[:numb_eig+1], s=3, color="red", zorder=4)

        self.eigVLine = self.subplotEigV.plot((numb_eig - 0.5,numb_eig - 0.5),(eigvalues[0] + 0.4,eigvalues[len(eigvalues) - 1] - 0.2), color=color_marked, zorder=1).pop(0)

        self.markedEig = self.subplotEigV.scatter(range(0, numb_eig), eigvalues[:numb_eig], s=3, color=color_marked, zorder=4)            
        #figEigVal.canvas.draw()

        self.canvasEigV = FigureCanvasTkAgg(self.figEigVal, self.frameEigenG)  #master=root)
        self.figEigVal.tight_layout()
        self.canvasEigV._tkcanvas.grid(row=1,column=0, columnspan=3, padx=5, sticky=tk.W)
        lab2=tk.Label(self.frameEigenG, text="Dimensions of Eigenvectores:", bg=color_bg, fg=color_text)
        lab2.grid(row=0, column=0, )
        self.labDim=tk.Label(self.frameEigenG, text=numb_eig, bg=color_bg, fg=color_text)
        self.labDim.grid(row=0, column=0, sticky=tk.E)
        #self.markedEig.remove()
        self.subplotEigV.format_coord = lambda x, y: ""

        toolbarFrameEV = ttk.Frame(self.frameEigenG, style="My.TFrame")
        toolbarFrameEV.grid(row=0,column=1, sticky=tk.E, padx=10, pady=10)
        toolbarEV = NavigationToolbar(self.figEigVal.canvas, toolbarFrameEV)    #Agg
        toolbarEV.config(background=color_bg)
        toolbarEV._message_label.config(background=color_bg, foreground=color_text)
        toolbarEV.update()
        #toolbarEV.remove_tool('forward')
        #print("Help me")
        #print(toolbarEV.toolmanager.get_tool(_views_positions))
        

        #toolbarEV.toolitems = filter(lambda x: x[0] != "Subplots", NavigationToolbar2Tk.toolitems)
        
        
        countEigV = int(self.eigVLine.get_xdata()[0]) + 1
        idx = eigvals.argsort()#[-countEigV:][::-1] 
        eigvals = -eigvals[idx]
        eigvecs = -eigvecs[idx]
        self.vecMat = np.ndarray.transpose(eigvecs)

        ########Hover Box
        self.annot = subplotGraph.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                    bbox=dict(boxstyle="circle", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        self.annot.set_visible(False)

        #########IndicationLine
        self.indicatorLine = subplotGraph.plot((0,0),(0,0), color=(0,0,0,0.5), zorder=2).pop(0)
        self.indicatorLine.set_linestyle('')
        
        self.figEigVal.canvas.mpl_connect('button_press_event', self.clickEigenG)
        self.figEigVal.canvas.mpl_connect("motion_notify_event", self.followmouse)
        fig.canvas.mpl_connect("motion_notify_event", self.hover)
        cid = fig.canvas.mpl_connect('button_press_event', self.onclick)
        self.figEigVal.canvas.mpl_connect("motion_notify_event", self.followmouse)
        self.figEigVal.canvas.mpl_connect('button_press_event', self.clickEigenG)


        ######Eigenvektor Slider
        #sliderEigenV = tk.Scale(self, from_=0, to=10, label='Eigenvectores',
        #bg=color_bg, orient=tk.HORIZONTAL)
        #sliderEigenV.grid(row=2, column=4, padx=5)

        #########Eigenvector tab

        self.figvecs = plt.figure(figsize=(6,5.5), dpi=100)
        self.figvecs.patch.set_facecolor(color_bg)
        self.subplotGraphVecs = self.figvecs.add_subplot(111)
        self.subplotGraphVecs.set_facecolor(color_bg)

        self.scatterVecs, =  self.subplotGraphVecs.plot(self.vecMat[:,0], self.vecMat[:,1],color=color_dots,marker='o',ls='', zorder=3)#self.subplotGraphVecs.scatter(self.vecMat[:,0], self.vecMat[:,1], c=color_dots, zorder=3) 

        self.canvasVecs = FigureCanvasTkAgg(self.figvecs, self.tab2) 
        self.figvecs.tight_layout()
        self.canvasVecs._tkcanvas.grid(row=0,column=0, padx=5, rowspan=5)

        ###Toolbar
        toolbarFrame = ttk.Frame(self.tab2, style="My.TFrame")
        toolbarFrame.grid(row=7,column=0, columnspan=2, sticky=tk.W, padx=10, pady=10)
        toolbar = NavigationToolbar2Tk(self.canvasVecs, toolbarFrame)    #Agg
        toolbar.config(background=color_bg)
        toolbar._message_label.config(background=color_bg, foreground=color_text)
        toolbar.update()


        ######PCA Tab
        self.pca=dec.PCA(n_components=2).fit_transform(X)

        self.figpca = plt.figure(figsize=(6,5.5), dpi=100)
        self.figpca.patch.set_facecolor(color_bg)
        self.subplotGraphpca = self.figpca.add_subplot(111)
        self.subplotGraphpca.set_facecolor(color_bg)

        self.scatterpca = self.subplotGraphpca.scatter(self.pca[:,0], self.pca[:,1], c=self.clustering.labels_, cmap='brg', zorder=3) 
        #self.scatterpca, =  self.subplotGraphpca.plot(pca[:,0], pca[:,1],color=color_dots,marker='o',ls='', zorder=3) 

        self.canvaspca = FigureCanvasTkAgg(self.figpca, self.tab3) 
        self.figpca.tight_layout()
        self.canvaspca._tkcanvas.grid(row=0,column=0, padx=5, rowspan=5)

        self.figpca.canvas.mpl_connect("motion_notify_event", self.hoverpca)

        ########Hover Box PCA
        self.annotpca = self.subplotGraphpca.annotate("", xy=(0,0), xytext=(10,10), textcoords="offset points",
                    bbox=dict(boxstyle="circle", fc="w"),
                    arrowprops=dict(arrowstyle="->"))
        self.annotpca.set_visible(False)

        ###Toolbar
        toolbarFrame = ttk.Frame(self.tab3, style="My.TFrame")
        toolbarFrame.grid(row=7,column=0, columnspan=2, sticky=tk.W, padx=10, pady=10)
        toolbar = NavigationToolbar2Tk(self.canvaspca, toolbarFrame)    #Agg
        toolbar.config(background=color_bg)
        toolbar._message_label.config(background=color_bg, foreground=color_text)
        toolbar.update()

        self.customLines=[]

        if self.chooseGraph is not None and self.chooseMatrix is not None:
            self.chooseGraph.set("kNN")
            self.chooseMatrix.set("Laplacian")
            self.sliderE.configure(to=eMax)

        


    
    ###############################Logical
    def findClosest(self, event):
        global  X
        cont, ind = self.scatter.contains(event)

        if cont:
            msx, msy = event.xdata, event.ydata
            datax,datay = X[0, 0],X[0, 1]
            mindist = math.sqrt((datax - msx) ** 2 + (datay - msy) ** 2)

            temp = ind["ind"][0]
            for i in ind["ind"]:
                msx, msy = event.xdata, event.ydata
                datax,datay = X[i, 0],X[i, 1]
                dist = math.sqrt((datax - msx) ** 2 + (datay - msy) ** 2)
                if dist < mindist:
                    mindist = dist
                    temp = i
            return temp
        return None

    def findClosestpca(self, event):
        global  X
        cont, ind = self.scatterpca.contains(event)

        if cont:
            msx, msy = event.xdata, event.ydata
            datax,datay = X[0, 0],X[0, 1]
            mindist = math.sqrt((datax - msx) ** 2 + (datay - msy) ** 2)

            temp = ind["ind"][0]
            for i in ind["ind"]:
                msx, msy = event.xdata, event.ydata
                datax,datay = X[i, 0],X[i, 1]
                dist = math.sqrt((datax - msx) ** 2 + (datay - msy) ** 2)
                if dist < mindist:
                    mindist = dist
                    temp = i
            return temp
        return None

    ####called when custom line is added or deleted, updates admatrix
    def customizeGraph(self, point=-1):
        global markedDataPoint, admatrix, linesMatrix, fig, lineLastID
        if point != -1 and markedDataPoint != -1:
            if admatrix[markedDataPoint][point] == 0 and admatrix[point][markedDataPoint] == 0 and markedDataPoint != point:
                dist = minkowski(X[markedDataPoint],X[point],2)   #math.sqrt((X[markedDataPoint,0] - X[point,0]) ** 2 + (X[markedDataPoint,0] - X[point,0]) ** 2)
                admatrix[markedDataPoint][point] = dist
                admatrix[point][markedDataPoint] = dist
                self.updateLaplacian(add=True, a=markedDataPoint, b=point, dist=dist)

                    
            else:
                admatrix[markedDataPoint][point] = 0
                admatrix[point][markedDataPoint] = 0
                self.updateLaplacian(add=False, a=markedDataPoint, b=point, dist=0)

    def updateLaplacian(self, add, a, b, dist):
        global laplacian
        Mtype=self.chooseMatrix.get()
        if Mtype!="alt Laplacian" and Mtype!="Laplacian":
            self.calcLaplacian()
        else:
            diagonal=1
            if add is False:
                dist=0

                if Mtype=="alt Laplacian":
                    diagonal=-1
                else:
                    diagonal=-laplacian[a][b]
            else:
                if Mtype=="Laplacian":
                    diagonal=laplacian[a][b]
            laplacian[a][b]=dist
            laplacian[b][a]=dist
            laplacian[a][a]=laplacian[a][a]+diagonal
            laplacian[b][b]=laplacian[b][b]+diagonal


    def symmetrize(self):
        global admatrix
        for i in range(0, len(admatrix)):
            for j in range(0, len(admatrix[i])):
                if admatrix[i][j] != 0 and admatrix[j][i] == 0:
                    admatrix[j][i] == admatrix[i][j]

    ####deletes Line
    def deleteLine(self,i,j):
        global linesMatrix, fig
        if linesMatrix[i][j] != 0:
            x = linesMatrix[i][j]
            linesMatrix[i][j] = 0
            for k in range(0, len(self.lines)):
                if self.lines[k].get_gid() == x:
                    self.lines.pop(k).remove()
                    break
            #if (i,j) in self.customLines:
            #    self.customLines.remove((i,j))
        if linesMatrix[j][i] != 0:
            x = linesMatrix[j][i]
            linesMatrix[j][i] = 0
            for k in range(0, len(self.lines)):
                if self.lines[k].get_gid() == x:
                    self.lines.pop(k).remove()
                    break
            #if (j,i) in self.customLines:
            #    self.customLines.remove((j,i))

    #####updateData when scales are changed
    def updateData(self, mode, value=None):
        global admatrix, gamma, k_neighbors, epsilon
        if mode == "k_neighbors":
            k_neighbors = value
            if(k_neighbors>30):
                admatrix = kneighbors_graph(X,k_neighbors, mode="distance").toarray()
            else:
                admatrix=[[0 for col in range(len(X))] for row in range(len(X))]
                for i in range(len(X)):
                    for j in range(len(X)):
                        if self.nextNeighbors[i][j]<=k_neighbors and self.nextNeighbors[i][j]!=0:
                            dist = minkowski(X[i],X[j])
                            admatrix[i][j] = dist
                admatrix=np.array(admatrix)
                np.around(admatrix, decimals=3)

        if mode == "epsilon":
            epsilon = value
            admatrix=[[0 for col in range(len(X))] for row in range(len(X))]
            for i in range(len(X)):
                    for j in range(len(X)):
                        if self.distances[i][j][0]<=epsilon:
                            admatrix[i][int(self.distances[i][j][1])]=self.distances[i][j][0]
                        else:
                            break
            admatrix=np.array(admatrix)

            #admatrix = radius_neighbors_graph(X, radius=epsilon).toarray()
        if mode == "gaussian":
            gamma = value
            ###precalculation for gaussian kernel
            admatrix=pairwise.rbf_kernel(X, Y=X, gamma=gamma)
            mean=np.mean(admatrix)
            for i in range(len(X)):
                    for j in range(len(X)):
                        if admatrix[i][j]>=mean:
                            admatrix[i][j]=0

            admatrix=np.array(admatrix)
        if mode == "complete":
            admatrix = kneighbors_graph(X,samples - 1, mode="distance").toarray()
        if keepCustom.get():
            for i in range(0, len(self.customLines)):
                a, b = self.customLines[i]
                dist = minkowski(X[a],X[b], 2)     #math.sqrt((X[a,0] - X[b,0]) ** 2 + (X[a,0] - X[b,0]) ** 2)
                admatrix[a][b] = dist
                admatrix[b][a] = dist
            for i in range(0, len(self.deletedLines)):
                a, b = self.deletedLines[i]
                admatrix[a][b]=0
                admatrix[b][a]=0
        #self.symmetrize()
        self.recluster()

    ##Calculate new EigenVectores, after Laplace Matrix changed
    def calcEigVecs(self):
        global laplacian
        eigvals, eigvecs = la.eig(laplacian)   #sorted, key= lambda x: x[0], reverse=True)
        #eigvals=la.eig(admatrix)
        countEigV = int(self.eigVLine.get_xdata()[0]) + 1
        idx = eigvals.argsort()[-countEigV:][::-1] 
        eigvals = eigvals[idx]
        eigvecs = eigvecs[idx]
        self.vecMat = np.ndarray.transpose(eigvecs)


    def calcLaplacian(self):
        global laplacian
        if self.chooseMatrix.get() == 'Laplacian':
            #eigvalues = sorted(np.linalg.eigvals(admatrix), reverse=True)
            laplacian = csgraph_laplacian(admatrix, normed=False)
            #laplacian = csr_matrix.todense(laplacian)
        if self.chooseMatrix.get() == 'alt Laplacian':
            laplacian = csgraph_laplacian(admatrix, normed=False)
            for i in range(len(laplacian)):
                temp = 0
                for j in range(len(laplacian[i])):
                    if i != j and laplacian[i][j] != 0:
                        temp = temp + 1
                laplacian[i][i] = temp
        if self.chooseMatrix.get() == 'Lsym':
            #laplacian = nx.normalized_laplacian_matrix(admatrix)
            laplacian = csgraph_laplacian(admatrix, normed=True)
        if self.chooseMatrix.get() == 'Lrw':
            laplacian = nx.directed_laplacian_matrix(from_numpy_matrix(np.real(admatrix)).to_directed(), walk_type='random')

    def change_vert(self, *args):
        self.scatterVecs.set_ydata(self.vecMat[:,int(self.vert.get())])
        #self.subplotGraphVecs.autoscale(enable=True)
        plt.figure(self.figvecs.number)
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view()
        self.figvecs.canvas.draw()
        plt.draw()
        fig.canvas.flush_events()
        

    def change_hor(self, *args):
        self.scatterVecs.set_xdata(self.vecMat[:,int(self.hor.get())])
        #self.subplotGraphVecs.autoscale(enable=True)
        plt.figure(self.figvecs.number)
        ax = plt.gca()
        ax.relim()
        ax.autoscale_view()
        self.figvecs.canvas.draw()
        plt.draw()
        fig.canvas.flush_events()

    def change_sliderDimension1(self, *args):
        global fig
        self.scatter.remove()
        self.scatter = subplotGraph.scatter(X[:,self.sliderDimension1.get()-1], X[:,self.sliderDimension2.get()-1], c=self.clustering.labels_, cmap='brg', zorder=3) 
        self.update_Lines(changeDim=True)

        fig.canvas.draw()
        plt.draw()
        fig.canvas.flush_events()

    #def switchDisplay(self, *args):
    #    global X, Xorig
    #    if self.variableDisplay.get()==2:
    #        #X=Xorig
    #        self.sliderDimension1.grid(row=1, column=5, padx=5)
    #        self.sliderDimension2.grid(row=1, column=5, padx=5, sticky=tk.S)
    #    if self.variableDisplay.get()==1:
    #        #Xorig=X
    #        #X=dec.PCA(n_components=2).fit_transform(X)
    #        self.sliderDimension1.grid_remove()
    #        self.sliderDimension2.grid_remove()
    #        #self.setup()


    ######################Drawing

    def recluster(self):
        global numb_eig, laplacian, n_clusters, fig

        dim1=0
        dim2=1
        if len(X[0])>2:
            dim1=self.sliderDimension1.get()-1
            dim2=self.sliderDimension2.get()-1

        #self.calcLaplacian()
        self.redraw_heatmap()

        self.calcEigVecs()
        self.change_vert()
        self.change_hor()

        self.clustering = SpectralClustering(n_clusters=n_clusters, affinity="precomputed", assign_labels="kmeans", random_state=0, n_components=numb_eig, l=laplacian).fit(admatrix) # n_components=numb_eig,
        #scatter.set_color(clustering.labels_)
        self.scatter.remove()
        self.scatter = subplotGraph.scatter(X[:,dim1], X[:,dim2], c=self.clustering.labels_, cmap='brg', zorder=3) 
        self.scatterpca.remove()
        self.scatterpca = self.subplotGraphpca.scatter(self.pca[:,0], self.pca[:,1], c=self.clustering.labels_, cmap='brg', zorder=3) 


        fig.canvas.draw()
        fig.canvas.flush_events()

    def redraw(self):

        fig.clear()
        fig2.clear()
        self.figEigVal.clear()
        self.figvecs.clear()

        self.setup()

        fig.canvas.draw()
        fig.canvas.flush_events()
        fig2.canvas.draw()
        fig2.canvas.flush_events()

    ####redraw the Heatmap
    def redraw_heatmap(self):
        global laplacian, fig2, admatrix
        fig2.clear()
        plt.figure(fig2.number)
        
        ax = sns.heatmap(np.real(laplacian), linewidth=0.0, linecolor="white", cmap="RdYlBu", robust=True, cbar_kws={"orientation": "horizontal"}) #YlGnBu

        ax.patch.set_facecolor(color_bg)
        plt.tight_layout()
             
        self.canvas2 = FigureCanvasTkAgg(fig2, self.frameHeatmap)  #master=root)
        self.canvas2._tkcanvas.grid(row=1, column=0, columnspan=2, sticky=tk.S, padx=5)

    ###marks point
    def mark_point(self, x=0, delete=False):
        global fig, markedPoint, color_marked

        dim1=0
        dim2=1
        if len(X[0])>2:
                dim1=self.sliderDimension1.get()-1
                dim2=self.sliderDimension2.get()-1

        if delete == False:
            markedPoint = subplotGraph.scatter(X[x][dim1], X[x][dim2], s=55, color=(0,0,0,0.3), edgecolors="black", zorder=4)    
        else:
            if markedPoint is not None:
                markedPoint.remove()

    ###draws and deletes lines, custom is True, if changed Line is custom
    def update_Lines(self, custom=False, changeDim=False):
        global fig, markedDataPoint, admatrix, linesMatrix, color_lines, lineLastID, lines, keepCustom

        dim1=0
        dim2=1
        if len(X[0])>2:
            dim1=self.sliderDimension1.get()-1
            dim2=self.sliderDimension2.get()-1


        for i in range(0, len(admatrix)):
            for j in range(0, len(admatrix[i])):

                if changeDim and i<j:
                    self.deleteLine(i,j)
                    #self.deleteLine(j,i)
                if admatrix[i][j] != 0 and linesMatrix[i][j] == 0:
                    if linesMatrix[j][i] == 0:

                        tup = (X[i,dim1], X[j,dim1])
                        tup2 = (X[i,dim2], X[j,dim2])
                        color = color_lines
                        if custom or (changeDim and ((i,j) in self.customLines)or ((j,i) in self.customLines)):
                            color = color_custom
                        self.lines.append(subplotGraph.plot(tup,tup2, color=color, zorder=1, gid=lineLastID).pop(0))
                        linesMatrix[i][j] = lineLastID
                        #linesMatrix[j][i] = lineLastID
                        if custom and i!=j:
                            self.customLines.append((i,j))
                            if (i,j) in self.deletedLines:
                                self.deletedLines.remove((i,j))
                            if (j,i) in self.deletedLines:
                                self.deletedLines.remove((j,i))
                        lineLastID+=1
                elif admatrix[i][j] == 0 and linesMatrix[i][j] != 0:
                    if custom:
                        self.deletedLines.append((i,j))
                        if (i,j) in self.customLines:
                            self.customLines.remove((i,j))
                        if (j,i) in self.customLines:
                            self.customLines.remove((j,i))
                    self.deleteLine(i,j)
            
        fig.canvas.draw()
        fig.canvas.flush_events()

    #######Events
    
    ####when Graph is clicked
    def onclick(self, event):
        global markedDataPoint

        closest = self.findClosest(event)
        if closest is not None:
            #if event.xdata<X[i][0]+click_radius and
            #event.xdata>X[i][0]-click_radius and event.ydata<X[i][1] and
            #event.ydata>X[i][1]-click_radius:
            if markedDataPoint == -1:
                #updateData(markPoint=i)
                markedDataPoint = closest
                self.mark_point(closest)
                #update_Lines(i)
            else:
                self.customizeGraph(point=closest)
                self.mark_point(delete=True)
                self.update_Lines(custom=True)
                markedDataPoint = -1
                self.recluster()
        else:
            if markedDataPoint != -1:
                self.mark_point(delete=True)
                markedDataPoint = -1
                self.recluster()

        self.indicatorLine.set_linestyle('')


        fig.canvas.draw()
        fig.canvas.flush_events()

    def hover(self, event):
        global fig
        vis = self.annot.get_visible()
        closest = self.findClosest(event)
        if closest is not None:
            #if hoveredPoint is None:
            #    hoveredPoint = subplotGraph.scatter(X[closest][0],
            #    X[closest][1], color=(1,1,1,0), edgecolors="black", zorder=5)
            self.update_annot(closest)
            self.annot.set_visible(True)
            fig.canvas.draw_idle()
        else:
            #if hoveredPoint is not None:
            #    hoveredPoint.remove()
            #    hoveredPoint=None
            if vis:
                self.annot.set_visible(False)
                fig.canvas.draw_idle()
        if markedDataPoint != -1:
            self.indicateLine(closest)

    def hoverpca(self, event):
        vis = self.annotpca.get_visible()
        closest = self.findClosestpca(event)
        if closest is not None:
            self.update_annot(closest, graph="pca")
            self.annotpca.set_visible(True)
            self.figpca.canvas.draw_idle()
        else:
            if vis:
                self.annotpca.set_visible(False)
                self.figpca.canvas.draw_idle()


    def update_annot(self, ind, graph="main"):
        if graph=="main":
            pos = self.scatter.get_offsets()[ind]
            annot=self.annot
        else:
            pos = self.scatterpca.get_offsets()[ind]
            annot=self.annotpca
        annot.xy = pos
        text = ind#"{}".format(" ".join(list(map(str,ind))))
        annot.set_text(text)
        annot.set_size(7)
        #annot.get_bbox_patch().set_facecolor(cmap(norm(c[ind["ind"][0]])))
        annot.get_bbox_patch().set_alpha(0.7)

    ###indicates Line drawn / deleted, when point is marked and second point is
    ###hovered
    def indicateLine(self, closest):
        global markedDataPoint, linesMatrix

        dim1=0
        dim2=1
        if len(X[0])>2:
            dim1=self.sliderDimension1.get()-1
            dim2=self.sliderDimension2.get()-1

        if closest is not None:
            tup = (X[markedDataPoint, dim1], X[closest, dim1])
            tup2 = (X[markedDataPoint, dim2], X[closest, dim2])
            self.indicatorLine.set_data(tup, tup2)
            if linesMatrix[markedDataPoint][closest] == 0 and linesMatrix[closest][markedDataPoint] == 0:
                self.indicatorLine.set_color((0.5,0.5,0.5,0.3))
            else:
                self.indicatorLine.set_color((0,0,0,0.5))
            self.indicatorLine.set_linestyle('solid')
        else:
            self.indicatorLine.set_linestyle('')

    def clickEigenG(self, event):
        global numb_eig, zoomEigsActive
        #markedEig= subplotEigV.scatter(range(0,numb_eig+1),
        #eigvalues[:numb_eig+1], s=3, color="red", zorder=4)
        if not zoomEigsActive:
            self.eigVLine.set_xdata([event.xdata, event.xdata])
            temp = int(event.xdata)
            if numb_eig != temp and temp>0:
                self.markedEig.remove()
                self.markedEig = self.subplotEigV.scatter(range(0,temp + 1), eigvalues[:temp + 1], s=3, color=color_marked, zorder=4)
                numb_eig = temp
                self.labDim.config(text=numb_eig+1)
                self.figEigVal.canvas.draw()
                #self.markedEig.remove()
                self.recluster()


            


    def followmouse(self, event):
        global numb_eig, zoomEigsActive
        if event.button == 1 and event.xdata is not None and zoomEigsActive is not True:
            self.eigVLine.set_xdata([event.xdata, event.xdata])
            #figEigVal.canvas.draw()
            temp = int(event.xdata)
            if numb_eig != temp and temp > 0:
                self.markedEig.remove()
                self.markedEig = self.subplotEigV.scatter(range(0,temp + 1), eigvalues[:temp + 1], s=3, color=color_marked, zorder=4)
                    
                self.figEigVal.canvas.draw()
                #self.markedEig.remove()
            numb_eig = temp
            self.labDim.config(text=numb_eig+1)

       
            
           
            

class LoadData(tk.Frame):

    def __init__(self, master, main):
        #self.master = master
        master.grid_propagate(False)
        master.geometry("400x380")
        self.filename = None

        def switchLoad():
            #.grid_remove()
            if v.get() == 1:
                frameGenerate.grid_remove()
                frameLoad.grid(row=1, column=0, columnspan=2)
            else:
                frameLoad.grid_remove()
                frameGenerate.grid(row=1, column=0, columnspan=2)

        def load():
            self.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("csv files","*.csv"),("all files","*.*")))
            tv.set(self.filename)
            master.lift()

        def loadSett():
            self.filename = filedialog.askopenfilename(initialdir = "/",title = "Select file",filetypes = (("txt files","*.txt"),("all files","*.*")))
            varSett.set(self.filename)
            master.lift()

        v = tk.IntVar()
        v.set(1)
        tk.Radiobutton(master, text="Load", padx = 5, variable=v, value=1, command=switchLoad).grid(row=0, column =0, sticky=tk.E)
        tk.Radiobutton(master, text="Create",padx = 5, variable=v, value=2, command=switchLoad).grid(row=0, column=1, sticky=tk.W)
        

        frameGenerate = tk.Frame(master, padx=10, pady=10)     #self.master
        labelSamples = tk.Label(frameGenerate, padx=10, text="Samples:")
        samples = tk.Entry(frameGenerate, width=4)
        labelCenters = tk.Label(frameGenerate, padx=10, text="Centers:")
        centers = tk.Scale(frameGenerate, from_=0, to=10, orient=tk.HORIZONTAL)
        labelstd = tk.Label(frameGenerate, padx=10, text="Standard Derivation:")
        std = tk.Scale(frameGenerate, from_=0, to=10,  resolution=0.1, orient=tk.HORIZONTAL)
        labelBox = tk.Label(frameGenerate, padx=10, text="Min and Max for centers:")
        boxMin = tk.Entry(frameGenerate, width=4)
        boxMax = tk.Entry(frameGenerate, width=4)

        labelSamples.grid(row=0, column=0, pady=10, sticky=tk.W)
        samples.grid(row=0, column=1, pady=10, sticky=tk.W)
        labelCenters.grid(row=1, column=0, pady=10, sticky=tk.W)
        centers.grid(row=1, column=1, pady=10, sticky=tk.E)
        labelstd.grid(row=2, column=0, pady=10, sticky=tk.W)
        std.grid(row=2, column=1, pady=10, sticky=tk.E)
        labelBox.grid(row=3, column=0, pady=10, sticky=tk.W)
        boxMin.grid(row=3, column=1, pady=10, sticky=tk.W)
        boxMax.grid(row=3, column=2, pady=10, sticky=tk.W)

        tv = tk.StringVar()
        frameLoad = tk.Frame(master, padx=10, pady=10)
        #frameLoad.grid_geometry("400x100")
        pathLabel = tk.Label(frameLoad, text="Open File (CSV):")
        entryPath = tk.Entry(frameLoad, textvariable=tv, width=48)
        buttonLoad = tk.Button(frameLoad, text="Search..",  command=load)
        buttonLoad.grid(row=2, column=1, pady=10, sticky=tk.E)
        pathLabel.grid(row=0, column=1, pady=10, sticky=tk.W)
        entryPath.grid(row=1, column=1, pady=10, sticky=tk.W)
        varLabeled = tk.IntVar()
        ttk.Checkbutton(frameLoad, text="Labeled", variable=varLabeled).grid(row=2, column=1, sticky=tk.W)

        varSett=tk.StringVar()
        settLabel = tk.Label(frameLoad, text="Use Settings (txt, optional):").grid(row=3, column=1, pady=10, sticky=tk.W)
        entrySett = tk.Entry(frameLoad, textvariable=varSett, width=48).grid(row=4, column=1, pady=10, sticky=tk.W)
        buttonLoad2 = tk.Button(frameLoad, text="Search..",  command=loadSett).grid(row=5, column=1, pady=10, sticky=tk.E)


        def use():
            global X, y, admatrix, linesMatrix, laplacian, Xorig
            if checkUse():
                if v.get() == 2:
                    X, y = ds.make_blobs(n_samples=int(samples.get()), centers=centers.get(), cluster_std=std.get(), center_box=(int(boxMin.get()),int(boxMax.get())))
                else: 
                    X = read(tv.get())
                    Xorig=X
                #admatrix = kneighbors_graph(X,k_neighbors, mode="distance").toarray()
                #linesMatrix = [[0 for col in range(len(admatrix[0]))] for row in range(len(admatrix))]
                #laplacian = csgraph.laplacian(admatrix, normed=False)
                PageThree.redraw(main)
                if varSett.get() is not None:
                        PageThree.restore(main, i=0, file=varSett.get())
                master.destroy()
            else:
                tk.Label(master, text="Check Input", fg="red").grid(row=2, column=0, sticky=tk.E)
            

            

        def checkUse(): 
            check = True
            if tv.get()=="test":
                test()
            if v.get() == 1:
                if self.filename is None:
                    check = False
                else:
                    check = os.path.isfile(self.filename)  #self.filename.isfile()
            else:
                if samples.get() is "":
                    check = False
                else:
                    if not samples.get().isdigit():
                        check = False
                if centers == 0:
                    check = False
                if std == 0:
                    check = False
                #if boxMin.get() is
                try:
                    if boxMin.get is "":
                        check = False
                    else:
                        if "." in boxMin.get() :
                            val = float(boxMin.get())
                        else:
                            val = int(boxMin.get())
                    if boxMax.get is "":
                        check = False
                    else:
                        if "." in boxMax.get() :
                            val = float(boxMax.get())
                        else:
                            val = int(boxMax.get())
                except ValueError:
                    check = False

            return check


        switchLoad()
        self.useButton = tk.Button(master, text="Use", command=use)
        self.useButton.grid(row=2, column=0, columnspan=2, sticky=tk.E, padx=10, pady=10)

        
        def read(filename):
            global labels
            if varLabeled.get() == 0:
                with open(filename, 'r') as csvfile:
                    reader = csv.reader(csvfile, delimiter= ',')
                    linecount = 0
                    temp = []
                    #temp=np.array()
                    reader = csv.reader(csvfile, delimiter= ',')
                    for row in reader:
                        rowfloat = [float(i) for i in row]
                        temp.append(rowfloat)
                        #temp[linecount]=rowfloat
                        #temp[linecount][0]=rowfloat[0]
                        #temp[linecount][1]=rowfloat[1]
                        linecount+=1
                    temp = np.array(temp)
            else:
                
                n = sum(1 for line in open(filename)) - 1 #number of records in file (excludes header)
                if n>150:
                    s = 150 #desired sample size
                    skip = sorted(random.sample(range(1,n+1),n-s)) #the 0-indexed header will not be included in the skip list
                    df = pandas.read_csv(filename, skiprows=skip)
                else:

                    df = pandas.read_csv(filename)
                ##header=list(df.columns.values)
                ##header=header.remove("class")
                df1 = df.drop(['class'], axis=1)
                temp= df1.values
                labels=np.asarray(df["class"].tolist())

            return temp

        def test():
            global X, y
            f= open("C:\\Users\\Saturn\\Documents\\Uni\\bachelorarbeit\\LogTest2.txt","w+")
            for i in range(1,12):
                sampleSize=100*i
                if i==11:
                    sampleSize=5000
                f.write("sample Size: "+str(sampleSize)+"\n")
                start=time.time()
                X, y = ds.make_blobs(n_samples=sampleSize, centers=3)
                PageThree.redraw(main)
                end=time.time()
                f.write("Time: "+str(end-start)+"\n")
                f.write("-------------\n")

            print("Done")
            f.close()







#################SpectralClustering Algorithms from scikit-learn
# Author: Gael Varoquaux gael.varoquaux@normalesup.org
#         Brian Cheung
#         Wei LI <kuantkid@gmail.com>
# License: BSD 3 clause
#TODO: Check for redundant calculation
class SpectralClustering(BaseEstimator, ClusterMixin):

    def __init__(self, n_clusters=8, eigen_solver=None, random_state=None,
                 n_init=10, gamma=1., affinity='rbf', n_neighbors=10,
                 eigen_tol=0.0, assign_labels='kmeans', degree=3, coef0=1,
                 kernel_params=None, n_jobs=None,  l=None, n_components=8): #n_components=8,
        self.n_clusters = n_clusters
        self.eigen_solver = eigen_solver
        self.random_state = random_state
        self.n_init = n_init
        self.gamma = gamma
        self.affinity = affinity
        self.n_neighbors = n_neighbors
        self.eigen_tol = eigen_tol
        self.assign_labels = assign_labels
        self.degree = degree
        self.coef0 = coef0
        self.kernel_params = kernel_params
        self.n_jobs = n_jobs
        self.n_components=n_components
        self.l = l

    def fit(self, X, y=None):
        """Creates an affinity matrix for X using the selected affinity,
        then applies spectral clustering to this affinity matrix.
        Parameters
        ----------
        X : array-like or sparse matrix, shape (n_samples, n_features)
            OR, if affinity==`precomputed`, a precomputed affinity
            matrix of shape (n_samples, n_samples)
        y : Ignored
        """
        X = check_array(X, accept_sparse=['csr', 'csc', 'coo'],
                        dtype=np.float64, ensure_min_samples=2)
        if X.shape[0] == X.shape[1] and self.affinity != "precomputed":
            warnings.warn("The spectral clustering API has changed. ``fit``"
                          "now constructs an affinity matrix from data. To use"
                          " a custom affinity matrix, "
                          "set ``affinity=precomputed``.")

        if self.affinity == 'nearest_neighbors':
            connectivity = kneighbors_graph(X, n_neighbors=self.n_neighbors,
                                            include_self=True,
                                            n_jobs=self.n_jobs)
            self.affinity_matrix_ = 0.5 * (connectivity + connectivity.T)
        elif self.affinity == 'precomputed':
            self.affinity_matrix_ = X
        else:
            params = self.kernel_params
            if params is None:
                params = {}
            if not callable(self.affinity):
                params['gamma'] = self.gamma
                params['degree'] = self.degree
                params['coef0'] = self.coef0
            self.affinity_matrix_ = pairwise_kernels(X, metric=self.affinity,
                                                     filter_params=True,
                                                     **params)

        random_state = check_random_state(self.random_state)
        self.labels_ = spectral_clustering(self.affinity_matrix_,
                                           n_clusters=self.n_clusters,
                                           eigen_solver=self.eigen_solver,
                                           n_components=self.n_components,
                                           random_state=random_state,
                                           n_init=self.n_init,
                                           eigen_tol=self.eigen_tol,
                                           assign_labels=self.assign_labels, l=self.l)
        return self

def spectral_clustering(affinity, n_clusters=8, n_components=None,
                        eigen_solver=None, random_state=None, n_init=10,
                        eigen_tol=0.0, assign_labels='kmeans', l=None):
    if assign_labels not in ('kmeans', 'discretize'):
        raise ValueError("The 'assign_labels' parameter should be "
                         "'kmeans' or 'discretize', but '%s' was given" % assign_labels)

    random_state = check_random_state(random_state)
    n_components = n_clusters if n_components is None else n_components

    # The first eigen vector is constant only for fully connected graphs
    # and should be kept for spectral clustering (drop_first = False)
    # See spectral_embedding documentation.
    maps = spectral_embedding(affinity, n_components=n_components,
                              eigen_solver=eigen_solver,
                              random_state=random_state,
                              eigen_tol=eigen_tol, drop_first=False, l=l)

    if assign_labels == 'kmeans':
        _, labels, _ = k_means(maps, n_clusters, random_state=random_state,
                               n_init=n_init)
    else:
        labels = spectral.discretize(maps, random_state=random_state)

    return labels



def spectral_embedding(adjacency, n_components=8, eigen_solver=None,
                       random_state=None, eigen_tol=0.0,
                       norm_laplacian=True, drop_first=False, l=None):
    adjacency = spectral_embedding_.check_symmetric(adjacency)

    try:
        from pyamg import smoothed_aggregation_solver
    except ImportError:
        if eigen_solver == "amg":
            raise ValueError("The eigen_solver was set to 'amg', but pyamg is "
                             "not available.")

    if eigen_solver is None:
        eigen_solver = 'arpack'
    elif eigen_solver not in ('arpack', 'lobpcg', 'amg'):
        raise ValueError("Unknown value for eigen_solver: '%s'."
                         "Should be 'amg', 'arpack', or 'lobpcg'" % eigen_solver)

    random_state = check_random_state(random_state)

    n_nodes = adjacency.shape[0]
    # Whether to drop the first eigenvector
    if drop_first:
        n_components = n_components + 1

    if not spectral_embedding_._graph_is_connected(adjacency):
        warnings.warn("Graph is not fully connected, spectral embedding"
                      " may not work as expected.")

    laplacian, dd = spectral_embedding_.csgraph_laplacian(adjacency, normed=norm_laplacian,
                                      return_diag=True)
    #laplacian=l
    #if l is not None:
    #    lalacian = l
    if (eigen_solver == 'arpack' or eigen_solver != 'lobpcg' and (not sparse.isspmatrix(laplacian) or n_nodes < 5 * n_components)):
        # lobpcg used with eigen_solver='amg' has bugs for low number of nodes
        # for details see the source code in scipy:
        # https://github.com/scipy/scipy/blob/v0.11.0/scipy/sparse/linalg/eigen
        # /lobpcg/lobpcg.py#L237
        # or matlab:
        # https://www.mathworks.com/matlabcentral/fileexchange/48-lobpcg-m
        laplacian = spectral_embedding_._set_diag(laplacian, 1, norm_laplacian)

        # Here we'll use shift-invert mode for fast eigenvalues
        # (see https://docs.scipy.org/doc/scipy/reference/tutorial/arpack.html
        #  for a short explanation of what this means)
        # Because the normalized Laplacian has eigenvalues between 0 and 2,
        # I - L has eigenvalues between -1 and 1.  ARPACK is most efficient
        # when finding eigenvalues of largest magnitude (keyword which='LM')
        # and when these eigenvalues are very large compared to the rest.
        # For very large, very sparse graphs, I - L can have many, many
        # eigenvalues very near 1.0.  This leads to slow convergence.  So
        # instead, we'll use ARPACK's shift-invert mode, asking for the
        # eigenvalues near 1.0.  This effectively spreads-out the spectrum
        # near 1.0 and leads to much faster convergence: potentially an
        # orders-of-magnitude speedup over simply using keyword which='LA'
        # in standard mode.
        try:
            # We are computing the opposite of the laplacian inplace so as
            # to spare a memory allocation of a possibly very large array
            laplacian *= -1
            v0 = random_state.uniform(-1, 1, laplacian.shape[0])
            lambdas, diffusion_map = eigsh(laplacian, k=n_components,
                                           sigma=1.0, which='LM',
                                           tol=eigen_tol, v0=v0)
            embedding = diffusion_map.T[n_components::-1]
            if norm_laplacian:
                embedding = embedding / dd
        except RuntimeError:
            # When submatrices are exactly singular, an LU decomposition
            # in arpack fails.  We fallback to lobpcg
            eigen_solver = "lobpcg"
            # Revert the laplacian to its opposite to have lobpcg work
            laplacian *= -1


    if eigen_solver == 'amg':
        # Use AMG to get a preconditioner and speed up the eigenvalue
        # problem.
        if not sparse.issparse(laplacian):
            warnings.warn("AMG works better for sparse matrices")
        # lobpcg needs double precision floats
        laplacian = check_array(laplacian, dtype=np.float64,
                                accept_sparse=True)
        laplacian = _set_diag(laplacian, 1, norm_laplacian)
        ml = smoothed_aggregation_solver(check_array(laplacian, 'csr'))
        M = ml.aspreconditioner()
        X = random_state.rand(laplacian.shape[0], n_components + 1)
        X[:, 0] = dd.ravel()
        lambdas, diffusion_map = lobpcg(laplacian, X, M=M, tol=1.e-12,
                                        largest=False)
        embedding = diffusion_map.T
        if norm_laplacian:
            embedding = embedding / dd
        if embedding.shape[0] == 1:
            raise ValueError


    elif eigen_solver == "lobpcg":
        # lobpcg needs double precision floats
        laplacian = check_array(laplacian, dtype=np.float64,
                                accept_sparse=True)
        if n_nodes < 5 * n_components + 1:
            # see note above under arpack why lobpcg has problems with small
            # number of nodes
            # lobpcg will fallback to eigh, so we short circuit it
            if sparse.isspmatrix(laplacian):
                laplacian = laplacian.toarray()
            lambdas, diffusion_map = eigh(laplacian)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
        else:
            laplacian = _set_diag(laplacian, 1, norm_laplacian)
            laplacian=l
            # We increase the number of eigenvectors requested, as lobpcg
            # doesn't behave well in low dimension
            X = random_state.rand(laplacian.shape[0], n_components + 1)
            X[:, 0] = dd.ravel()
            lambdas, diffusion_map = lobpcg(laplacian, X, tol=1e-15,
                                            largest=False, maxiter=2000)
            embedding = diffusion_map.T[:n_components]
            if norm_laplacian:
                embedding = embedding / dd
            if embedding.shape[0] == 1:
                raise ValueError

    
    embedding = spectral_embedding_._deterministic_vector_sign_flip(embedding)
    if drop_first:
        #return embedding[1:n_components].T
        return embedding[-n_components:].T
    else:
        #return embedding[:n_components].T
        return embedding[-n_components:].T

# ************************
# Scrollable Frame Class
# ************************
class ScrollFrame(tk.Frame):
    def __init__(self, parent):
        super().__init__(parent) # create a frame (self)
        self.configure(height=640,width=200, background=color_bg)

        self.canvas = tk.Canvas(self,height=640,width=200, borderwidth=0, background=color_bg)          #place canvas on self
        self.viewPort = tk.Frame(self.canvas, background=color_bg)                    #place a frame on the canvas, this frame will hold the child widgets 
        self.vsb = tk.Scrollbar(self, orient="vertical", command=self.canvas.yview) #place a scrollbar on self 
        self.canvas.configure(yscrollcommand=self.vsb.set)                          #attach scrollbar action to scroll of canvas

        self.vsb.pack(side="right", fill="y")                                       #pack scrollbar to right of self
        self.canvas.pack(side="left", fill="both", expand=True)                     #pack canvas to left of self and expand to fil
        self.canvas.create_window((4,4), window=self.viewPort, anchor="nw",            #add view port frame to canvas
                                  tags="self.viewPort")

        self.viewPort.bind("<Configure>", self.onFrameConfigure)                       #bind an event whenever the size of the viewPort frame changes.

    def onFrameConfigure(self, event):                                              
        '''Reset the scroll region to encompass the inner frame'''
        self.canvas.configure(scrollregion=self.canvas.bbox("all"))                 #whenever the size of the frame changes, alter the scroll region respectively.

class NavigationToolbar(NavigationToolbar2Tk):
    # only display the buttons we need
    toolitems = [t for t in NavigationToolbar2Tk.toolitems if
                 t[0] in ('Home', 'Zoom')]

                    

        



app = SeaofBTCapp()


#app.style = ttk.Style()
#('winnative', 'clam', 'alt', 'default', 'classic', 'vista', 'xpnative')
#app.style.theme_use("clam")
app.mainloop()