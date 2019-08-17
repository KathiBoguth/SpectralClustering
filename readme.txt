README
Spectral Clustering Tool Version 1.0 06/08/2019

Loading Data:

-file must be CSV file, separator: comma
-for unlabeled data:
	-no header
	-no index etc.
-for labeled data (with "Labeled" option activated):
	-header, labeled column must be named "class"
	-no index etc.
	-class labels have to be integer or float values, no string
-Use Settings field:
	-txt file, format see "Save data"

-Create Data:
	-Samples must be integer value >0
	-Min and Max for centers must be integer or float value
	-centers must not be 0

Save data:
	-first file: saves datapoints as CSV file with class label, Header: Each dimension as empty string, label: "class"; eg. ",,class"
	-second file: 
		-Header: chooseGraph, value, chooseMatrix, Eigenvalues, number of clusters, custom lines, deleted lines
		-values in this order, new line for each, eg.:

chooseGraph, value, chooseMatrix, Eigenvalues, number of clusters, custom lines, deleted lines
kNN
6
Laplacian
3
3
[(1,5),(3,16)]
[(0,2)]

"Graph" Tab:

-possible Interaction:
	-hover points
	-mark points by clicking
	-connect or disconnect points by marking one dot and clicking on a second one
	-remove mark by clicking dot again or on empty area
-toolbar: Default view, previous view, next view, Pan and Zoom, Zoom, customize graph view, save as img
-Dimensions Slider: only active for Multidimensional data (>2), controls which dimension is displayed
-NMI and ARI: only acitve for labeled input data, "Recalc" for recalculation of metrics

"Eigenvectors" Tab:

-Transformed Eigenvectors (each Dimension equals corresponding Eigenvector to one Eigenvalue)
-"vertical" and "horizontal": displayed dimension, must be integer value <= chosen Eigenvalues

"PCA" Tab:

-Displays First two components of PCA
-Clustering corresponds to untransformed data

Generate Similarity Graph:

-Method of generation: kNN, epsilon, gaussian, complete
-Sliders: k, e, o, None
-Keep custom Lines: custom Lines and deleted Lines should be kept, when changing methode / value

Type of Laplacian matrix:

-Laplacian, alt Laplacian, Lsym, Lrw
-Heatmap adjusts automatically

Dimensions of Eigenvectors:

-Graph shows Eigenvalues
-red line separates chosen (left, marked red) from discarded
-can be dragged or clicked for positioning
-number equals number of chosen Eigenvalues
-buttons: Default view, Zoom
-drag and click is only possible, if zoom is deactivated

Clustering algorithm:

-only kMeans, b and c have no functionality
-Slider k for number of clusters

Your last Results:

-adds result after Saving:
	-Title corresponds to name of the saved file
	-image shows status when saved (current graph in current view)
	-Modularity and Conductance show quality of clustering
	-Restore retains points, edges and settings
