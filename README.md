# support-vector-machine

This code implements a support vector machine (SVM) to classify arbitrary data using the sklearn machine learning library.

An SVM works by creating a line (aka a hyperplane) between data points that belong to 2 categories. This line is positioned to maximize the distance between the data of each classification. The results of the program linear_svm.py are displayed below. The data is 2 diminsional, and the SVM finds the most optimal position for a linear line between the red and green data points.

<img src="https://github.com/PopeyedLocket/support-vector-machine/blob/master/linear_svm_img.png" width="600" height="400">


An SVM can also use a kernel to find a hyperplane in higher dimensions that separates the data more effectively. The results of the program n_dim_svm.py are displayed below. The hyperplane is still straight/planar mathematically but when depicted on 2 dimensions it appears curved.

<img src="https://github.com/PopeyedLocket/support-vector-machine/blob/master/n_dim_svm_img.png" width="600" height="400">
