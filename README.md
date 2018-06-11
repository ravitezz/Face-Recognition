# Face-Recognition

In this project we perform face recognition using subspace measure. We make use of eigen vectors to project the image in to lower dimension and to obtain feature vectors.
 

Algorithmic Approach:


1.We take the Training Images and make them in to single matrix, with each column representing an image

2.We do preprocessing on the image by subtracting the mean.

3.Since the dimension of each image is 3600*1 we make use of eigen decomposition (A^T Av=λv ) to reduce the dimensions. 
4.Here K value will decide the dimension.

5.We take eigen vectors(v1, v2,….. vK) corresponding to top K eigen values.

6.We then compute eigen faces(U_k=AV_k) and project each image to the subspace by determining α_k=u_k^T I_m, we represent the image based on these Alpha Values as the feature vectors
	We obtain feature vectors for all the training images

7.We project the test images in to this subspace and obtain the feature vectors.

8.We compare these test images to all the reference images based on Euclidean, Norm1, Norm2 and Manhattan(City Block) distance
	We take the minimum distance for the reference image and compute accuracy.

