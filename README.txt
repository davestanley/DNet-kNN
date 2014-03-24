1. download MNIST data

2. mex all the .c files
mex addchv.c  
mex addh.c  
mex addv.c  
mex sumiflessh2.c  
mex sumiflessv2.c   

3. pretraining
mnistdeepauto_d2

(note: if you have already trained the first several layers, and you want to 
change the dimensionality to another value and train the final layer,
use computeRBM4_v2.m) 

4. set the parameters in backprop_DNetkNN and run backprop
Open backprop_DNetkNN.m, set:

restart = 1;
&
paramters to set: 
  nologistic = 1 % use linear output units for top layer
  max_iter=20 % perform conjugate gradient max_iter iterations of line searches
  k = 5       % free parameter k in kNN classification
  k1 = 5      % the number of true nearest neighbors for computing triples
  k2 = 30     % the number of imposter nearest neighbors for computing triples

finally, run:
backprop_DNetkNN
(note that we use Carl Edward Rasmussen's minimize.m for performing
conjugate gradient descent) 

