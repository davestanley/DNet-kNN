DNet-kNN
========

Deep learning classifier. Original code and paper available on Renqiang's site:
http://www.cs.toronto.edu/~cuty/


Adapted from original deep belief network code by Geoff E. Hinton and R. R. Salakhutdinov (Science, 2006).
https://www.cs.toronto.edu/~hinton/


INSTRUCTIONS
------------



### 1. Download MNIST data
MNIST data can be found here: http://yann.lecun.com/exdb/mnist/


### 2. Mex all the .c files

        mex addchv.c
        mex addh.c  
        mex addv.c  
        mex sumiflessh2.c  
        mex sumiflessv2.c   
  
### 3. Pretraining

        mnistdeepauto_d2

(note: if you have already trained the first several layers, and you want to 
change the dimensionality to another value and train the final layer,
use computeRBM4_v2.m) 

### 4. Set the parameters in backprop_DNetkNN and run backprop.
Open backprop_DNetkNN.m, set:

    restart = 1;
 

and paramters to set:

    nologistic = 1 % use linear output units for top layer
    max_iter=20 % perform conjugate gradient max_iter iterations of line searches
    k = 5       % free parameter k in kNN classification
    k1 = 5      % the number of true nearest neighbors for computing triples
    k2 = 30     % the number of imposter nearest neighbors for computing triples

### 5. Finally, run:

    backprop_DNetkNN

(Note that we use Carl Edward Rasmussen's minimize.m for performing
conjugate gradient descent) 
