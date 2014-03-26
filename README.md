DNet-kNN
========

Deep learning classifier. Original code and paper available on Renqiang's site:
http://www.cs.toronto.edu/~cuty/


Adapted from original deep belief network code by Geoff E. Hinton and R. R. Salakhutdinov (Science, 2006).
https://www.cs.toronto.edu/~hinton/


INSTRUCTIONS
============



### Classes

If your markup can be translated using a Ruby library, that's
great. Check out `lib/github/markups.rb` for some
examples. Let's look at Markdown:

    markup(:markdown, /md|mkdn?|markdown/) do |content|
      Markdown.new(content).to_html
    end

We give the `markup` method three bits of information: the name of the
file to `require`, a regular expression for extensions to match, and a
block to run with unformatted markup which should return HTML.



1. download MNIST data http://yann.lecun.com/exdb/mnist/


2. mex all the .c files
    mex addchv.c  
  mex addh.c  
  mex addv.c  
  mex sumiflessh2.c  
  mex sumiflessv2.c   
  
  
    markup(:markdown, /md|mkdn?|markdown/) do |content|
      Markdown.new(content).to_html
    end


3. pretraining
 * mnistdeepauto_d2

(note: if you have already trained the first several layers, and you want to 
change the dimensionality to another value and train the final layer,
use computeRBM4_v2.m) 

4. set the parameters in backprop_DNetkNN and run backprop. Open backprop_DNetkNN.m, set:
 * restart = 1;
 
and

paramters to set:
 * nologistic = 1 % use linear output units for top layer
 * max_iter=20 % perform conjugate gradient max_iter iterations of line searches
 * k = 5       % free parameter k in kNN classification
 * k1 = 5      % the number of true nearest neighbors for computing triples
 * k2 = 30     % the number of imposter nearest neighbors for computing triples

5. finally, run:
backprop_DNetkNN
(note that we use Carl Edward Rasmussen's minimize.m for performing
conjugate gradient descent) 
