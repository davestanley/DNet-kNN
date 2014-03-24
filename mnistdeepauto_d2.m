
% Version 1.000
%
% Code provided by Ruslan Salakhutdinov and Geoff Hinton  
% Modifed by Renqiang Min 2009
% Permission is granted for anyone to copy, use, modify, or distribute this
% program and accompanying programs and documents for any purpose, provided
% this copyright notice is retained and prominently displayed, along with
% a note saying that the original programs are available from our 
% web page. 
% The programs and documents are distributed without any warranty, express or
% implied.  As the programs were written for research purposes only, they have
% not been tested to the degree that would be advisable in any important
% application.  All use of these programs is entirely at the user's own risk.


% This program pretrains a deep autoencoder for MNIST dataset
% You can set the maximum number of epochs for pretraining each layer
% and you can set the architecture of the multilayer net.

%clear all
close all

maxepoch=50; % Originally they used 10 %In the Science paper we use maxepoch=50, but it works just fine. 
numhid=500; numpen=500; numpen2=2000; numopen=30;    

NUMHID = numhid;

fprintf(1,'Converting Raw files into Matlab format \n');
% converter;            % Only need to run this the first time - once files are created, can comment this out

fprintf(1,'Pretraining a deep autoencoder. \n');
fprintf(1,'The Science paper used 50 epochs. This uses %3i \n', maxepoch);

makebatches_MNIST_pretrain;
% makebatches_USPS
% load  USPS_digits_yzn.mat;
[numcases numdims numbatches]=size(batchdata);

fprintf(1,'Pretraining Layer 1 with RBM: %d-%d \n',numdims,numhid);
restart=1;
rbm;
hidrecbiases=hidbiases; 
save(['mnistvh' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen) 'ep' '_' int2str(maxepoch)], ...
'vishid', 'hidrecbiases', 'visbiases');

fprintf(1,'\nPretraining Layer 2 with RBM: %d-%d \n',numhid,numpen);
batchdata=batchposhidprobs;
numhid=numpen;
restart=1;
rbm;
hidpen=vishid; penrecbiases=hidbiases; hidgenbiases=visbiases;
save( ['mnisthp' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen) 'ep' '_' int2str(maxepoch)], ...
		 'hidpen', 'penrecbiases', 'hidgenbiases');

fprintf(1,'\nPretraining Layer 3 with RBM: %d-%d \n',numpen,numpen2);
batchdata=batchposhidprobs;
numhid=numpen2;
restart=1;
rbm;
hidpen2=vishid; penrecbiases2=hidbiases; hidgenbiases2=visbiases;
save(['mnisthp2' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen) 'ep' '_' int2str(maxepoch)], ...
		 'hidpen2', 'penrecbiases2', 'hidgenbiases2');

fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d \n',numpen2,numopen);
batchdata=batchposhidprobs;
numhid=numopen; 
restart=1;
rbmhidlinear;
hidtop=vishid; toprecbiases=hidbiases; topgenbiases=visbiases;
save(['mnistpo' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen) 'ep' '_' int2str(maxepoch)], ...
  'hidtop', 'toprecbiases', 'topgenbiases');

%%% after pretraining, now run backprop_DNetkNN
backprop_DNetkNN
