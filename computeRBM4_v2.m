% Version 1.000
% initmaxepoch
% This program pretrains the final layer of a deep autoencoder for MNIST dataset
% given the pre-trained lower-layer weights
% load the batchdata first if you have the batchdata generated, otherwise, call:
% makesinglebatch_v2 (for a single batch)
% or
% makebatches_MNIST (for mini batches)

load batchdata_MNIST;

%load USPS_digit_yzn;

 NUMOPEN = 1 

 initmaxepoch = 50; numhid=500; numpen=500; numpen2=2000; numopen=30;

 NUMHID = numhid;

  load(['mnistvh' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen) 'ep' '_' int2str(initmaxepoch)])
  load(['mnisthp' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen) 'ep' '_' int2str(initmaxepoch)])
  load(['mnisthp2' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen) 'ep' '_' int2str(initmaxepoch)])

  %batchdata = AllxTr;
  [numcases numdims numbatches]=size(batchdata);

 numopen =  NUMOPEN;

% layer 1
hidbiases=hidrecbiases; 
for batch=1:numbatches
  data = batchdata(:,:,batch);                                                % Calculating probability of hidden transition
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));    % Does all cases at once, store in rows
  clear batchposhidprobs;
  batchposhidprobs(:,:,batch)=poshidprobs;
end
  clear data;

% layer 2
batchdata=batchposhidprobs;
clear batchposhidprobs;
vishid=hidpen; hidbiases=penrecbiases; visbiases=hidgenbiases;

for batch=1:numbatches
  data = batchdata(:,:,batch);                                                % Calculating probability of hidden transition
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));    % Does all cases at once, store in rows
  batchposhidprobs(:,:,batch)=poshidprobs;
end
clear data;

% for layer 3
batchdata=batchposhidprobs;
clear batchposhidprobs;
vishid=hidpen2; hidbiases=penrecbiases2; visbiases=hidgenbiases2;

for batch=1:numbatches
  data = batchdata(:,:,batch);                                                % Calculating probability of hidden transition
  poshidprobs = 1./(1 + exp(-data*vishid - repmat(hidbiases,numcases,1)));    % Does all cases at once, store in rows
  batchposhidprobs(:,:,batch)=poshidprobs;
end
clear data;

fprintf(1,'\nPretraining Layer 4 with RBM: %d-%d \n',numpen2,numopen);
batchdata=batchposhidprobs;
numhid=numopen; 
restart=1;
maxepoch = initmaxepoch;
rbmhidlinear;
hidtop=vishid; toprecbiases=hidbiases; topgenbiases=visbiases;
save(['mnistpo' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen) 'ep' '_' int2str(initmaxepoch)], ...
  'hidtop', 'toprecbiases', 'topgenbiases');

%backprop_DNetkNN;

