
% Version 2.000
% 
% Renqiang Min copyright 2009
% for minbatch training on MNIST
% backpropagation for training DNet-kNN by minimizing margin violations
% adapted from Ruslan and Geoff Hinton's code for training deep autoencoder

if restart == 1

makebatches_MNIST;
%load batchdata_MNIST;

nologistic = 1;   % linear output units on layer 4

minimizetimes = [];
maxepoch=50; % 200
fprintf(1,'\nFine-tuning deep neural network by minimizing margin violations. \n');
[numcases numdims numbatches]=size(batchdata);
fprintf(1, 'there are %d batches of data with size %d\n', numbatches, numcases);
N=numcases; 


  initmaxepoch=50;
  NUMHID=500
  numpen=500
  numpen2=2000
  numopen=30
  %numopen1=30
  numopen1 = 30
  load(['mnistvh' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen) 'ep' '_' int2str(initmaxepoch)])
  load(['mnisthp' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen) 'ep' '_' int2str(initmaxepoch)])
  load(['mnisthp2' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen) 'ep' '_' int2str(initmaxepoch)])
  load(['mnistpo' '_' int2str(NUMHID) '_' int2str(numpen) '_' int2str(numpen2) '_' int2str(numopen1) 'ep' '_' int2str(initmaxepoch)])

 % used for training 1 dim code, comment the following lines for higher dimensions
 %hidtop = randn(size(hidtop));
 %toprecbiases = randn(size(toprecbiases));

  test_err=[];
  train_err=[];

end


%%%% PREINITIALIZE WEIGHTS OF THE AUTOENCODER %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
if restart == 1
    w1=[vishid; hidrecbiases];
    w2=[hidpen; penrecbiases];
    w3=[hidpen2; penrecbiases2];
    w4=[hidtop; toprecbiases];
end

%%%%%%%%%% END OF PREINITIALIZATIO OF WEIGHTS  %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

l1=size(w1,1)-1;
l2=size(w2,1)-1;
l3=size(w3,1)-1;
l4=size(w4,1)-1;
l5=size(w4,2);

if (restart)
  max_iter=20
  k = 5
  k1 = 5
  k2 = 30
  starting_epoch = 1;
  Triples = {};
  for batch=1:numbatches
    data = batchdata(:,:, batch);
    datatargets = batchtargets(:,:,batch);
    [dataclass junk] = find (datatargets' == 1);
    fprintf(1,'creating triples for batch %d...\n', batch);
    NNs = KNN_inclass(data, k1, dataclass);
    impNNs = calcImpNNs(data, dataclass, 10, k2);
    triples = createTriplesNew(NNs, impNNs);
    Triples{batch,1} = triples;
    clear NNs impNNs triples;
  end
else
    starting_epoch = length(train_err) + 1;
end

for epoch = starting_epoch:maxepoch

%%%%%%%%%%%%%%%%%%%% COMPUTE KNN CLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

  err=0; 
  data = xTr;
  len_data = size(data,1);
  data = [data ones(len_data,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(len_data,1)];             % We extend the dimensionality
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(len_data,1)];           % of each level RBM by 1. This is to account
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(len_data,1)];          % for the biases, in addition to vishid
  w4probs_tr = w3probs*w4;
  if ~(nologistic)
      w4probs_tr = 1./(1 + exp(-w4probs_tr));  
  end

  data = xTe;
  len_data = size(data,1);
  data = [data ones(len_data,1)];
  w1probs = 1./(1 + exp(-data*w1)); w1probs = [w1probs  ones(len_data,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(len_data,1)];
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(len_data,1)];
  w4probs_te = w3probs*w4;
  if ~(nologistic)
      w4probs_te = 1./(1 + exp(-w4probs_te));    
  end
  
  
  code_dim = size(w4probs_tr,2);
  fprintf(1,'computing knn err ... \n');
  [Eval Details] = knnclassify (eye(code_dim,code_dim),w4probs_tr,yTr,w4probs_te,yTe,k);
  %% 
  %  [err,yy,Value]=energyclassify_v2(eye(code_dim,code_dim),w4probs_tr,yTr,w4probs_te,yTe,k);
  %%
  train_err(epoch) = Eval(1)*100;
  test_err(epoch) = Eval(2)*100;
  
 fprintf(1,'Before epoch %d Train error: %6.3f Test error: %6.3f \t \t \n',epoch,train_err(epoch),test_err(epoch));

%%%%%%%%%%%%%% END OF COMPUTING KNN CLASSIFICATION ERROR %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

 tt=0;
 for batch = 1:numbatches
   fprintf(1,'epoch %d batch %d\r',epoch,batch);
   triples = Triples{batch};
   data = batchdata(:,:, batch);
%%%%%%%%%%%%%%% PERFORM CONJUGATE GRADIENT WITH max_iter LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 
  VV = [w1(:)' w2(:)' w3(:)' w4(:)']';  % Make one long row vector of weights
  Dim = [l1; l2; l3; l4; l5];   % Row vector of lengths


t1 = clock;
  if nologistic
        % minimize(starting point, function, length, "data" is passed to f)
        [X, fX] = minimize(VV,'CG_dnetknn_nologistic_v2',max_iter,Dim,data,triples);  
  else
        [X, fX] = minimize(VV,'CG_dnetknn_v2',max_iter,Dim,data,triples);
  end
  t2 = clock;
  minimizetimes = [minimizetimes; (t2-t1)];
  

  w1 = reshape(X(1:(l1+1)*l2),l1+1,l2);     % Unrolls the new weights
  xxx = (l1+1)*l2;
  w2 = reshape(X(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
  xxx = xxx+(l2+1)*l3;
  w3 = reshape(X(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
  xxx = xxx+(l3+1)*l4;
  w4 = reshape(X(xxx+1:xxx+(l4+1)*l5),l4+1,l5);
%%%%%%%%%%%%%%% END OF CONJUGATE GRADIENT WITH max_iter LINESEARCHES %%%%%%%%%%%%%%%%%%%%%%%%%%%%%
 end
 %save mnist_weights w1 w2 w3 w4 
 %save mnist_error test_err train_err minimizetimes;

end



