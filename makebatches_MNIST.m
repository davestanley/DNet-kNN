makesinglebatch_v2;
fprintf(1,'make mini-batches ... \n');
xTr = AllxTr;
yTr = AllyTr;
clear AllxTr AllyTr;
xTe = AllxTe;
yTe = AllyTe;
clear AllxTe AllyTe;
% For training data:

digitdata = xTr;

totnum = size(digitdata,1);
numdims  =  size(digitdata,2); 
fprintf(1, 'Size of the training dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the training data
randomorder=randperm(totnum);

targets = zeros(totnum,10);
for i = 1:totnum
    targets(i,yTr(i)) = 1;
end

batchsize = 10000;

BD =[];
BT = [];
numLoops = 1

numbatches=totnum/batchsize;
batchdata = zeros(batchsize, numdims, numbatches*numLoops);
batchtargets = zeros(batchsize, 10, numbatches*numLoops);


for loop=1:numLoops
  rand('state',sum(100*clock));
  randn('state',sum(100*clock));
  randomorder=randperm(totnum);

  for b=1:numbatches                                  % Populate batches
    batchdata(:,:,(loop-1)*numbatches + b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
    batchtargets(:,:,(loop-1)*numbatches + b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  end;
end

clear targets digitdata;

%  For test data:
digitdata = xTe;

totnum=size(digitdata,1);
fprintf(1, 'Size of the test dataset= %5d \n', totnum);

rand('state',0); %so we know the permutation of the test data
randomorder=randperm(totnum);

targets = zeros(totnum,10);
for i = 1:totnum
    targets(i,yTe(i)) = 1;
end


numdims  =  size(digitdata,2);
batchsize = 10000;
numbatches=totnum/batchsize;
testbatchdata = zeros(batchsize, numdims, numbatches);
testbatchtargets = zeros(batchsize, 10, numbatches);

for b=1:numbatches
  testbatchdata(:,:,b) = digitdata(randomorder(1+(b-1)*batchsize:b*batchsize), :);
  testbatchtargets(:,:,b) = targets(randomorder(1+(b-1)*batchsize:b*batchsize), :);
end;
clear digitdata targets;

clear y data;
clear i;


%%% Reset random seeds 
rand('state',sum(100*clock)); 
randn('state',sum(100*clock)); 
