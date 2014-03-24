function triples = createTriplesNew(NNs, impNNs)
%function triples = createTriples(NNs, impNNs)

if iscell(impNNs) || iscell(NNs)
  error('impNNs and NNs must be matrices');
end

N = size(NNs,1);
if N ~= size(impNNs,1)
  error('NNs and impNNs dont match on numcases');
end

kk1 = size(NNs,2);
kk2 = size(impNNs,2);
impNNs = repmat(impNNs', kk1, 1);
temp1 = repmat([1:N], kk1*kk2, 1);
temp2 = [];
for i=1:kk1
  temp2 = [temp2;repmat(NNs(:,i)', kk2, 1)];
end   
triples = [reshape(temp1, N*kk1*kk2, 1) reshape(temp2, N*kk1*kk2, 1) reshape(impNNs, N*kk1*kk2, 1)];
