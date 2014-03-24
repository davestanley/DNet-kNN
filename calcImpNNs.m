function impNNs = calcImpNNs(x, y, numclasses, k2)
% x: each row is a data point
x = x';
impNNs = [];
fprintf('Computing imposter nearest neighbors ...\n');
[D,N]=size(x);
un=unique(y);
Gnn=zeros(k2,N);
if length(un)~=numclasses
  error('incorrect label inputs or numclasses');
end

for ii=1:numclasses-1
  for c=1:length(un) 
   i=find(y==c);
   c2= mod((c - 1 + ii),numclasses) + 1; 
   i2=find(y==c2);
   fprintf('%i nearest imposter neighbors for class %i with imposter class %i:',k2, c, c2);
   nn=LSKnn(x(:,i2),x(:,i),1:k2);
   Gnn(:,i)=i2(nn);
   fprintf('\n');
  end
  NNs = Gnn';
  impNNs = [impNNs NNs];
 end




function NN=LSKnn(X1, X2,ks);
B=3000;
%ks=[1:k]
[D,N]=size(X2);
NN=zeros(length(ks),N);
DD=zeros(length(ks),N);

for i=1:B:N
  BB=min(B,N-i);
fprintf('.');
Dist=distance(X1,X2(:,i:i+BB));
fprintf('.');

[dist,nn]=mink(Dist,length(ks)+1);
clear('Dist');
fprintf('.');
%  keyboard;
% note here X1 doesnt intersect X2 
NN(:,i:i+BB)=nn(ks,:);
clear('nn','dist');
fprintf('(%i%%) ',round((i+BB)/N*100));
end;

