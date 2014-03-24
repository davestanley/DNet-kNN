function NN = KNN_inclass(x, k, y)
% x: each row is a data point
x = x';
fprintf('Computing nearest neighbors ...\n');
[D,N]=size(x);
un=unique(y);
Gnn=zeros(k,N);
for c=1:length(un)
   fprintf('%d nearest genuine neighbors for class %d:',k,c);
   i=find(y==c);
   nn=LSKnn(x(:,i),x(:,i),2:k+1);
   Gnn(:,i)=i(nn);
   fprintf('\n');
end;

NN=Gnn';
	      
	      
				

function NN=LSKnn(X1, X2,ks);
B=3000;
[D,N]=size(X2);
k = length(ks);
NN=zeros(k,N);
DD=zeros(k,N);

for i=1:B:N
    BB=min(B,N-i);
    fprintf('.');
    Dist=distance(X1,X2(:,i:i+BB));
    fprintf('.');
    % include itself
    [dist,nn]=mink(Dist,k+1);
    clear('Dist');
    fprintf('.');
    %  keyboard;
    NN(:,i:i+BB)=nn(ks,:);
    clear('nn','dist');
    fprintf('(%i%%) ',round((i+BB)/N*100));
end;

