function dist=distanceBlock(X,x);
% dist=distance(X,x)
%
% computes the pairwise squared distance matrix between any column vectors in X and
% in x
%
  % INPUT:
%
% X     dxN matrix consisting of N column vectors
% x     dxn matrix consisting of n column vectors
%
  % OUTPUT:
%
% dist  Nxn matrix
%
  % Example:
% Dist=distance(X,X);
% is equivalent to
% Dist=distance(X);
%

[D,N] = size(X);
if(nargin==1)
  dist=zeros(N);
 else
   dist=zeros(N,size(x,2));
end;

B=round(0.05*N);
fprintf(1, 'Blocksize:%i\n',B);
%dist=zeros(N);
block = 0;
for i=1:B:N
  block = block + 1;
  fprintf(1, 'compute block %d\n',block);
  bi=min(B,N-i);
for j=1:B:N
  bj=min(B,N-j);
dist([i,j]);
if(nargin>1)
  dist(i:i+bi,j:j+bj)=distance(X(:,i:i+bi),x(:,j:j+bj));
 else
   dist(i:i+bi,j:j+bj)=distance(X(:,i:i+bi),X(:,j:j+bj));
end;
dist(j:j+bj,i:i+bi)=dist(i:i+bi,j:j+bj).';
     end;
  end;
fprintf(1, '\n');

