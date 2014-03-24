function [err,yy,Value]=energyclassify_v2(Min_L,Min_xTr,Min_yTr,Min_xTe,Min_yTe,Kg,varargin);
% function [err,yy,Value]=energyclassify_v2(Min_L,Min_xTr,Min_yTr,Min_xTe,Min_yTe,Kg,varargin);
%
% INPUT:
%  Min_L	:   transformation matrix (learned by KMMNN)
%  Min_xTr	:   training vectors (each row is an instance)
%  Min_yTr	:   training labels  (column vector!!)
%  Min_xTe  :   test vectors
%  Min_yTe  :   test labels
%  Kg	:   number of nearest neighbors
%
% Good luck!
%
% copyright by Kilian Q. Weinberger, 2006
% Modifed by Renqiang Min (02/2008)

L = Min_L;
x = Min_xTr';
y = Min_yTr';
xTest = Min_xTe';
yTest = Min_yTe';


% checks
D=length(L);
x=x(1:D,:);
xTest=xTest(1:D,:);
if(size(x,1)>length(L)) error('x and L must have matching dimensions!\n');end;

% set parameters
pars.alpha=1e-09;
pars.tempid=0;
pars.save=0;
pars.speed=10;
pars.skip=0;
pars.factor=1;
pars.correction=15;
pars.prod=0;
pars.thresh=1e-16;
pars.ifraction=1;
pars.scale=0;
pars.obj=0;
pars.union=1;
pars.margin=0;
pars.tabularasa=Inf;
pars.blocksize=500;
%pars=extractpars(varargin,pars);


%pars


%tempname=sprintf('temp%i.mat',pars.tempid);
% Initializationip
[D,N]=size(x);
[gen,NN]=getGenLS(x,y,Kg,pars);




Lx=L*x;
Lx2=sum(Lx.^2);
LxT=L*xTest;


for inn=1:Kg
 Ni(inn,:)=sum((Lx-Lx(:,NN(inn,:))).^2)+1;
end;

MM=min(y);
y=y-MM+1;
un=unique(y);
Value=zeros(length(un),length(yTest));

B=pars.blocksize;
if(size(x,2)>50000) B=250;end;
NTe=size(xTest,2);
for n=1:B:NTe
  %fprintf('%2.2f%%: ',n/NTe*100);
  nn=n:n+min(B-1,NTe-n);
  DD=distance(Lx,LxT(:,nn));  
 for i=1:length(un)
 % Main Loopfor iter=1:maxiter 
  testlabel=un(i);
  %fprintf('%i.',testlabel+MM-1);
  
  enemy=find(y~=testlabel);
  friend=find(y==testlabel);

  Df=mink(DD(friend,:),Kg);

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% NOTE: change here %%%%%
  %Value(i,nn)=sumiflessv2(DD,Ni(:,enemy),enemy)+sumiflessh2(DD,Df,enemy)+sum(Df);

  %%%%%%%%%%%%% ignore distances between genuine nearest neighbors
  Value(i,nn)=sumiflessv2(DD,Ni(:,enemy),enemy)+sumiflessh2(DD,Df,enemy);

  %Value(i,nn)=sumiflessh2(DD,Df+pars.margin,enemy)+sum(Df);  
 end;
 %fprintf('\n');
end;

 %fprintf('\n');
 [temp,yy]=min(Value);

 yy=un(yy)+MM-1;
err=sum(yy~=yTest)./length(yTest);
fprintf('Energy error:%2.2f%%\n',err*100);




function [gen,NN]=getGenLS(x,y,Kg,pars);
fprintf('Computing nearest neighbors ...\n');
[D,N]=size(x);

un=unique(y);
Gnn=zeros(Kg,N);
for c=un
%fprintf('%i nearest genuine neighbors for class %i:',Kg,c);
i=find(y==c);
nn=LSKnn(x(:,i),x(:,i),2:Kg+1);
Gnn(:,i)=i(nn);
%fprintf('\n');
end;

NN=Gnn;
gen1=vec(Gnn(1:Kg,:)')';
gen2=vec(repmat(1:N,Kg,1)')';

gen=[gen1;gen2];








function NN=LSKnn(X1,X2,ks,pars);
B=2000;
[D,N]=size(X2);
NN=zeros(length(ks),N);
DD=zeros(length(ks),N);

for i=1:B:N
  BB=min(B,N-i);
%  fprintf('.');
  Dist=distance(X1,X2(:,i:i+BB));
%  fprintf('.');
%  [dist,nn]=sort(Dist);
  [dist,nn]=mink(Dist,max(ks));
  clear('Dist');
%  fprintf('.'); 
%  keyboard;
  NN(:,i:i+BB)=nn(ks,:);
  clear('nn','dist');
%  fprintf('(%i%%) ',round((i+BB)/N*100)); 
end;


  




function v=vec(M);
% vectorizes a matrix

v=M(:);

