% Version 2.000
%
% Copyright Renqiang Min 2009
%
function [f, df] = CG_dnetknn_v2(VV,Dim,XX,triples)      
fprintf(1,'logistic output\n');
l1 = Dim(1);
l2 = Dim(2);
l3 = Dim(3);
l4= Dim(4);
l5= Dim(5);
N = size(XX,1);
 w1 = reshape(VV(1:(l1+1)*l2),l1+1,l2);
 xxx = (l1+1)*l2;
 w2 = reshape(VV(xxx+1:xxx+(l2+1)*l3),l2+1,l3);
 xxx = xxx+(l2+1)*l3;
 w3 = reshape(VV(xxx+1:xxx+(l3+1)*l4),l3+1,l4);
 xxx = xxx+(l3+1)*l4;
 w4 = reshape(VV(xxx+1:xxx+(l4+1)*l5),l4+1,l5);

  XX = [XX ones(N,1)];
  w1probs = 1./(1 + exp(-XX*w1)); w1probs = [w1probs  ones(N,1)];
  w2probs = 1./(1 + exp(-w1probs*w2)); w2probs = [w2probs ones(N,1)];
  w3probs = 1./(1 + exp(-w2probs*w3)); w3probs = [w3probs  ones(N,1)];
  w4probs = w3probs*w4; w4probs = [w4probs  ones(N,1)];
  code = 1./(1 + exp(-w4probs(:,1:(size(w4probs, 2)-1) ) ) );
  [err new_code_grad] = get_code_grad (code, triples);
  length_triples = size(triples, 1);
  err = 1/(length_triples)*err;
  new_code_grad = 1/(length_triples)*new_code_grad;

  f = -1/N*sum(sum( XX(:,1:end-1).*log(XXout) + (1-XX(:,1:end-1)).*log(1-XXout))) + err;   % This is our error function

Ix4 = new_code_grad;
dw4 =  w3probs'*Ix4;

Ix3 = (Ix4*w4').*w3probs.*(1-w3probs); 
Ix3 = Ix3(:,1:end-1);
dw3 =  w2probs'*Ix3;

Ix2 = (Ix3*w3').*w2probs.*(1-w2probs); 
Ix2 = Ix2(:,1:end-1);
dw2 =  w1probs'*Ix2;

Ix1 = (Ix2*w2').*w1probs.*(1-w1probs); 
Ix1 = Ix1(:,1:end-1);
dw1 =  XX'*Ix1;

df = [dw1(:)' dw2(:)' dw3(:)' dw4(:)']';    % Returns vector of partial derivatives WRT the weights

