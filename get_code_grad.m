function [err new_code_grad] = get_code_grad (Y, triples)
% Assume we have created triples (i, j, k) using createTriples(NNs, impNNs)
% assume Y stores the code vectors
% numcases
% dim

numcases = size(Y,1);
dim = size(Y,2);

Dist_nn = sum((Y(triples(:,1),:) - Y(triples(:,2),:)).^2, 2);
Dist_impnn = sum((Y(triples(:,1),:) - Y(triples(:,3),:)).^2, 2);
marginViolations = 1 + Dist_nn - Dist_impnn;
violationIndices = find(marginViolations > 0);
violationTriples = triples(violationIndices, :);
% time complextity 3 * N * some parallel vector substractions
% margin_violations = \sum_ijk n_ij (1 - y_ik) (1 - label(i,k))(1 + d_ij - d_ik)

err = sum(marginViolations(violationIndices));

new_code_grad1 = zeros(numcases, dim);
new_code_grad2 = zeros(numcases, dim);
new_code_grad3 = zeros(numcases, dim);

unique_is = unique(violationTriples(:,1));
unique_js = unique(violationTriples(:,2));
unique_ks = unique(violationTriples(:,3));


for tmpind = 1:length(unique_is)
  curr_i = unique_is(tmpind);
  curr_i_ind = find(violationTriples(:,1) == curr_i);
  new_code_grad1(curr_i,:) = -2*sum(Y(violationTriples(curr_i_ind, 2),:) ...
				  - Y(violationTriples(curr_i_ind, 3),:), 1);
end

    % for js
for tmpind = 1:length(unique_js)
  curr_j = unique_js(tmpind);
  curr_j_ind = find(violationTriples(:,2) == curr_j);
  new_code_grad2(curr_j,:) = -2*sum(Y(violationTriples(curr_j_ind, 1),:) ...
				  - Y(violationTriples(curr_j_ind, 2),:), 1);
end

    % for ks
for tmpind = 1:length(unique_ks)
  curr_k = unique_ks(tmpind);
  curr_k_ind = find(violationTriples(:, 3) == curr_k);
  new_code_grad3(curr_k,:) = 2*sum(Y(violationTriples(curr_k_ind, 1),:) ...
				   - Y(violationTriples(curr_k_ind, 3),:), 1);
end


new_code_grad = new_code_grad1 + new_code_grad2 + new_code_grad3;
clear Dist*  margin*  new_code_grad1  new_code_grad2  new_code_grad3 ...
    unique* violation*;