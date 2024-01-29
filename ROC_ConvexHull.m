function [rocch,thr_ch] = ROC_ConvexHull(rates,thr)

% calculate convex hull of ROC curve
% rates: (fpr, tpr)

curr = 1;
k = 1;
rocch = rates(curr,:);
thr_ch = thr(curr,:);
% I(k) = curr;
n = size(rates,1);

while curr < n
    % find the steepest line  
    rho = (rates(curr+1:end,2)-rates(curr,2))./(rates(curr+1:end,1)-rates(curr,1));    
    [~,ind] = max(rho);
    curr = curr+ind;
    k = k+1;
    rocch(k,:) = rates(curr,:);
    thr_ch(k,:) = thr(curr,:);
%     I(k) = curr;
end