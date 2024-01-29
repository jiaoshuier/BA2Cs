function [rates,thr] = generateROC(p,testtarget)

% Ref: Robust classification systems for imprecise environments

Tcount = 0;
Fcount = 0;
plast = -inf;

E = [p testtarget];
E = sortrows(E);
E = flipud(E);
n = size(E,1);
index = 1;
for i = 1:n
    if E(i,1)~=plast
        tp(index,1) = Tcount;
        fp(index,1) = Fcount; 
        thr(index,1) = plast;
        plast = E(i,1); 
        index = index+1;
    end
    if E(i,2) == 1
        Tcount = Tcount+1;
    else
        Fcount = Fcount+1;
    end
end
thr(index,1) = plast;
tp(index,1) = Tcount;
fp(index,1) = Fcount;
tpr = tp/sum(testtarget);
fpr = fp/(n-sum(testtarget));
rates = [fpr tpr];
thr(1,1) = Inf;


