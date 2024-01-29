
function  R = demo(filename,categorical)
addpath(genpath(pwd));
dir = fullfile('D:\Guan\Work\datasets\');

file = filename;
[~,remain] = strtok(file,'.');
if strcmp(remain,'.csv')
    instances = csvread(strcat(dir,filename),1,0);
else
    instances = xlsread(strcat(dir,filename));
end

num_data = size(instances,1);

%-----stratified cross validation---------%
cv = 10;
permutation = randsample(num_data,num_data);
instances = instances(permutation,:);
all_labels = instances(:,end);
permut_pos = instances(all_labels==1,:);
h_pos = size(permut_pos,1);
permut_neg = instances(all_labels==0,:);
h_neg = size(permut_neg,1);
% Split the current training dataset E into n equal sized balanced subsets
pos = floor(h_pos/cv);
remainder_pos = h_pos-pos*cv;
id = randperm(cv);
No_pos = ones(1,cv)*pos;
No_pos(id(1:remainder_pos)) = No_pos(id(1:remainder_pos))+1;
No_pos = [0 cumsum(No_pos)];

neg = floor(h_neg/cv);
remainder_neg = h_neg-neg*cv;
id = randperm(cv);
No_neg = ones(1,cv)*neg;
No_neg(id(1:remainder_neg)) = No_neg(id(1:remainder_neg))+1;
No_neg = [0 cumsum(No_neg)];

for i=1:cv
    fold{i} = [permut_pos(No_pos(i)+1:No_pos(i+1),:);permut_neg(No_neg(i)+1:No_neg(i+1),:)];
end

%%
AUC = [];
Acc = [];
G = [];
F = [];
SEN = [];
SPE = [];
Rej = [];
RPR = [];
RNR = [];

for i = 1:cv
    test_data = fold{i}(:,1:end-1); % external test set
    test_labels = fold{i}(:,end);
    m = 1;
    train_allinstances = [];
    for j = 1:cv
        if j~=i
            train_instances{m} = fold{j};  %cv-1 fold corss validation to obtain threshold averaged ROC
            train_allinstances = [train_allinstances;fold{j}];  % external training set
            m = m+1;
        end
    end
    T = [];
    for j = 1:cv-1
        subtest_data = train_instances{j}(:,1:end-1); % internal test set
        subtest_labels = train_instances{j}(:,end);
        subtrain_instances = [];
        for jj = 1:cv-1
            if jj~=j
                subtrain_instances = [subtrain_instances;train_instances{jj}]; % internal trainin set
            end
        end
        
   
        score_subtest{j} = ComputeTWSVMScore(subtrain_instances,subtest_data);
        T = [T;score_subtest{j}];
        [rates{j},thr{j}] = generateROC(score_subtest{j},subtest_labels);
        
    end
    T = sort(T,'descend');
    s = 1;
    sample = 50;
    for tidx = 1:round(length(T)/sample):length(T)
        FP = 0;
        TP = 0;
        th = 0;
        n = 0;
        flag = 0;
        for kk = 1:cv-1
            ix = find(thr{kk}<=T(tidx),1,'first');
            if isempty(ix)
                flag = 1;
                break;
            end
            FP = FP+rates{kk}(ix,1);
            TP = TP+rates{kk}(ix,2);
            th = th+thr{kk}(ix);
            n = n+1;
        end
        if flag == 1
            break;
        end
        avg_rates(s,:) = [FP/n TP/n];   % threshold averaged ROC curve
        avg_thr(s,:) = th/n;
        s = s+1;
    end
    avg_rates = [[0 0];avg_rates;[1 1]];
    avg_thr = [inf;avg_thr;-inf];
    i1 = find(avg_rates(:,1)<=0,1,'last');
    i2 = find(avg_rates(:,2)<=0,1,'last');
    avg_rates = avg_rates(min(i1,i2):end,:);
    avg_thr = avg_thr(min(i1,i2):end);
    i1 = find(avg_rates(:,1)>=1,1,'first');
    i2 = find(avg_rates(:,1)>=1,1,'first');
    avg_rates = avg_rates(1:max(i1,i2),:); % threshold averaged ROC curve
    avg_thr = avg_thr(1:max(i1,i2));
    [avg_rates,in] = sortrows(avg_rates);
    avg_thr = avg_thr(in);
    [rocch_rates,rocch_thr] = ROC_ConvexHull(avg_rates,avg_thr); % ROC on external training set
    
    %%  ====================================================================
    %------bounded-abstention to find two thresholds-----
    %%  =======================================================
    %-----------AUC maximization  BA2--------
    para = 0.99;
    l_step = 0.01;
    r_step = 0.01;
    times = 1;
    l = 0;
    r = 0;
    A = [];
    B = [];
    while para>0
        pos_max = para;
        neg_max = para;
        x1 = l_step*l;
        x2 = 1-r_step*r;
        while x1<x2
            unr = x2-x1;
            ix1 = find(rocch_rates(:,1)<=x1,1,'last');
            y1 = spline([rocch_rates(ix1,1) rocch_rates(ix1+1,1)],[rocch_rates(ix1,2) rocch_rates(ix1+1,2)],x1);
            ix2 = find(rocch_rates(:,1)>=x2,1,'first');
            y2 = spline([rocch_rates(ix2-1,1) rocch_rates(ix2,1)],[rocch_rates(ix2-1,2) rocch_rates(ix2,2)],x2);
            upr = y2-y1;
            if unr>neg_max && upr>pos_max
                if neg_max>pos_max
                    if unr>upr*times
                        r = r+1;
                    else
                        l = l+1;
                    end
                else
                    if upr>unr*times
                        l = l+1;
                    else
                        r = r+1;
                    end
                end
                
                x1 = l_step*l;
                x2 = 1-r_step*r;
                continue;
            end
            if unr>neg_max && upr<=pos_max
                r = r+1;
                x1 = l_step*l;
                x2 = 1-r_step*r;
                continue;
            end
            if unr<=neg_max && upr>pos_max
                l = l+1;
                x1 = l_step*l;
                x2 = 1-r_step*r;
                continue;
            end
            break;
        end
        if ix1==1
            a = rocch_thr(2)+0.01;
        else
            a = spline([rocch_rates(ix1,1) rocch_rates(ix1+1,1)],[rocch_thr(ix1) rocch_thr(ix1+1)],x1); % threshols obtained based on
        end
        if ix2 == length(rocch_thr)
            b = rocch_thr(end-1)-0.01;
        else
            b = spline([rocch_rates(ix2-1,1) rocch_rates(ix2,1)],[rocch_thr(ix2-1) rocch_thr(ix2)],x2); % ROCCH on external training set
        end
        A = [A a];
        B = [B b];
        para = para-0.02;
    end
  
    % ------BA2Cs  metrics----
    score_test = ComputeTWSVMScore(train_allinstances,test_data);
    reject = [];
    reject1 = [];
    reject0 = [];
    auc = [];
    acc = [];
    g = [];
    sen = [];
    spe = [];
    for t = 1:length(A)
        pred_test = zeros(size(test_labels))+2;
        pred_test(score_test>A(t),:) = 1;
        pred_test(score_test<B(t),:) = 0;
        
        h_test = length(test_labels);
        h1_test = sum(test_labels==1);
        h0_test = sum(test_labels==0);
        tp = sum(pred_test==1 & test_labels==1);
        fn = sum(pred_test==0 & test_labels==1);
        tn = sum(pred_test==0 & test_labels==0);
        fp = sum(pred_test==1 & test_labels==0);
        tpr = tp/(tp+fn);
        tnr = tn/(tn+fp);
        sen = [sen tpr];
        spe = [spe tnr];
        auc = [auc (tpr+tnr)/2];
        acc = [acc (tp+tn)/(tp+tn+fp+fn)];
        g = [g sqrt(tpr*tnr)];
        rpr = sum(pred_test==2 & test_labels==1)/h1_test;
        rnr = sum(pred_test==2 & test_labels==0)/h0_test;
        reject = [reject sum(pred_test==2)/h_test];
        reject1 = [reject1 rpr];
        reject0 = [reject0 rnr];
    end
    AUC = [AUC;auc];
    Acc = [Acc;acc];
    G = [G; g];
    SEN = [SEN;sen];
    SPE = [SPE;spe];
    Rej = [Rej; reject];
    RPR = [RPR; reject1];
    RNR = [RNR; reject0];
end
for x = 1:50
    R(1,x) =mean(Acc(~isnan(Acc(:,x)),x));
    R(2,x) =mean(AUC(~isnan(AUC(:,x)),x));
    R(3,x) =mean(G(~isnan(G(:,x)),x));
    R(4,x) =mean(SEN(~isnan(SEN(:,x)),x));
    R(5,x) =mean(SPE(~isnan(SPE(:,x)),x));
    R(6,x) =mean(Rej(~isnan(Rej(:,x)),x));
    R(7,x) =mean(RPR(~isnan(RPR(:,x)),x));
    R(8,x) =mean(RNR(~isnan(RNR(:,x)),x));
end