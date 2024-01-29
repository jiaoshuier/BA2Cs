function [score_test] = ComputeTWSVMScore(train_instances,test_data)

train_data = train_instances(:,1:end-1);
train_labels = train_instances(:,end);
% model = fitcsvm(train_data,train_labels);
% [~,score_test] = predict(model,test_data);

TestX = test_data;
pos_data = train_data(train_labels==1,:);
neg_data = train_data(train_labels==0,:);
DataTrain.A = pos_data;
DataTrain.B = neg_data;
FunPara.c1=0.1;
FunPara.c2=0.1;
FunPara.c3=0.1;
FunPara.c4=0.1;
% FunPara.kerfPara.type = 'rbf';
% FunPara.kerfPara.pars = 0.5;
FunPara.kerfPara.type = 'lin';
score_test = TWSVM(TestX,DataTrain,FunPara);