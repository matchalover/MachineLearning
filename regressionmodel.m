

%% Initialization
clear ; close all; clc
addpath('.\LDA');
addpath('.\funcs');

%% Data reading
fprintf('\nLoading Data ...\n')
load .\data\trainx_mat_sample.mat
% fea_tr = squeeze(mean(trainx_mat_sample,3));
% fea_tr = double(squeeze(trainx_mat_sample(:,:,1,:)));
fea_tr = double(trainx_mat_sample);
load .\data\trainy_mat_sample.mat
[temp, gnd_tr] = max(trainy_mat_sample, [], 1);
load .\data\testx_mat_sample.mat
% fea_te = squeeze(mean(testx_mat_sample,3));
% fea_te = double(squeeze(testx_mat_sample(:,:,1,:)));
fea_te = double(testx_mat_sample);
load .\data\testy_mat_sample.mat
[temp, gnd_te] = max(testy_mat_sample, [], 1);
clear trainx_mat_sample trainy_mat_sample testx_mat_sample testy_mat_sample temp

%% Data dimension shuffle
fea_tr = permute(fea_tr, [4,1,2,3]);
fea_tr = reshape(fea_tr, [100000, 784*4]);
gnd_tr = permute(gnd_tr, [2,1]);
fea_te = permute(fea_te, [4,1,2,3]);
fea_te = reshape(fea_te, [25000, 784*4]);
gnd_te = permute(gnd_te, [2,1]);

%% Normalization
fprintf('\nNormalizing Data ...\n')
fea_tr = bsxfun(@minus, fea_tr, mean(fea_tr));
sigma = sqrt(mean(fea_tr.^2));
fea_tr = bsxfun(@times, fea_tr, 1./sigma);

fea_te = bsxfun(@minus, fea_te, mean(fea_te));
sigma = sqrt(mean(fea_te.^2));
fea_te = bsxfun(@times, fea_te, 1./sigma);

%% PCA & LDA
fprintf('\nPerforming PCA & LDA...\n')
%------------------PCA----------------------
pca_dim = 100;
options=[];
options.ReducedDim = pca_dim;
[eigvector, eigvalue] =PCA(fea_tr,options);
fea_tr = fea_tr * eigvector;
fea_te = fea_te * eigvector;

% whiten
% epsilon = 10^(-5);
% fea_tr = bsxfun(@times, fea_tr, 1./sqrt(eigvalue' + epsilon)); 
% fea_te = bsxfun(@times, fea_te, 1./sqrt(eigvalue' + epsilon)); 

%------------------LDA----------------------
options=[];
% options.Fisherface=1;
options.PCARatio = 1;
% options.ReducedDim=200;
[eigvector, eigvalue] = LDA(gnd_tr,options,fea_tr);
fea_tr = fea_tr * eigvector;
fea_te = fea_te * eigvector;

%% Logistic regression
input_layer_size  = pca_dim;  
num_labels = 4;          % 4 classes 

fprintf('\nTraining One-vs-All Logistic Regression...\n')
lambda = 0.0;
[all_theta] = oneVsAll(fea_tr, gnd_tr, num_labels, lambda);
pred = predictOneVsAll(all_theta, fea_te);

fprintf('\nTest Accuracy: %f\n', mean(double(pred == gnd_te)) * 100);

% Confusion Matrices
fprintf('\nClass 1 metrices ("1000"):\n')
[C1,order1] = confusionmat(gnd_te,pred,'order',[1,2,3,4]);
accuracy1 = sum(diag(C1))/sum(C1(:));
disp(accuracy1);
precision1 = C1(1,1)/sum(C1(:,1));
disp(precision1);
recall1 = C1(1,1)/sum(C1(1,:));
disp(recall1);
F1 = 2*recall1*precision1/(recall1+precision1);
disp(F1);

fprintf('\nClass 2 metrices ("0100"):\n')
[C2,order2] = confusionmat(gnd_te,pred,'order',[2,1,3,4]);
accuracy2 = sum(diag(C2))/sum(C2(:));
disp(accuracy2);
precision2 = C2(1,1)/sum(C2(:,1));
disp(precision2);
recall2 = C2(1,1)/sum(C2(1,:));
disp(recall2);
F2 = 2*recall2*precision2/(recall2+precision2);
disp(F2);

fprintf('\nClass 3 metrices ("0010"):\n')
[C3,order3] = confusionmat(gnd_te,pred,'order',[3,1,2,4]);
accuracy3 = sum(diag(C3))/sum(C3(:));
disp(accuracy3);
precision3 = C3(1,1)/sum(C3(:,1));
disp(precision3);
recall3 = C3(1,1)/sum(C3(1,:));
disp(recall3);
F3 = 2*recall3*precision3/(recall3+precision3);
disp(F3);

fprintf('\nClass 4 metrices ("0001"):\n')
[C4,order4] = confusionmat(gnd_te,pred,'order',[4,1,2,3]);
accuracy4 = sum(diag(C4))/sum(C4(:));
disp(accuracy4);
precision4 = C4(1,1)/sum(C4(:,1));
disp(precision4);
recall4 = C4(1,1)/sum(C4(1,:));
disp(recall4);
F4 = 2*recall4*precision4/(recall4+precision4);
disp(F4);










