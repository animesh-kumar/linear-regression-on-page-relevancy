load project1_data.mat
% get M-1 random datapoint from the dataset and take them as mean
Mu  = Input_Matrix(randperm(100),:);
save mu_cfs.mat Mu;