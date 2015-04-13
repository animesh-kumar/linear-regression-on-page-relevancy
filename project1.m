load project1_data.mat
load mu_cfs.mat
load mu_gd.mat
myubitname = 'animeshk';
mystudentnumber = 50134753;
fprintf('My ubit name is %s\n',myubitname);
fprintf('My student number is %d \n',mystudentnumber);
[M_cfs,lambda_cfs,W_cfs,s_cfs] = train_cfs(Input_Matrix,Target_Matrix,mu_cfs);
save W_cfs.mat W_cfs
save s_cfs.mat s_cfs
N = size(Input_Matrix,1);
Validation_End_Index = ceil(0.9*N);
rms_cfs = test_cfs(Input_Matrix(Validation_End_Index+1:N,:),Target_Matrix(Validation_End_Index+1:N,:),M_cfs,mu_cfs,W_cfs,s_cfs);

[M_gd,lambda_gd,W_gd,s_gd] = train_gd(Input_Matrix,Target_Matrix,mu_gd);
save W_gd.mat W_gd
save s_gd.mat s_gd
N = size(Input_Matrix,1);
Validation_End_Index = ceil(0.9*N);
rms_gd = test_gd(Input_Matrix(Validation_End_Index+1:N,:),Target_Matrix(Validation_End_Index+1:N,:),M_gd,mu_cfs,W_gd,s_gd);

fprintf('the model complexity M_cfs is %d\n', M_cfs);
fprintf('the model complexity M_gd is %d\n', M_gd);
fprintf('the regularization parameters lambda_cfs is %4.2f\n', lambda_cfs);
fprintf('the regularization parameters lambda_gd is %4.2f\n', lambda_gd);
fprintf('the root mean square error for the closed form solution is %4.2f\n', rms_cfs);
fprintf('the root mean square error for the gradient descent method is %4.2f\n', rms_gd);

