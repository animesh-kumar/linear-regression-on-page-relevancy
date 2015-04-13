function [M_cfs,lambda_cfs,W_cfs,S_cfs] = train_cfs(Input_Matrix,Target_Matrix,Mu)

% phi matrix will be N*M matrix where N is the number of datapoints i.e.,

% 69623 and M is nummber of basis functions

N = size(Input_Matrix,1);



% Training is 80%, Validation is 10%, Testing is 10%

Training_End_Index = ceil(0.8*N);

Validation_Start_Index = Training_End_Index+1;

Validation_End_Index = Validation_Start_Index+ceil(0.1*N);



% Array of basis functions

BasisArr = zeros(1);

% Array of Error

ErrorArr = zeros(1);

count = 1;

M = 4;

MinErrorOnM = 100;

% Let's first find out the number of basis functions which will

% minimize the error

% generate M in the increments of3 from 4 to 30

% Initialize with all zeros

Phi = zeros(N,M-1);

% Append a column of ones at beginning of phi matrix

Phi=[ones(size(Phi,1),1) Phi];

basisStartIndex = 2; % Just a pointer for convenience 

% Also when taking increments, just keep on adding to existing Phi Matrix

% Why waste time, memory and everything in calculating from scratch!!!

for i=4:1:25
    
    M = i;
    
    Sigma = var(Input_Matrix(:));
    
    %Phi = generateBasis(Input_Matrix,Phi,M,Mu,Sigma,basisStartIndex);
    
    Phi = generateBasis(Input_Matrix,Phi,M,Mu,Sigma,basisStartIndex);
    
    % Once basis functions are created.
    
    % Partition the dataset into training, validation and testing
    
    PhiForTraining = Phi(1:Training_End_Index,:);
    
    TargetforTraining = Target_Matrix(1:Training_End_Index,:);
    
    PhiForValidation = Phi(Validation_Start_Index:Validation_End_Index,:);
    
    TargetforValidation = Target_Matrix(Validation_Start_Index:Validation_End_Index,:);
    
    % Ignore the regularization factor to fix M
    
    Lambda = 0;
    W_Cfs = calculateMinimizedWeights(PhiForTraining, TargetforTraining,Lambda);
    Error = calculateRootMeanSquaredError(PhiForValidation,TargetforValidation,W_Cfs);
    basisStartIndex = M + 1;
    if Error < MinErrorOnM
        MinErrorOnM = Error;
        MWithMinError = M;
        MinimumWeight = W_Cfs;
        
    end
    
    % Store the number of basis functions and error in different arrays
    
    % so that it can be plotted later
    
    BasisArr(count) = M;
    
    ErrorArr(count) = Error;
    
    count = count + 1;
    
end

% figure(1); % create new figure

% plot(BasisArr,ErrorArr);

% title('Model Complexity vs Error (ERMS) For ');

% ylabel('Error (ERMS)') % y-axis label

% xlabel('Model Complexity') % x-axis label

% grid on;

% We will now fix M at MWithMinError

M = MWithMinError;

M_cfs = M;
W_cfs = MinimumWeight;
Sigma = var(Input_Matrix(:));
% Initialize with all zeros
Phi = zeros(N,M-1);
% Append a column of ones at beginning of phi matrix
Phi=[ones(size(Phi,1),1) Phi];
% Iterate over Lambda with the chosen M
Phi = generateBasis(Input_Matrix,Phi,M,Mu,Sigma,2);
% Once basis functions are created.
% Partition the dataset into training, validation and testing
PhiForTraining = Phi(1:Training_End_Index,:);
TargetforTraining = Target_Matrix(1:Training_End_Index,:);
PhiForValidation = Phi(Validation_Start_Index:Validation_End_Index,:);

TargetforValidation = Target_Matrix(Validation_Start_Index:Validation_End_Index,:);

MinErrorOnLambda = 100;

for Lambda = 1:2:15
    W_Cfs = calculateMinimizedWeights(PhiForTraining, TargetforTraining,Lambda);
    Error = calculateRootMeanSquaredError(PhiForValidation,TargetforValidation,W_Cfs);
    ErrorWithLambda(count) = Error;
    LambdaArr(count) = Lambda;
    count = count + 1;
    if Error < MinErrorOnLambda
        MinErrorOnLambda = Error;
        LambdaWithMinError = Lambda;
        Chosen_W = W_Cfs;
    end
    
end
W_cfs = Chosen_W;

% figure(2);
% plot(LambdaArr,ErrorWithLambda);
% title('Lambda (λ) vs Error (ERMS)');
% ylabel('Error (ERMS)') % y-axis label
% xlabel('Lambda (λ)') % x-axis label
% grid on;


lambda_cfs = LambdaWithMinError;
S_cfs = Sigma;


function Phi = generateBasis(Input_Matrix,Phi,M,Mu,Sigma,basisStartIndex)
N = size(Input_Matrix,1);
% Create a matrix of M-previous columns and then append ones to the beginning

for basisCount = basisStartIndex:M
    
    % Generate the basis functions
    % from 1 to number of rows
    % Find x-mu
    % x(i) - Mu - Here basisCount
    % https://www.mathworks.com/matlabcentral/newsreader/view_thread/267082
    
    row = bsxfun(@minus,Input_Matrix,Mu(basisCount-1));
    
    for rowCount = 1:N
        
        chosenRow = row(rowCount,:);
        
        row_transpose = transpose(chosenRow);
        
        % Formula Used
        
        % https://www.cs.cmu.edu/~epxing/Class/10701-08s/recitation/gaussian.pdf
        
        %Phi(rowCount,basisCount) = exp(-(chosenRow*Cov_Inverse*row_transpose/2));
        
        Phi(rowCount,basisCount) = exp(-(chosenRow*row_transpose/(2*Sigma)));
        
    end;
    
end;



function W_Cfs = calculateMinimizedWeights(PhiForTraining, TargetforTraining,Lambda)
% Given a λ, Phi and Target calculate the weights and Error
% w M L = (λI + Φ T Φ) −1 Φ T t
% It represents regularised weights. Let's say λ is between 1 and 100
LambdaMatrix = Lambda * eye(size(PhiForTraining,2));
% Regularised weight
W_Cfs = pinv(LambdaMatrix+(transpose(PhiForTraining)*PhiForTraining))*transpose(PhiForTraining)*TargetforTraining;


function Error = calculateRootMeanSquaredError(PhiForValidation,TargetforValidation,W_Cfs)

% Validate the error
% Apply these regularised weight on the validation data
% E(w) = E D (w) + λE W (w)
% Summation of squared difference between target values and calculated
% values
CalculatedTarget = (transpose((transpose(W_Cfs)*transpose(PhiForValidation))));
Edw = sum((TargetforValidation - CalculatedTarget).^2)/2;
% Eww = Sum(w^q), we take q=2 to reflect quad regularization
%Eww = sum(W_Cfs.^2)/2;
%Ew = Edw + Lambda * Eww;
Ew = Edw;
Error = sqrt(2*Ew/size(PhiForValidation,1));