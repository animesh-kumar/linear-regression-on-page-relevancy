function Error = test_cfs(InputMatrixForTesting,TargetforTesting,M,Mu,W_Cfs,Sigma)

% Initialize with all zeros
N = size(InputMatrixForTesting,1);
Phi = zeros(N,M-1);

% Append a column of ones at beginning of phi matrix

Phi=[ones(size(Phi,1),1) Phi];

% generate the phi for testing
PhiForTesting = generateBasis(InputMatrixForTesting,Phi,M,Mu,Sigma);

% Given a λ, Phi and Target calculate the weights and Error
% w M L = (λI + Φ T Φ) −1 Φ T t
% It represents regularised weights. Let's say λ is between 1 and 100
Edw = sum((TargetforTesting - (transpose((transpose(W_Cfs)*transpose(PhiForTesting))))).^2)/2;
% Eww = Sum(w^q), we take q=2 to reflect quad regularization
%Eww = sum(W_Cfs.^2)/2;
Error = sqrt(2*Edw/size(PhiForTesting,1));

function Phi = generateBasis(Input_Matrix,Phi,M,Mu,Sigma)

N = size(Input_Matrix,1);

for basisCount = 2:M
    
    % Generate the basis functions
    
    % from 1 to number of rows
    
    % Find x-m
    
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