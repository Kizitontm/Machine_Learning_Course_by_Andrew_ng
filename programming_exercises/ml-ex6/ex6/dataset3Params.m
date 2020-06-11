function [C, sigma] = dataset3Params(X, y, Xval, yval)
%DATASET3PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = DATASET3PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%

% different landmarks can be used
% I don't fully understand the basis of this choice
% 3x1 works: x1 = [1 2 1]; x2 = [0 4 -1];
% 4x1 works: x1 = [1 2 1 5]; x2 = [0 4 -1 3];
% also starting at different landmark values works
x1 = [1 2 1]; x2 = [0 4 -1]; 

C_list = [0.01 0.03 0.1 0.3 1 3 10 30];
s_list = [0.01 0.03 0.1 0.3 1 3 10 30];
% create a blank results matrix
results = zeros(length(C_list) * length(s_list), 3)   

row = 1;
for C_val = C_list
    for sigma_val = s_list

        % your code goes here to train using C_val and sigma_val
        %    and compute the validation set errors 'err_val'
		
		model = svmTrain(X, y, C_val, @(x1, x2)...
		gaussianKernel(x1, x2, sigma_val));
		
		predictions = svmPredict(model, Xval);
		
		err_val = mean(double(predictions ~= yval));

        % save the results in the matrix
        results(row,:) = [C_val sigma_val err_val];
        row = row + 1;
    end
end

% use the min() function on the results matrix to find 
%   the C and sigma values that give the lowest validation error 

[min_err_val, index_min_err_val] = min(results(:,3));

C = results(index_min_err_val, 1);
sigma = results(index_min_err_val, 2);



% =========================================================================

end