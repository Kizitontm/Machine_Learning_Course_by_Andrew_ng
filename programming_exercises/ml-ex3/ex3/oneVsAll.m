function [all_theta] = oneVsAll(X, y, num_labels, lambda)
%ONEVSALL trains multiple logistic regression classifiers and returns all
%the classifiers in a matrix all_theta, where the i-th row of all_theta 
%corresponds to the classifier for label i
%   [all_theta] = ONEVSALL(X, y, num_labels, lambda) trains num_labels
%   logistic regression classifiers and returns each of these classifiers
%   in a matrix all_theta, where the i-th row of all_theta corresponds 
%   to the classifier for label i

% y: argument to this function is a vector of labels from 1 to
% 		10, where we have mapped the digit 0 to the label 10 
% 		(to avoid confusions with indexing)

% Some useful variables
m = size(X, 1);
n = size(X, 2);

% You need to return the following variables correctly 
all_theta = zeros(num_labels, n + 1);

% Add ones to the X data matrix
X = [ones(m, 1) X];

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the following code to train num_labels
%               logistic regression classifiers with regularization
%               parameter lambda. 
%
% Hint: theta(:) will return a column vector.
%
% Hint: You can use y == c to obtain a vector of 1's and 0's that tell you
%       whether the ground truth is true/false for this class. true for one
%		class and false for all others. the resulting vector of 1's and 0's 
%		is same dimensions as y.
%
% Note: For this assignment, we recommend using fmincg to optimize the cost
%       function. It is okay to use a for-loop (for c = 1:num_labels) to
%       loop over the different classes.
%
%       fmincg works similarly to fminunc, but is more efficient when we
%       are dealing with large number of parameters.
%
% Example Code for fmincg:
%
%     % Set Initial theta
%     initial_theta = zeros(n + 1, 1);
%     
%     % Set options for fminunc
%     options = optimset('GradObj', 'on', 'MaxIter', 50);
% 
%     % Run fmincg to obtain the optimal theta
%     % This function will return theta and the cost 
%     [theta] = ...
%         fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), ...
%                 initial_theta, options);
%


% each row of all_theta stores the weights/parameter corresponding 
% to one of the K=num_labels classes. note the bias term increases
% the number of columns of this matrix by one

all_theta = zeros(num_labels, n+1);  


% for each class, we get the parameters for that class and then
% update the all_theta row to store those parameters
for c = 1:num_labels

	%initialize all weights/params to zeros
	initial_theta = zeros(n+1, 1);
	
	% Set Options for fmincg
	options = optimset('GradObj', 'on', 'MaxIter', 50);

	% Advanced Optimization: Run fmincg to obtain the optimal theta
	% This function will return theta and the cost, J. 
	% y==c says in the vector y, where an element == c put a 1 and 0's 
	% everywhere else.
	
	[theta, J] = fmincg (@(t)(lrCostFunction(t, X, (y == c), lambda)), initial_theta, options);
	
	% update the c'th all_theta row to store the parameters theta
	% hence convert theta from vector n+1x1 to 1xn+1
	all_theta(c,:) = theta';
	
end








% =========================================================================


end
