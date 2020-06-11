function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%


% y is a vector of labels
%y_one_hot = zeros( size( y, 1 ), num_labels );

% assuming class labels start from one
%for i = 1:num_labels
%    rows = y == i;
%    y_one_hot( rows, i ) = 1;
%end

% Convert the y labels into onehot encoded vectors
y_matrix = eye(num_labels)(y,:);
% Or in matlab separate to two statements
% eye_matrix = eye(num_labels)
% y_matrix = eye_matrix(y,:)


% Forward propagate to get the probability of each class
% Stored in a3: activation of the last layer

a1 = [ones(m,1) X];

z2 = a1 * Theta1'; % mx401 * (25x401)' = mx25

% Applying the activation function to each input
a2 = sigmoid(z2); % mx25

% add the bias term of 1's to the units in the second layer
% this makes a2 mx26
a2 = [ones(m,1) a2]; % mx26

% dimensions of Theta2 = number of units in the next layer
%						x number of units in previous layer + 1 (bias added)
% Hence, Theta2 = 10x26
z3 = a2 * Theta2'; % mx26 * (10x26)' = mx10

% applying the activation function to each input in this layer
% The hyp (hypothesis) corresponds to the probability of each class
a3 = sigmoid(z3); % mx10


% Compute the non-regularized cost function.
% Please visit the tutorials for ex4 under Resources section of the course.
J = (1/m) * (sum(sum(-y_matrix.*log(a3))) - sum(sum(((1-y_matrix) .* log(1-a3))))); %


% Compute the additional cost due to regularization

reg_cost = (lambda/(2*m)) * (sum(sum( Theta1(:,2:end).*Theta1(:,2:end))) + sum(sum(Theta2(:,2:end).*Theta2(:,2:end))));

% Cost with regularization is therefore:
J = J + reg_cost;


% Computing gradients of the weights using backpropagation: Unregularized

% Error in the output layer d3
d3 = a3 - y_matrix; % 

d2 = (d3 * Theta2(:,2:end)) .* sigmoidGradient(z2);

Delta1 = d2' * a1;

Delta2 = d3' * a2;

Theta1_grad = Delta1/m; 
Theta2_grad = Delta2/m;


% Regularized
% setting the bias column of Theta1 and Theta2 to all zeros
Theta1(:,1) = 0; 
Theta2(:,1) = 0;

Theta1 = (lambda/m) *Theta1;
Theta2 = (lambda/m) *Theta2;


Theta1_grad = Theta1_grad + Theta1;
Theta2_grad = Theta2_grad + Theta2;





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
