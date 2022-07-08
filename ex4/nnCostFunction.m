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

X = [ones(m, 1), X];
h1 = sigmoid([X] * Theta1');
z2 = [X] * Theta1';
a2 = h1;
h1 = [ones(m, 1), h1];
h2 = sigmoid([h1] * Theta2');
z3 = [h1] * Theta2';
a3 = h2;

new_y = zeros(m, num_labels);
for i = 1:m
  if y(i,:) == num_labels
    new_y(i,:) = [zeros(1,num_labels - 1), 1];
  else
    new_y(i,:) = [zeros(1, y(i,1) - 1), 1, zeros(1, num_labels - y(i,1))];
  endif
endfor

for i = 1:m
  J = J - (new_y(i,:) * log(h2(i,:)') + (1-new_y(i,:)) * log(1-h2(i,:)'));
endfor

J = J / m;

theta1_sum = 0;
theta2_sum = 0;

for i = 1:hidden_layer_size
  theta1_sum = theta1_sum + (sum(Theta1(i,:).^2) - Theta1(i,1).^2);
endfor

for i = 1:num_labels
  theta2_sum = theta2_sum + (sum(Theta2(i,:).^2) - Theta2(i,1).^2);
endfor

J = J + (lambda / (2 * m)) * (theta1_sum + theta2_sum);

%Backprop


for t = 1:m
  error3 = a3(t,:) - new_y(t,:);
  %error3 is a 1 * 10 vector of the errors
  %Theta2 is a 10 * 26 matrix
  %26 * 10 x 10 * 1
  %first part of equation is 26 * 1, second is 25 * 1
  thetaError2 = (Theta2' * error3');
  thetaError2([1],:) = [];
  %25 * 1 .x 25 * 1
  error2 = thetaError2 .* sigmoidGradient(z2(t,:))';
 
 %get the gradients
 %10 * 1 x 1 * 26
 %25 * 1 x 1 * 401
  Theta2_grad = Theta2_grad + error3' * h1(t,:);
  Theta1_grad = Theta1_grad + error2 * X(t,:);
endfor

r_Theta1 = [zeros(size(Theta1, 1), 1), Theta1(:,2:end)];
Theta1_grad = Theta1_grad / m + (lambda / m) * r_Theta1;

r_Theta2 = [zeros(size(Theta2, 1), 1), Theta2(:,2:end)];
Theta2_grad = Theta2_grad / m + (lambda / m) * r_Theta2;






% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
