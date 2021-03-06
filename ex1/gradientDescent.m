function [theta, J_history] = gradientDescent(X, y, theta, alpha, num_iters)
%GRADIENTDESCENT Performs gradient descent to learn theta
%   theta = GRADIENTDESCENT(X, y, theta, alpha, num_iters) updates theta by 
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCost) and gradient here.
    %

  diff = 0;
  for i = 1:m
    diff = diff + (theta' * X(i,:)' - y(i,:));
  end
  temp = theta(1,:) - ((alpha / m) * diff);
  diff2 = 0;
  for i = 1:m
    diff2 = diff2 + (theta' * X(i,:)' - y(i,:)) * X(i,2);
  end
  temp2 = theta(2,:) - ((alpha / m) * diff2);
  theta(1,:) = temp;
  theta(2,:) = temp2;

    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCost(X, y, theta);

end

end
