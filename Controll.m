%controll function
%% Initialization
%clear ; close all; clc

%% Setup the parameters you will use
input_layer_size  = 19;  % 20x20 Input Images of Digits
hidden_layer_size = 4;   % 25 hidden units
num_labels = 2;          % 10 labels, from 1 to 10   


%==========load data and visualize
% needs to have "X" as a varible holding all the data in examples x
% features and y with labels in for examples x 1
% Load Training Data
fprintf('Loading and Visualizing Data ...\n')

load('rain.mat');
m = size(X, 1);

% Randomly select 100 data points to display

%sel = randperm(size(X, 1));
%sel = sel(1:100);

%displayData(X(sel, :));

fprintf('Program paused. Press enter to continue.\n');
pause;



%% ================ Initializing Pameters 
%  In this portion the nn is initizlized with small random values to break
%  symetry

fprintf('\nInitializing Neural Network Parameters ...\n')

initial_Theta1 = randInitializeWeights(input_layer_size, hidden_layer_size);
initial_Theta2 = randInitializeWeights(hidden_layer_size, num_labels);

% Unroll parameters
initial_nn_params = [initial_Theta1(:) ; initial_Theta2(:)];



%% =================== Training NN
%  we will now use "fmincg", which
%  is a function which works similarly to "fminunc". Recall that these
%  advanced optimizers are able to train our cost functions efficiently as
%  long as we provide them with the gradient computations.
%
fprintf('\nTraining Neural Network... \n')


%  increasin value should improve acuracy.
options = optimset('MaxIter', 5000);

%  increasing lambda decreases variance
lambda = 2;

% Create "short hand" for the cost function to be minimized
costFunction = @(p) nnCostFunction(p, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, X, y, lambda);

% Now, costFunction is a function that takes in only one argument (the
% neural network parameters)
[nn_params, cost] = fmincg(costFunction, initial_nn_params, options);

% Obtain Theta1 and Theta2 back from nn_params
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

fprintf('Program paused. Press enter to continue.\n');
pause;

%% ================= Visualize Weights
%  You can now "visualize" what the neural network is learning by 
%  displaying the hidden units to see what features they are capturing in 
%  the data.

fprintf('\nVisualizing Neural Network... \n')

%displayData(Theta1(:, 2:end),8);

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;
%can we visualize theta2?
fprintf('\nVisualizing Neural Network... \n')

%displayData(Theta2(:, 2:end));

fprintf('\nProgram paused. Press enter to continue.\n');
%pause;

%% ================= Predict
%  After training the neural network, we would like to use it to predict
%  the labels. You will now implement the "predict" function to use the
%  neural network to predict the labels of the training set. This lets
%  you compute the training set accuracy.

pred = predict(Theta1, Theta2, X);

fprintf('\nTraining Set Accuracy: %f\n', mean(double(pred == y)) * 100);

