% runs training procedure for supervised multilayer network
% softmax output layer with cross entropy loss function


% add common directory to your path for
% minfunc and mnist data helpers
home = pwd;
cd ../mnist; addpath(genpath(pwd)); cd (home);
% cd ../minFunc_2012; addpath(genpath(pwd)); cd (home);
cd ../common; addpath(genpath(pwd)); cd (home);


DEBUG = false;
SGD = 1;

if DEBUG
	dnnCheckGradients();
end

%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();


%% setup environment
% experiment information
% a struct containing network layer sizes etc
ei = [];
ei.plot_runtime = 0;
ei.early_stop = 1;
ei.early_max_alpha = 25; % nber of epochs
ei.early_max_stop = 75; % nber of epochs

ei.verbose = 1;

%% populate ei with the network architecture to train
% ei is a structure you can use to store hyperparameters of the network
% the architecture specified below should produce  100% training accuracy
% You should be able to try different network architectures by changing ei
% only (no changes to the objective function code)

% dimension of input features
ei.input_dim = 784;
% number of output classes
ei.output_dim = 10;
% sizes of all hidden layers and the output layer
ei.layer_sizes = [200, 200, ei.output_dim];
% scaling parameter for l2 weight regularization penalty
ei.lambda = 1e-5;
% which type of activation function to use in hidden layers
% feel free to implement support for only the logistic sigmoid function
ei.activation_fun = 'sigmoid';

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

%% setup minfunc options
options = [];
options.Method = 'lbfgs';
options.maxIter = 400;
options.display = 'iter';
options.maxFunEvals = 1e6;
options.optTol = 1e-7; % PWE


tic;

[~, ~, pred] = supervised_dnn_cost(params, ei, data_test, labels_test, true);
%[~,pred] = max(pred);
%acc_test = mean(pred'==labels_test);
acc_test = mean(pred == labels_test);
fprintf('Without training, test accuracy: %f\n', acc_test);

if SGD
	%  Use minFuncSGD to minimize the function
	% default settings
	optionsSGD.epochs = 200;
	optionsSGD.epochs = 1000; % TEST
	optionsSGD.minibatch = 256;
	optionsSGD.alpha = 0.2;
	optionsSGD.momentum = .95;

	opt_params = minFuncSGD(@(theta, data, labels, pred_only) supervised_dnn_cost(theta, ...
								ei, data, labels, pred_only),...
								params, data_train, labels_train, data_test, labels_test, optionsSGD, ...
								ei);

else
	%% run training
	[opt_params, opt_value, exitflag, output] = minFunc(@supervised_dnn_cost,...
	    params, options, ei, data_train, labels_train);
end

time2train = toc;
fprintf('time2train = %f\n', time2train);

%% compute accuracy on the test and train set
%[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_test, [], true);
% to get the cost
[~, ~, pred] = supervised_dnn_cost(opt_params, ei, data_test, labels_test, true);
%[~, pred] = max(pred);
%acc_test = mean(pred' == labels_test);
acc_test = mean(pred == labels_test);
fprintf('test accuracy: %f\n', acc_test);

%[~, ~, pred] = supervised_dnn_cost( opt_params, ei, data_train, [], true);
% to get the cost
[~, ~, pred] = supervised_dnn_cost(opt_params, ei, data_train, labels_train, true);
%[~, pred] = max(pred);
%acc_train = mean(pred' == labels_train);
acc_train = mean(pred == labels_train);
fprintf('train accuracy: %f\n', acc_train);



% 98.49% in 6 minutes (CPU training L-BFGS)
% -----------------------------------------
%options.Method = 'lbfgs';
%options.maxIter = 400;
%options.display = 'iter';
%options.optTol = 1e-7; % PWE

%ei.layer_sizes = [200, 200, ei.output_dim];
%ei.lambda = 1e-5;

%time2train = 368.902745
%test accuracy: 0.984900
%train accuracy: 1.000000


% 98.5% in 6 minutes (CPU training SGD)
% -------------------------------------
% optionsSGD.minibatch = 256;
% optionsSGD.alpha = 0.2;
% optionsSGD.momentum = .95;

%ei.layer_sizes = [200, 200, ei.output_dim];
%ei.lambda = 1e-5;

% time2train = 365.998644
% test accuracy: 0.985000
% train accuracy: 0.999783
