function load_dnn(dnn_name)
% runs training procedure for supervised multilayer network

% add common directory to your path for
% minfunc and mnist data helpers
home = pwd;
cd ../mnist; addpath(genpath(pwd)); cd (home);
% cd ../minFunc_2012; addpath(genpath(pwd)); cd (home);
cd ../common; addpath(genpath(pwd)); cd (home);


%% load mnist data
[data_train, labels_train, data_test, labels_test] = load_preprocess_mnist();

%% load a dnn network
load(dnn_name);
params = best_theta;

tic;

[~, ~, pred] = supervised_dnn_cost(params, ei, data_test, labels_test, true);
acc_test = mean(pred == labels_test);
fprintf('Without training, test accuracy: %f\n', acc_test);

opt_params = minFuncSGD(@(theta, data, labels, pred_only) supervised_dnn_cost(theta, ...
							ei, data, labels, pred_only),...
							params, data_train, labels_train, data_test, labels_test, options, ...
							ei);

time2train = toc;
fprintf('time2train = %f\n', time2train);


%% compute accuracy on the test and train set
[~, ~, pred] = supervised_dnn_cost(opt_params, ei, data_test, labels_test, true);
acc_test = mean(pred == labels_test);
fprintf('test accuracy: %f\n', acc_test);

[~, ~, pred] = supervised_dnn_cost(opt_params, ei, data_train, labels_train, true);
acc_train = mean(pred == labels_train);
fprintf('train accuracy: %f\n', acc_train);
