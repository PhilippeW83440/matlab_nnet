%% Convolution Neural Network
global testImages;
global testLabels;


CONFIG = 2;
DEBUG = 0;

imageDim = 28;

% add common directory to your path for
% minfunc and mnist data helpers
home = pwd;
cd ../mnist; addpath(genpath(pwd)); cd (home);
% cd ../minFunc_2012; addpath(genpath(pwd)); cd (home);
cd ../common; addpath(genpath(pwd)); cd (home);


% Load MNIST Train
images = loadMNISTImages('train-images-idx3-ubyte');
images = reshape(images,imageDim,imageDim,[]);
labels = loadMNISTLabels('train-labels-idx1-ubyte');
labels(labels==0) = 10; % Remap 0 to 10

% Load MNIST Test
testImages = loadMNISTImages('t10k-images-idx3-ubyte');
testImages = reshape(testImages,imageDim,imageDim,[]);
testLabels = loadMNISTLabels('t10k-labels-idx1-ubyte');
testLabels(testLabels==0) = 10; % Remap 0 to 10

if DEBUG
	cnnCheckGradients3(images(:, :, 1:11), labels(1:11));
end



%---------------------------------
% Define your network architecture
%---------------------------------

net = [];

if CONFIG == 1
	% first layer is always an input layer
	l = 1;
	net.layers{l}.type = 'input';
	net.layers{l}.inputDim = imageDim; % => 28*28
	
	% convolution layer
	l = l + 1;
	net.layers{l}.type = 'convol';
	net.layers{l}.numFilters = 20;
	net.layers{l}.filterDim = 9; % => 9*9
	net.layers{l}.func = 'sigmoid'; % f(x) = sigmoid(x)
	
	% sub sampling / pooling layer
	l = l + 1;
	net.layers{l}.type = 'pool';
	net.layers{l}.poolDim = 2;	 % => 2*2
	net.layers{l}.func = 'identity'; % f(x) = x
	
	%net.layers{l}.type = 'full';
	%net.layers{l}.layerDim = 100;
	%net.layers{l}.func = 'sigmoid';
	
	% last layer is always a softmax layer
	l = l + 1;
	net.layers{l}.type = 'softmax';
	net.layers{l}.numClasses = 10;
end



if CONFIG == 2
	% first layer is always an input layer
	l = 1;
	net.layers{l}.type = 'input';
	net.layers{l}.inputDim = imageDim; % => 28*28
	
	% convolution layer
	l = l + 1;
	net.layers{l}.type = 'convol';
	net.layers{l}.numFilters = 20; %6
	net.layers{l}.filterDim = 9; % => 9*9
	%net.layers{l}.filterDim = 5; % => 5*5
	net.layers{l}.func = 'sigmoid'; % f(x) = sigmoid(x)
	
	% sub sampling / pooling layer
	l = l + 1;
	net.layers{l}.type = 'pool';
	net.layers{l}.poolDim = 2;	 % => 2*2
	net.layers{l}.func = 'identity'; % f(x) = x
	
	% convolution layer
	l = l + 1;
	net.layers{l}.type = 'convol';
	net.layers{l}.numFilters = 12; %12
	net.layers{l}.filterDim = 5;
	net.layers{l}.func = 'sigmoid'; % f(x) = sigmoid(x)
	
	% sub sampling / pooling layer
	l = l + 1;
	net.layers{l}.type = 'pool';
	net.layers{l}.poolDim = 2;	 % => 2*2
	net.layers{l}.func = 'identity'; % f(x) = x
	
	
	% last layer is always a softmax layer
	l = l + 1;
	net.layers{l}.type = 'softmax';
	net.layers{l}.numClasses = 10;
end


%---------------------------------
% Initialize Parameters
%---------------------------------

theta = cnnInitParams(net);


optionsSGD.epochs = 100;
optionsSGD.minibatch = 256; % 256
optionsSGD.minibatch = 16; % 16 % much better results
optionsSGD.alpha = 1e-1; % for 1 conv/pool
optionsSGD.momentum = .95;

if CONFIG == 2
	optionsSGD.epochs = 200;
	optionsSGD.alpha = 0.1;
	optionsSGD.minibatch = 32; %50;
end

%% STEP 1: Train
tstart = tic;
opttheta = minFuncSGD(@(x, im, lab, pred) cnnCost(x, im, lab, net, pred), ...
                      theta, images, labels, optionsSGD);
timeTrain = toc(tstart);
fprintf('timeTrain %f\n', timeTrain);

%% STEP 2: Test
[~, ~, preds] = cnnCost(opttheta, testImages, testLabels, net, true);

acc = sum(preds==testLabels) / length(preds);

fprintf('Accuracy is %f\n',acc);




% Accuracy initiale is 0.100900
% Epoch 1: testCost on iteration 1200 is 0.080761
%  - timeEpoch 205.806168
%  - Accuracy is 0.975000
% Epoch 2: testCost on iteration 2400 is 0.060044
%  - timeEpoch 204.437278
%  - Accuracy is 0.979900
% Epoch 3: testCost on iteration 3600 is 0.043329
%  - timeEpoch 203.143871
%  - Accuracy is 0.986200
% Epoch 4: testCost on iteration 4800 is 0.040769
%  - timeEpoch 203.562970
%  - Accuracy is 0.987900
% Epoch 5: testCost on iteration 6000 is 0.047871
%  - timeEpoch 202.997572
%  - Accuracy is 0.985200
% Epoch 6: testCost on iteration 7200 is 0.041811
%  - timeEpoch 203.442969
%  - Accuracy is 0.986700
% Epoch 7: testCost on iteration 8400 is 0.039653
%  - timeEpoch 202.970856
%  - Accuracy is 0.987400
% Epoch 8: testCost on iteration 9600 is 0.041348
%  - timeEpoch 202.236026
%  - Accuracy is 0.986400
% Epoch 9: testCost on iteration 10800 is 0.037284
%  - timeEpoch 203.641350
%  - Accuracy is 0.988900
% Epoch 10: testCost on iteration 12000 is 0.043104
%  - timeEpoch 204.489747
%  - Accuracy is 0.986500
% Epoch 11: testCost on iteration 13200 is 0.043780
%  - timeEpoch 204.868628
%  - Accuracy is 0.987300
% Epoch 12: testCost on iteration 14400 is 0.044650
%  - timeEpoch 203.029203
%  - Accuracy is 0.987200
% Epoch 13: testCost on iteration 15600 is 0.042868
%  - timeEpoch 203.083014
%  - Accuracy is 0.987200
% Epoch 14: testCost on iteration 16800 is 0.033333
%  - timeEpoch 205.088365
%  - Accuracy is 0.989800
% Epoch 15: testCost on iteration 18000 is 0.046413
%  - timeEpoch 203.602696
%  - Accuracy is 0.986500
% Epoch 16: testCost on iteration 19200 is 0.031592
%  - timeEpoch 202.815485
%  - Accuracy is 0.990500

% Epoch 30: testCost on iteration 36000 is 0.035736
%  - timeEpoch 203.364094
%  - Accuracy is 0.990600
