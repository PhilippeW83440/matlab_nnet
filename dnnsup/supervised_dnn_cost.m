function [ cost, grad, pred] = supervised_dnn_cost(theta, ei, data, labels, pred_only)
%SPNETCOSTSLAVE Slave cost function for simple phone net
%   Does all the work of cost / gradient computation
%   Returns cost broken into cross-entropy, weight norm, and prox reg
%        components (ceCost, wCost, pCost)

%% default values
po = false;
if exist('pred_only','var')
  po = pred_only;
end;

%% reshape into network
stack = params2stack(theta, ei);
numHidden = numel(ei.layer_sizes) - 1;
% PWE hAct = cell(numHidden+1, 1);
gradStack = cell(numHidden+1, 1);

%% --------------------------- Philippe WEINGERTNER September 2014 -----------------------------

m = size(data, 2); % number of samples

% 1 input + numHidden + output
numLayers = 1 + numHidden + 1;
numStacks = numel(stack);
assert(numStacks == numLayers - 1);

% 1) perform fwd propagation
%----------------------------

a = cell(numLayers, 1);
a{1} = data;

% hidden layers
for l = 1:numHidden
	z = stack{l}.W * a{l};
	z = bsxfun(@plus, z, stack{l}.b);

	switch ei.activation_fun
	case 'sigmoid'
		a{l+1} = sigmoid(z);
	case 'tanh'
		a{l+1} = tanh(z);
	case 'relu'
		a{l+1} = relu(z);
	otherwise
		error('Unknown transfer function');
	end
end

% output layer: softmax layer
z = stack{numStacks}.W * a{numLayers-1};
z = bsxfun(@plus, z, stack{numStacks}.b);
a{numLayers} = softmax(z); % numClasses x numSamples

pred_prob = a{numLayers};

[~, pred] = max(pred_prob, [], 1);
pred = pred';


%%% h = bsxfun(@minus, z, max(z)); % to prevent overflows
%%% h = exp(h);
%%% pred_prob = bsxfun(@rdivide, h, sum(h)); % numClasses x numSamples
%%% %[~, pred_labels] = max(pred_prob);
%%% a{numLayers} = pred_prob; % numClasses x numSamples


% 2) compute cost function
%-------------------------

if exist('labels', 'var')
	groundTruth = full(sparse(labels, 1:m, 1));

	% Cross Entropy cost
	ceCost = - sum(sum(groundTruth .* log(pred_prob))) / m;
	
	% L2 weight regularization
	wCost = 0;
	for l = 1:numStacks
		wCost = wCost + (ei.lambda / 2) * sum(sum(stack{l}.W .^ 2));
	end
	
	cost = ceCost + wCost;
else
	ceCost = - 1; wCost= -1, cost = -1;
end;


%% return here if only predictions desired.
%  with real cost (without regularization)
if po
  cost = ceCost;
  grad = [];  
  return;
end;


% 3) compute error at output layer
%----------------------------------

delta = cell(numLayers, 1);
% for softmax output layer:
delta{numLayers} = pred_prob - groundTruth;


% 4) backpropagation of the error
%--------------------------------

for l = numLayers-1:-1:2
	delta{l} = stack{l}.W' * delta{l+1};

	switch ei.activation_fun
	case 'sigmoid'
		gradFunc = a{l} .* (1 - a{l});
	case 'tanh'
		gradFunc = 1 - a{l}.^2;
	case 'relu'
		gradFunc = a{l} > 0;
	otherwise
		error('Unknown transfer function');
	end

	delta{l} = delta{l} .* gradFunc;
end


% 5) compute the desired partial derivates
%------------------------------------------

for l = numStacks:-1:1
	gradStack{l}.W = (1/m) * delta{l+1} * a{l}'; % 200x200
	gradStack{l}.b = (1/m) * sum(delta{l+1}, 2); % 200x1

	gradStack{l}.W = gradStack{l}.W + ei.lambda * stack{l}.W; % regularization
end


%% reshape gradients into vector
[grad] = stack2params(gradStack);
end

