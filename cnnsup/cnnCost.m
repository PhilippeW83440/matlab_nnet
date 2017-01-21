function [cost, grad, preds] = cnnCost(theta, images, labels, net, pred)

if ~exist('pred','var')
    pred = false;
end;

stack = cnnParamsToStack(theta, net);

numLayers = numel(net.layers);
a = cell(numLayers, 1);
delta = cell(numLayers, 1);

inputDim = net.layers{1}.inputDim;
numClasses = net.layers{numLayers}.numClasses;

% 1) Forward propagation:
%------------------------

numImages = size(images, 3); % number of images

% no specific filtering done yet => 1 filter format
% 28x28x1x60000
a{1} = reshape(images, inputDim, inputDim, 1, numImages);

for l = 2:numLayers-1
	switch net.layers{l}.type
	case 'convol'
		filterDim = net.layers{l}.filterDim;
		numFilters = net.layers{l}.numFilters;
		% in        : 28x28x1x60000
		% activatins: 20x20x20x60000
		a{l} = cnnConvolve(filterDim, numFilters, a{l-1}, stack{l}.W, stack{l}.b);
	case 'pool'
		assert(strcmp(net.layers{l-1}.type, 'convol'), 'The pool layer comes always after a convol layer');
		poolDim = net.layers{l}.poolDim;
		% in : 20x20x20x60000
		% out: 10x10x20x60000
		a{l} = cnnPool(poolDim, a{l-1});
	otherwise
		fprintf('unexpected layer type\n');
	end
end

% last layer is always a softmax layer
%probs = zeros(numClasses, numImages);
aForSoftmax = reshape(a{numLayers-1}, [], numImages);
z = stack{numLayers}.W * aForSoftmax;
z = bsxfun(@plus, z, stack{numLayers}.b);
probs = softmax(z);
a{numLayers} = probs;


% 2) Calculate Cost:
%-------------------
groundTruth = full(sparse(labels, 1:numImages, 1, numClasses, numImages));
cost = - sum(sum(groundTruth .* log(probs))) / numImages;

% Makes predictions given probs and returns without backproagating errors.
if pred
    [~,preds] = max(probs,[],1);
    preds = preds';
    grad = 0;
    return;
end;

% 3) Back propagation:
%---------------------
% Refer to: 'Notes on Convolutional Neural Networks' from Jake Bouvrie

delta{numLayers} = probs - groundTruth;

for l = numLayers-1:-1:2
	switch net.layers{l+1}.type
	case 'softmax'
		% numFilters: last one set during FF
		filterDim = size(a{l}, 1);
		% reshape: 2000x256 -> 10x10x20x256
		delta{l} = reshape(stack{l+1}.W' * delta{l+1}, filterDim, filterDim, numFilters, numImages);
	case 'pool'
		assert( size(a{l}, 3) == size(a{l+1}, 3));
		numFilters = size(a{l}, 3);
		% in /delta_pool: 10x10x20x256
		% out/delta_conv: 20x20x20x256
		poolDim = net.layers{l+1}.poolDim;
		for f = 1:numFilters
		  for iNum = 1:numImages
			% Upsample the incoming error using kron (mean pooling)
			delta{l}(:, :, f, iNum) = kron(delta{l+1}(:, :, f, iNum), ones(poolDim)) / (poolDim^2); % faster
		  end
		end
	case 'convol'
		% naming convention: filters at layer l - maps at layer l+1
		[nY, nX, numFilters, nObs] = size(a{l});

		numMaps = size(delta{l+1}, 3);
		delta{l} = zeros(size(a{l}));
		% 28 = 20 + 9 - 1
		for f = 1:numFilters % layer l
			deltaProp = zeros(nY, nX, 1, nObs);
			for m = 1:numMaps % layer l+1
				% delta{l} = delta{l+1} CONV W
				deltaProp = deltaProp + convn(delta{l+1}(:, :, m, :), stack{l+1}.W(:, :, f, m), 'full');
	  		end
			delta{l}(:, :, f, :) = deltaProp;
		end
	otherwise
		fprintf('unexpected layer type\n');
	end

	switch net.layers{l}.func
	case 'sigmoid'
		gradFunc = a{l} .* (1 - a{l});
	case 'identity'
		gradFunc = 1;
	otherwise
		error('unexpected transfer function');
	end

	delta{l} = delta{l} .* gradFunc;
end


% 4) Compute gradients:
%----------------------
% Refer to: 'Notes on Convolutional Neural Networks' from Jake Bouvrie

grad = [];
for l = 2:numLayers-1

	switch net.layers{l}.type
	case 'convol'
		% naming convention: filters at layer l - maps at layer l-1
		numMaps = size(a{l-1}, 3);
		numFilters = net.layers{l}.numFilters;
		Wc_grad = zeros(size(stack{l}.W));
		bc_grad = zeros(size(stack{l}.b));
		for f = 1:numFilters
			for m = 1:numMaps
				Wc_grad(:, :, m, f) = convn(a{l-1}(:, :, m, :), flipall(delta{l}(:, :, f, :)), 'valid');
			end
			deltaf = delta{l}(:, :, f, :);
			bc_grad(f) = sum(deltaf(:));
		end
		% average over all samples
		Wc_grad = Wc_grad / numImages;
		bc_grad = bc_grad / numImages;
		grad = [grad ; Wc_grad(:) ; bc_grad(:)];
	end

end

% final softmax layer
Wd_grad = delta{numLayers} * aForSoftmax' ./ numImages;
bd_grad = sum(delta{numLayers}, 2) ./ numImages;
grad = [grad ; Wd_grad(:) ; bd_grad(:)];

end


function X=flipall(X)
    for i=1:ndims(X)
        X = flip(X,i);
    end
end

