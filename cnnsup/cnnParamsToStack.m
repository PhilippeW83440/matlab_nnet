function stack = cnnParamsToStack(theta, net)

% Converts unrolled parameters for a convolutional neural
% network followed by a softmax layer into structured weight
% tensors/matrices and corresponding biases

numLayers = numel(net.layers);
stack = cell(numLayers, 1);

mapDim = net.layers{1}.inputDim;
indS = 1;
indE = 0;

numMaps = 1;

for l=2:numLayers

	switch net.layers{l}.type
	case 'convol'
		stack{l} = struct;
		filterDim = net.layers{l}.filterDim;
		numFilters = net.layers{l}.numFilters;

		indE = indE + filterDim^2 * numMaps * numFilters;
		stack{l}.W = reshape(theta(indS:indE), filterDim, filterDim, numMaps, numFilters);

		indS = indE + 1;
		indE = indE + numFilters;
		stack{l}.b = theta(indS:indE);

		numMaps = numFilters;
		mapDim = mapDim - filterDim + 1;
		
		indS = indE + 1;
	case 'pool'
		mapDim = mapDim / net.layers{l}.poolDim;
		hiddenSize = mapDim^2 * numFilters;
		% no W,b here
	case 'softmax'
		stack{l} = struct;
		numClasses = net.layers{numLayers}.numClasses;
		indE = indE + hiddenSize * numClasses;

		stack{l}.W = reshape(theta(indS:indE), numClasses, hiddenSize);
		stack{l}.b = theta(indE+1:end);
	end
end

end
