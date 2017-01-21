function theta = cnnInitParams(net)

mapDim = net.layers{1}.inputDim; % 28 with mnist
numLayers = numel(net.layers);
numClasses = net.layers{numLayers}.numClasses;

theta = [];

numMaps = 1;

for l = 2:numLayers-1;
	switch net.layers{l}.type
	case 'convol'
		filterDim = net.layers{l}.filterDim;
		numFilters = net.layers{l}.numFilters;

        fanOut = numFilters * filterDim ^ 2;
        fanIn = numMaps * filterDim ^ 2;
    	% Xaxier's scaling factor (Xavier Glorot)
    	range = 2 * sqrt(6 / (fanIn + fanOut));
		Wc = range * (rand(filterDim, filterDim, numMaps, numFilters) - 0.5);
		bc = zeros(numFilters, 1);
		theta = [theta ; Wc(:) ; bc(:)];

		numMaps = numFilters;
		mapDim = mapDim - filterDim + 1; % dimension of convolved image
	case 'pool'
		poolDim = net.layers{l}.poolDim;
		% assume outDim is multiple of poolDim
		assert(mod(mapDim, poolDim)==0, 'poolDim must divide mapDim');
		mapDim = mapDim / poolDim;
	end
end

% final layer is always a softmax layer
assert(strcmp(net.layers{numLayers}.type, 'softmax'), 'The last layer must be a softmax layer');


hiddenSize = mapDim^2 * numFilters;
r  = sqrt(6) / sqrt(numClasses + hiddenSize + 1);
Wd = rand(numClasses, hiddenSize) * 2 * r - r;
bd = zeros(numClasses, 1);

theta = [theta ; Wd(:) ; bd(:)];

end
