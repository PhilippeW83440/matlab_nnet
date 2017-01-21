function convolvedFeatures = cnnConvolve(filterDim, numFilters, inputs, W, b)

% images: imageDim x imageDim             x numImages
% inputs:   mapDim x   mapDim x   numMaps x numImages

%cnnConvolve Returns the convolution of the features given by W and b with
%the given images
%
% Parameters:
%  filterDim - filter (feature) dimension
%  numFilters - number of feature maps
%  images - large images to convolve with, matrix in the form
%           images(r, c, image number)
%  W, b - W, b for features from the sparse autoencoder
%         W is of shape (filterDim,filterDim,numFilters)
%         b is of shape (numFilters,1)
%
% Returns:
%  convolvedFeatures - matrix of convolved features in the form
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)

numImages = size(inputs, 4);
numMaps = size(inputs, 3); % XXX

imageDim = size(inputs, 1);
convDim = imageDim - filterDim + 1;

convolvedFeatures = zeros(convDim, convDim, numFilters, numImages);

for filterNum = 1:numFilters
	% convolution of image with feature matrix
	%%% convolvedMap = zeros(convDim, convDim, numImages);

	convolvedMap = zeros(convDim, convDim, 1, numImages);
	
	% TODO use a connectionMap
	for mapNum = 1:numMaps
		%%% map = squeeze(inputs(:, :, mapNum, :));
		%%% filter = squeeze(W(:, :, mapNum, filterNum));
		%%% filter = rot90(filter, 2);
		%%% convolvedMap = convolvedMap + convn(map, filter, 'valid');

		map = inputs(:, :, mapNum, :);
		filter = W(:, :, mapNum, filterNum);
		convolvedMap = convolvedMap + convn(map, flipall(filter), 'valid');
	end
	
	% TODO: sigmoid -> more generic
	convolvedMap = sigmoid(convolvedMap + b(filterNum));
	convolvedFeatures(:, :, filterNum, :) = convolvedMap;
end


end

function X=flipall(X)
    for i=1:ndims(X)
        X = flip(X,i);
    end
end

