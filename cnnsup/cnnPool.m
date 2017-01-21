function pooledFeatures = cnnPool(poolDim, convolvedFeatures)
%cnnPool Pools the given convolved features
%
% Parameters:
%  poolDim - dimension of pooling region
%  convolvedFeatures - convolved features to pool (as given by cnnConvolve)
%                      convolvedFeatures(imageRow, imageCol, featureNum, imageNum)
%
% Returns:
%  pooledFeatures - matrix of pooled features in the form
%                   pooledFeatures(poolRow, poolCol, featureNum, imageNum)
%     

numImages = size(convolvedFeatures, 4);
numFilters = size(convolvedFeatures, 3);
convolvedDim = size(convolvedFeatures, 1);


pooledFeatures = zeros(convolvedDim / poolDim, ...
        convolvedDim / poolDim, numFilters, numImages);

% Use mean pooling here.
poolFilter = ones(poolDim) / (poolDim^2);

%%% for imageNum = 1:numImages
%%% 	for filterNum = 1:numFilters
%%% 		features = convolvedFeatures(:, :, filterNum, imageNum);
%%% 		poolConvolvedFeatures = conv2(features, poolFilter, 'valid');
%%% 		pooledFeatures(:, :, filterNum, imageNum) = poolConvolvedFeatures(1:poolDim:end, 1:poolDim:end);
%%% 	end
%%% end

for filterNum = 1:numFilters
	features = convolvedFeatures(:, :, filterNum, :);
	poolConvolvedFeatures = convn(features, poolFilter, 'valid');
	pooledFeatures(:, :, filterNum, :) = poolConvolvedFeatures(1:poolDim:end, 1:poolDim:end, :);
end


end

