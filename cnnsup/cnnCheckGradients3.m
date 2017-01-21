function cnnCheckGradients3(images, labels)

	fprintf('checkGradients3 with 2 Convol/Pool layers \n');

    %images = randn(14,14,11);
    %labels = randi(10,11,1);
    
	net = [];
	
	% first layer is always an input layer
	l = 1;
	net.layers{l}.type = 'input';
	net.layers{l}.inputDim = 28; % => 28*28
	%net.layers{1}.a: input data will be stored here
	
	% convolution layer
	l = l + 1;
	net.layers{l}.type = 'convol';
	net.layers{l}.numFilters = 3;
	net.layers{l}.filterDim = 9; % => 9*9
	net.layers{l}.func = 'sigmoid'; % f(x) = sigmoid(x)
	
	% sub sampling / pooling layer
	l = l + 1;
	net.layers{l}.type = 'pool';
	net.layers{l}.poolDim = 2;	 % => 2*2
	net.layers{l}.func = 'identity'; % f(x) = x

	% convolution layer
	l = l + 1;
	net.layers{l}.type = 'convol';
	net.layers{l}.numFilters = 2;
	net.layers{l}.filterDim = 5; %
	net.layers{l}.func = 'sigmoid'; % f(x) = sigmoid(x)

	% sub sampling / pooling layer
	l = l + 1;
	net.layers{l}.type = 'pool';
	net.layers{l}.poolDim = 2;
	net.layers{l}.func = 'identity'; % f(x) = x
	
	%net.layers{.}.type = 'full';
	%net.layers{.}.layerDim = 100;
	%net.layers{.}.layerFunc = 'sigmoid';
	
	% last layer is always a softmax layer
	l = l + 1;
	net.layers{l}.type = 'softmax';
	net.layers{l}.numClasses = 10;

	
	% TODO add an assert to ckeck filterDim and poolDim consistency
	
	theta = cnnInitParams(net);
	
	[~, grad] = cnnCost(theta, images, labels, net);
	
	% Check gradients
	numGrad = computeNumericalGradient( @(x) cnnCost(x, images, labels, net), theta);
	
	% Use this to visually compare the gradients side by side
	disp([numGrad grad]);
	
	diff = norm(numGrad-grad)/norm(numGrad+grad);
	% Should be small. These values are usually less than 1e-9.
	disp(diff); 
	
	assert(diff < 1e-9, 'Difference too large. Check your gradient computation again');

end
