function cnnCheckGradients()

ei = [];

ei.early_stop = 0;
ei.validationSet = 0.1; % validation set = 10% of training set
ei.early_max = 10;

ei.verbose = 1;


ei.input_dim = 8;
ei.output_dim = 10;
ei.layer_sizes = [9, 8, 7, ei.output_dim];
ei.lambda = 0.1;
ei.activation_fun = 'relu';

nSamples = 100;
data = randn(ei.input_dim, nSamples);
labels = randi(ei.output_dim, 1, nSamples);

%% setup random initial weights
stack = initialize_weights(ei);
params = stack2params(stack);

[cost, grad] = supervised_dnn_cost(params, ei, data, labels);

numGrad = computeNumericalGradient( @(x) supervised_dnn_cost(x, ei, data, labels), params);

% Use it to visually compare the gradients side by side
disp([numGrad grad]); 

  % Compare numerically computed gradients with those computed analytically
diff = norm(numGrad-grad) / norm(numGrad+grad);
disp(diff); 

assert(diff < 1e-8, 'Difference too large. Check your gradient computation again');

end
