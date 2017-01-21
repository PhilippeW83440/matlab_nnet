function [opttheta] = minFuncSGD(funObj, theta, train_data, train_labels, valid_data, valid_labels,...
                        options, ei)
% Runs stochastic gradient descent with momentum to optimize the
% parameters for the given objective.
%
% Parameters:
%  funObj     -  function handle which accepts as input theta,
%                train_data, train_labels and returns cost and gradient w.r.t
%                to theta.
%  theta      -  unrolled parameter vector
%  train_data 		-  stores data in m x n x numExamples tensor
%  train_labels     -  corresponding train_labels in numExamples x 1 vector
%  options    -  struct to store specific options for optimization
%
%  valid_data & valid_labels: Cross Validation set
%
% Returns:
%  opttheta   -  optimized parameter vector
%
% Options (* required)
%  epochs*     - number of epochs through train_data
%  alpha*      - initial learning rate
%  minibatch*  - size of minibatch
%  momentum    - momentum constant, defualts to 0.9


%%======================================================================
%% Setup
assert(all(isfield(options,{'epochs','alpha','minibatch'})),...
        'Some options not defined');
if ~isfield(options,'momentum')
    options.momentum = 0.9;
end;
epochs = options.epochs;
alpha = options.alpha;
minibatch = options.minibatch;
m = length(train_labels); % training set size
% Setup for momentum
mom = 0.5;
momIncrease = 20;
velocity = zeros(size(theta));

%TEST XXX PWE
[~, ~, valid_preds] = funObj(theta, valid_data, valid_labels, true);
acc = 100 * sum(valid_preds == valid_labels) / length(valid_preds);
fprintf('Accuracy initiale is %f\n', acc);


best_acc = acc;
best_theta = theta;

% train_cost & valid_cost
train_cost = zeros(1, epochs);
valid_cost = zeros(1, epochs);

% train_accuracy & valid_accuracy
train_acc = zeros(1, epochs);
valid_acc = zeros(1, epochs);

close all;


% early_stop strategy
plot_runtime = ei.plot_runtime;
early_stop = ei.early_stop;
early_max_alpha = ei.early_max_alpha;
early_max_stop = ei.early_max_stop;
early_count = 0;

dnn_name = sprintf('dnn%s.mat', datestr(now));
dnn_name = strrep(dnn_name, ' ', '_');

best_acc_val = [];
best_acc_time = [];

%%======================================================================
%% SGD loop
it = 0;
t0 = tic;
for e = 1:epochs
    
    % randomly permute indices of train_data for quick minibatch sampling
    rp = randperm(m);
    
	tstart = tic;
    for s=1:minibatch:(m-minibatch+1)
        it = it + 1;

        % increase momentum after momIncrease iterations
        if it == momIncrease
            mom = options.momentum;
        end;

        % get next randomly selected minibatch
        %mb_data = train_data(:, :, rp(s:s+minibatch-1));
        mb_data = train_data(:, rp(s:s+minibatch-1));
        mb_labels = train_labels(rp(s:s+minibatch-1));

        % evaluate the objective function on the next minibatch
        [~, grad, ~] = funObj(theta, mb_data, mb_labels, false);
        
        % Instructions: Add in the weighted velocity vector to the
        % gradient evaluated above scaled by the learning rate.
        % Then update the current weights theta according to the
        % sgd update rule
        
        %%% Philippe WEINGERTNER September 2014 %%%
HINTON = 0;
if HINTON
		% correct formula
		velocity = mom * velocity - alpha * grad;
		theta = theta + velocity;
else
		% incorrect formula
		velocity = mom * velocity + alpha * grad;
		theta = theta - velocity;
end
        %fprintf('Epoch %d: Cost on iteration %d is %f\n',e,it,cost);
    end;

	timeEpoch = toc(tstart);
	timeRun = toc(t0);

    [train_cost(e), ~, train_preds] = funObj(theta, train_data, train_labels, true);
	train_acc(e) = 100 * sum(train_preds == train_labels) / length(train_preds);

    [valid_cost(e), ~, valid_preds] = funObj(theta, valid_data, valid_labels, true);
	valid_acc(e) = 100 * sum(valid_preds == valid_labels) / length(valid_preds);

	if (valid_acc(e) > best_acc)
		best_acc = valid_acc(e);
		best_theta = theta;
		save(dnn_name, 'best_theta', 'best_acc', 'alpha', 'options', 'ei');
		early_count = 0;

		best_acc_val = [ best_acc_val; best_acc];
		batime = round(timeRun);
		best_acc_time = [ best_acc_time; batime];

	elseif early_stop
		early_count = early_count + 1;
		if early_count >= early_max_stop
			break;
		elseif early_count >= early_max_alpha
			alpha = alpha / 2;
			early_count = 0;
		end
	end

    % aneal learning rate by factor of two after each epoch
    %alpha = alpha / 2.0;

    fprintf('Epoch %d: iteration %d - timeEpoch %f - timeRun %f \n', e, it, timeEpoch, timeRun);
    fprintf(' - train_cost %f - valid_cost %f \n', train_cost(e), valid_cost(e));
    fprintf(' - train_acc  %f - valid_acc  %f  - best_acc %f (in %d seconds) \n', train_acc(e), valid_acc(e), best_acc, batime);
	fprintf(' - alpha = %f\n', alpha);

	if (plot_runtime)
		figure(1);
		plot(1:e, train_cost(1:e), 1:e, valid_cost(1:e));
		legend('Training', 'Validation');
		ylabel('Cost');
		xlabel('Epoch');
		title('Cost Error');
		grid on;

		figure(2);
		plot(1:e, train_acc(1:e), 1:e, valid_acc(1:e));
		legend('Training', 'Validation');
		ylabel('Accuracy');
		xlabel('Epoch');
		title('Classification Accuracy');
		grid on;

		figure(3);
		plot(best_acc_time, best_acc_val, 'k');
		ylabel('Accuracy');
		xlabel('Seconds');
		title('Best Classification Accuracy on Validation Set');
		grid on;

		drawnow;
	end;

end;

%opttheta = theta;
opttheta = best_theta;

end
