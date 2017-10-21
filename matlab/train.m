function [ weights, biases, train_costs, test_costs, validation_costs, train_accuracies, test_accuracies, validation_accuracies ] = train( inputs, targets, nodeLayers, numEpochs, batchSize, eta, split, max_fail )
%train SUMMARY
%   DETAILED EXPLANATION

if (length(inputs) ~= length(targets))
   fprintf('Length of inputs not equal to outputs\n');
   return;
end

% split up the train, test, validation sets
if (sum(split) ~= 100)
    fprintf('Split does not add up to 100%%\n');
    return;
end

n = length(inputs);

% convert split inputs into percentages
train_split = split(:, 1) / 100;
test_split = split(:, 2) / 100;
validation_split = split(:, 3) / 100;

% calculate how many of each set do we need
train_count = round(n * train_split); % TODO: is round correct?
test_count = floor(n * test_split); % TODO: is floor correct?
validation_count = ceil(n * validation_split); % TODO: is ceil correct?

if (train_count + test_count + validation_count ~= n)
    fprintf('Cannot split data as requested\n'); 
    return;
end

% randomly select indices for train, test, validation data
indices = randperm(n);
train_indices = indices(:, 1:train_count);
indices(:, 1:train_count) = [];
test_indices = indices(:, 1:test_count);
indices(:, 1:test_count) = [];
validation_indices = indices(:, 1:validation_count);
indices(:, 1:validation_count) = [];

if (~isempty(indices))
    fprintf('There were leftover indices after splitting. This shouldn''t happen\n');
    return;
end

% finally, split up the data based on the random indices
train_inputs = inputs(:, train_indices);
train_targets = targets(:, train_indices);
test_inputs = inputs(:, test_indices);
test_targets = targets(:, test_indices);
validation_inputs = inputs(:, validation_indices);
validation_targets = targets(:, validation_indices);

% initialize some stuff
% and update our example count (since we split)
n = length(train_inputs);
miniBatchSize = min(batchSize, n);
layerCount = length(nodeLayers);

% store costs and accuracies for each set of data
train_costs = zeros(1, numEpochs);
test_costs = zeros(1, numEpochs);
validation_costs = zeros(1, numEpochs);
train_accuracies = zeros(1, numEpochs);
test_accuracies = zeros(1, numEpochs);
validation_accuracies = zeros(1, numEpochs);

% weights, biases initialization
biases = cell(1, layerCount-1);
weights = cell(1, layerCount-1);

% "better" initialization
% makes neurons less likely to saturate
weights_standard_deviation = 1 / sqrt(n);
weights_mean = 0;

for i = 2:layerCount
    biases{i-1} = randn(nodeLayers(i), 1);
    weights{i-1} = weights_standard_deviation .* randn(nodeLayers(i), nodeLayers(i-1)) + weights_mean;
end

fprintf('    |          TRAIN           ||           TEST           ||        VALIDATION        \n');
fprintf('---------------------------------------------------------------------------------------\n');
fprintf('Ep  |  Cost  |  Corr  |  Acc   ||  Cost  |  Corr  |  Acc   ||  Cost  |  Corr  |  Acc   \n');
fprintf('---------------------------------------------------------------------------------------\n');

for currentEpoch = 1:numEpochs
    
    % print out the header every 25 epochs so it's easier to read 
    if (mod(currentEpoch, 25) == 0)
        fprintf('Ep  |  Cost  |  Corr  |  Acc   ||  Cost  |  Corr  |  Acc   ||  Cost  |  Corr  |  Acc   \n');
    end
    fprintf('%4i|  ', currentEpoch);   
    
    % shuffle data and process in batches
    indices = randperm(n);
    miniBatchCount = ceil(n / miniBatchSize);
    
    for currentMiniBatch = 1:miniBatchCount
        indicesCount = min(miniBatchSize, length(indices));
        miniBatchIndices = indices(:, 1:indicesCount);
        indices(:, 1:indicesCount) = [];
        
        % inputs
        x = train_inputs(:, miniBatchIndices);
        y = train_targets(:, miniBatchIndices);

        % setup for forward step
        zs = cell(1, layerCount);
        activations = cell(1, layerCount);

        % setup for backward step
        nabla_biases = cell(1, layerCount-1);
        nabla_weights = cell(1, layerCount-1);

        for i = 2:layerCount
            nabla_biases{i-1} = zeros(nodeLayers(i), 1);
            nabla_weights{i-1} = zeros(nodeLayers(i), nodeLayers(i-1));
        end

        % for each training example
        for ex = 1:indicesCount
            activations{1} = x(:, ex);
            zs{1} = x(:, ex);

            % forward step
            for l = 2:layerCount
                z = weights{l-1} * activations{l-1} + biases{l-1};
                zs{l} = z;
                activations{l} = arrayfun(@logsig, z);
            end

            % error calculation
            delta_nabla_biases = cell(1, layerCount-1);
            delta_nabla_weights = cell(1, layerCount-1);

            error = activations{end} - y(:, ex);
            delta = error .* arrayfun(@logsig_prime, zs{end});
            delta_nabla_biases{end} = delta;
            delta_nabla_weights{end} = delta * activations{end-1}';

            % backward step
            for l = (layerCount-1):-1:2
                delta = weights{l}' * delta .* arrayfun(@logsig_prime, zs{l});
                delta_nabla_biases{l-1} = delta;
                delta_nabla_weights{l-1} = delta * activations{l-1}';
            end

            % stored for weight updates
            for l = 1:(layerCount-1)
                nabla_biases{l} = nabla_biases{l} + delta_nabla_biases{l};
                nabla_weights{l} = nabla_weights{l} + delta_nabla_weights{l};
            end

        end
        
        % weight updates
        for l = 1:(layerCount-1)
            biases{l} = biases{l} - (eta / indicesCount) * nabla_biases{l};
            weights{l} = weights{l} - (eta / indicesCount) * nabla_weights{l};
        end
        
    end
   
    
    %
    % TODO: vvv This gets ugly. Refactor vvv
    % EVALUATE TRAINING SET
    %

    % inputs
    x = train_inputs;
    y = train_targets;
    
    n = length(x(1, :));

    % outputs for calculating MSE
    outputs = zeros(length(y(:, 1)), n);
    correct = 0;

    % setup for forward step
    activations = cell(1, layerCount);

    % for each input example
    for ex = 1:n
        activations{1} = x(:, ex);

        % forward step
        for l = 2:layerCount
            z = weights{l-1} * activations{l-1} + biases{l-1};
            activations{l} = arrayfun(@logsig, z);
        end

        % error calculation
        correct = correct + isequal(round(activations{end}), y(:, ex));

        error = activations{end} - y(:, ex);
        outputs(:, ex) = error;  % store output for MSE calculation
    end   

    train_cost = sum(reshape(outputs, [1 numel(outputs)]) .^ 2) / n;
    train_costs(currentEpoch) = train_cost;
    train_accuracy = correct / n;
    train_accuracies(currentEpoch) = train_accuracy;
    fprintf('%.3f |  %i/%i  |  %.3f ||  ', train_cost, correct, n, train_accuracy);   

    %
    % TODO: ^^^ This gets ugly. Refactor ^^^
    %
    
    %
    % TODO: vvv This gets ugly. Refactor vvv
    % EVALUATE TEST SET
    %

    % inputs
    x = test_inputs;
    y = test_targets;
    
    n = length(x(1, :));
    
    % outputs for calculating MSE
    outputs = zeros(length(y(:, 1)), n);
    correct = 0;

    % setup for forward step
    activations = cell(1, layerCount);
        
    % for each input example
    for ex = 1:n
        activations{1} = x(:, ex);

        % forward step
        for l = 2:layerCount
            z = weights{l-1} * activations{l-1} + biases{l-1};
            activations{l} = arrayfun(@logsig, z);
        end

        % error calculation
        correct = correct + isequal(round(activations{end}), y(:, ex));

        error = activations{end} - y(:, ex);
        outputs(:, ex) = error;  % store output for MSE calculation
    end   

    test_cost = sum(reshape(outputs, [1 numel(outputs)]) .^ 2) / n;
    test_costs(currentEpoch) = test_cost;
    test_accuracy = correct / n;
    test_accuracies(currentEpoch) = test_accuracy;
    fprintf('%.3f |  %i/%i  |  %.3f ||  ', test_cost, correct, n, test_accuracy); 
    
    %
    % TODO: ^^^ This gets ugly. Refactor ^^^
    %
    
    %
    % TODO: vvv This gets ugly. Refactor vvv
    % EVALUATE VALIDATION SET
    %

    % inputs
    x = validation_inputs;
    y = validation_targets;
    
    n = length(x(1, :));

    % outputs for calculating MSE
    outputs = zeros(length(y(:, 1)), n);
    correct = 0;

    % setup for forward step
    activations = cell(1, layerCount);

    % for each input example
    for ex = 1:n
        activations{1} = x(:, ex);

        % forward step
        for l = 2:layerCount
            z = weights{l-1} * activations{l-1} + biases{l-1};
            activations{l} = arrayfun(@logsig, z);
        end

        % error calculation
        correct = correct + isequal(round(activations{end}), y(:, ex));

        error = activations{end} - y(:, ex);
        outputs(:, ex) = error;  % store output for MSE calculation
    end   
    
    validation_cost = sum(reshape(outputs, [1 numel(outputs)]) .^ 2) / n;
    validation_costs(currentEpoch) = validation_cost;
    validation_accuracy = correct / n;
    validation_accuracies(currentEpoch) = validation_accuracy;
    fprintf('%.3f |  %i/%i  |  %.3f \n', validation_cost, correct, n, validation_accuracy); 
    
    %
    % TODO: ^^^ This gets ugly. Refactor ^^^
    %
        
    % stop early if all validation correct
    % OR if error has increased during the last max_fail epochs
    
    error_has_increased = false;
    if (currentEpoch > max_fail + 1)	
        error_has_increased = all(validation_costs(:, currentEpoch-max_fail:currentEpoch) > validation_costs(:, currentEpoch-max_fail-1));
        if (error_has_increased)
            fprintf('Validation error has increased over the last %i epochs. Stopping.\n', max_fail);
        end        
    end
    
    if (correct == n || error_has_increased)
        % trim the accuracies if we stop early
        train_accuracies = train_accuracies(:, 1:currentEpoch);
        test_accuracies = test_accuracies(:, 1:currentEpoch);
        validation_accuracies = validation_accuracies(:, 1:currentEpoch);
        break
    end
    
end



end