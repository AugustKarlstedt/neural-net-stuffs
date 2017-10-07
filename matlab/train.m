function [ weights, biases, accuracies ] = train( inputs, targets, nodeLayers, numEpochs, batchSize, eta )
%train SUMMARY
%   DETAILED EXPLANATION

% weight, bias initialization
n = length(inputs);
miniBatchSize = min(batchSize, n);
layerCount = length(nodeLayers);
biases = cell(1, layerCount-1);
weights = cell(1, layerCount-1);
accuracies = zeros(1, numEpochs);

for i = 2:layerCount
    biases{i-1} = randn(nodeLayers(i), 1);
    weights{i-1} = randn(nodeLayers(i), nodeLayers(i-1));
end

for currentEpoch = 1:numEpochs
    
    % shuffle data and process in batches
    indices = randperm(n);
    miniBatchCount = ceil(n / miniBatchSize);
    
    for currentMiniBatch = 1:miniBatchCount
        indicesCount = min(miniBatchSize, length(indices));
        miniBatchIndices = indices(:, 1:indicesCount);
        indices(:, 1:indicesCount) = [];
        
        % inputs
        x = inputs(:, miniBatchIndices);
        y = targets(:, miniBatchIndices);

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
        for ex = 1:miniBatchSize
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
    %

    % inputs
    x = inputs;
    y = targets;

    % outputs for calculating MSE
    outputs = zeros(nodeLayers(end), n);
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

    
    %
    % TODO: ^^^ This gets ugly. Refactor ^^^
    %
    
    
    mse = sum(reshape(outputs, [1 numel(outputs)]) .^ 2) / n;
    accuracy = correct / n;
    accuracies(currentEpoch) = accuracy;
    fprintf('[%s] Epoch %i, MSE: %.4f, Correct: %i / %i, Acc: %.4f\n', datestr(now, 'HH:MM:SS'), currentEpoch, mse, correct, n, accuracy);   

    % stop early if all correct
    if (correct == n)
        accuracies = accuracies(:, 1:currentEpoch);
        break
    end
    
end



end