function [ out ] = train( inputs, targets, nodeLayers, numEpochs, batchSize, eta )
%train SUMMARY
%   DETAILED EXPLANATION

% weight, bias initialization
layerCount = length(nodeLayers);

biases = cell(1, layerCount-1);
weights = cell(1, layerCount-1);

for i = 2:layerCount
    biases{i-1} = randn(nodeLayers(i), 1);
    weights{i-1} = randn(nodeLayers(i), nodeLayers(i-1));
end

for i = 1:numEpochs
    
    % shuffle data
    n = length(inputs);
    miniBatchSize = min(batchSize, n);
    miniBatchIndices = zeros(1, miniBatchSize);
    
    for j = 1:miniBatchSize
        index = randi(n);
        while (ismember(index, miniBatchIndices))
            index = randi(n);
        end
        miniBatchIndices(j) = index;
    end
    
    % inputs
    x = inputs(:, miniBatchIndices);
    y = targets(:, miniBatchIndices);
    
    % outputs for calculating MSE
    outputs = zeros(nodeLayers(end), miniBatchSize);
    correct = 0;
    
    % setup for forward step
    zs = cell(1, layerCount);
    activations = cell(1, layerCount);
    
    % setup for backward step
    nabla_biases = cell(1, layerCount-1);
    nabla_weights = cell(1, layerCount-1);
    
    for j = 2:layerCount
        nabla_biases{j-1} = zeros(nodeLayers(j), 1);
        nabla_weights{j-1} = zeros(nodeLayers(j), nodeLayers(j-1));
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
        
        correct = correct + isequal(round(activations{end}), y(:, ex));
        
        error = activations{end} - y(:, ex);
        outputs(:, ex) = error;  % store output for MSE calculation
        
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
        biases{l} = biases{l} - (eta / miniBatchSize) * nabla_biases{l};
        weights{l} = weights{l} - (eta / miniBatchSize) * nabla_weights{l};
    end
    
    mse = sum(reshape(outputs, [1 numel(outputs)]) .^ 2) / miniBatchSize;
    accuracy = correct / miniBatchSize;
    fprintf('[%s] Epoch %i, MSE: %.4f, Correct: %i / %i, Acc: %.4f\n', datestr(now, 'HH:MM:SS'), i, mse, correct, miniBatchSize, accuracy);   

    if (correct == miniBatchSize)
        break
    end
    
end



end