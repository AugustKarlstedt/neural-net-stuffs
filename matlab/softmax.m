function [ out ] = softmax( n )
out = exp(n) / sum(exp(n));
end

