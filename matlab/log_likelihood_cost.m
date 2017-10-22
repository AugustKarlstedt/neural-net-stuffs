function [ out ] = log_likelihood_cost( a, ~ )
out = sum(-log(a));
end

