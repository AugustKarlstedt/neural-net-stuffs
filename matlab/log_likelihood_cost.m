function [ out ] = log_likelihood_cost( a, ~ )
out = -mean(log(a));
end

