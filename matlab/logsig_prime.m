function [ out ] = logsig_prime( n )
out = logsig(n) * (1 - logsig(n));
end