function [ out ] = tanh_prime( n )
out = 1 - pow2(tanh(n));
end