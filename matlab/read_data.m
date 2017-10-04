function [ inputs, targets ] = read_data( filename, feature_count)
%READ_DATA Reads in a csv file assuming the last feature_count columns are the target
%   Detailed explanation goes here

csv_data = csvread(filename);
data = csv_data;
inputs = data(:, 1:end-feature_count)';
targets = data(:, end-feature_count+1:end)';

end

