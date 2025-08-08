%%% Sample script for IDVG measurement data format output and plotting.
%%% <acyu@mit.edu>
%%%
%%% Data format shape is (nbias, ndirections, npoints).
%%% - nbias: number of Vds drain bias steps
%%% - ndirections: number of forward/reverse sweep directions
%%% - npoints: number of points in Vgs sweep
%%% 
clear all; close all;

data = load('./keysight_id_vgs.mat')

% get number of sequences and sweep directions (fwd/rev)
data_shape = size(data.i_d);
num_bias = data_shape(1)
num_directions = data_shape(2)
num_points = data_shape(3)

% plots
h_id_vd = figure;
set(gca, 'yscale', 'log')
xlabel("v_{gs} [V]")
ylabel("i_{d} [A]")

h_ig_vd = figure;
set(gca, 'yscale', 'log')
xlabel("v_{gs} [V]")
ylabel("i_{g} [A]")

for b = 1:num_bias
for d = 1:num_directions
    % unpack
    vds = squeeze(data.v_ds(b,d,1))
    vgs = squeeze(data.v_gs(b,d,:));
    id = abs(squeeze(data.i_d(b,d,:)));
    ig = abs(squeeze(data.i_g(b,d,:)));
    
    % plot id vs. vd
    figure(h_id_vd);
    hold on;
    plot(vgs, id);
    
    % plot ig vs. vd
    figure(h_ig_vd);
    hold on;
    plot(vgs, ig);
end
end