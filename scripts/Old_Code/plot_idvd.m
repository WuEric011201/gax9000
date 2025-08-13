%%% Sample script for IDVG measurement data format output and plotting.
%%% <acyu@mit.edu>
%%%
%%% Data format shape is (nbias, ndirections, npoints).
%%% - nbias: number of Vds drain bias steps
%%% - ndirections: number of forward/reverse sweep directions
%%% - npoints: number of points in Vgs sweep
%%% 
clear all; close all;

data = load('./data/keysight_id_vds.mat');

% get number of sequences and sweep directions (fwd/rev)
data_shape = size(data.i_d);
num_bias = data_shape(1);
num_directions = data_shape(2);
num_points = data_shape(3);

% plots
h_id_vd = figure;
xlabel("v_{ds} [V]")
ylabel("i_{d} [A]")
title("I_{D} vs V_{DS} for Different V_{GS} Sweeps")
hold on;

h_ig_vd = figure;
xlabel("v_{ds} [V]")
ylabel("i_{g} [A]")
set(gca, 'yscale', 'log')
title("I_{G} vs V_{DS} for Different V_{GS} Sweeps")
hold on;

legend_labels_id = {};
legend_labels_ig = {};

for b = 1:num_bias
    for d = 1:num_directions
        % unpack
        vds = squeeze(data.v_ds(b,d,:));
        vgs = squeeze(data.v_gs(b,d,1));  % assuming Vgs is constant per sweep
        id = squeeze(data.i_d(b,d,:));
        ig = squeeze(data.i_g(b,d,:));

        % plot id vs. vd
        figure(h_id_vd);
        plot(vds, id);
        legend_labels_id{end+1} = sprintf('V_{GS} = %.2f V, %s', vgs, direction_label(d));

        % plot ig vs. vd
        figure(h_ig_vd);
        plot(vds, ig);
        legend_labels_ig{end+1} = sprintf('V_{GS} = %.2f V, %s', vgs, direction_label(d));
    end
end

% apply legends
figure(h_id_vd);
legend(legend_labels_id, 'Location', 'best');

figure(h_ig_vd);
legend(legend_labels_ig, 'Location', 'best');

% ---- Helper function ----
function label = direction_label(d)
    if d == 1
        label = 'Forward';
    else
        label = 'Reverse';
    end
end
