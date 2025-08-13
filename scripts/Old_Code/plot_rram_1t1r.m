%%% Sample script for RRAM 1T1R measurement data format output and plotting.
%%% <acyu@mit.edu>
%%%
%%% Data format shape is (nsequences, ndirections, npoints_max).
%%% - nsequences: number of measurement sequences, e.g. a form/reset/set sequence:
%%%     "frsrsr" -> 6 sequences
%%% - ndirections: number of forward/reverse sweep directions = 2
%%% - npoints_max: maximum number of points in a sweep sequence, e.g. if we have
%%%     v_d sequences:
%%%     v_d_form = [0, 1, 2, 3, 4]   -> npoints_form = 5
%%%     v_d_set = [0, 1, 2]          -> npoints_set = 3
%%%     v_d_reset = [0, -1, -2, -3]  -> npoints_reset = 4
%%%     npoints_max = max(npoints_form, npoints_set, npoints_reset) = 5
%%% 
%%% Data block contains groups of data with np.nan padding to match npoints_max:
%%%     v_g = 
%%%         FORM  [0,   1,   2,   3,   4]
%%%         RESET [0,  -1,  -2,  -3, nan]
%%%         SET   [0,   1,   2, nan, nan]
%%%         RESET [0,  -1,  -2,  -3, nan]
%%%         SET   [0,   1,   2, nan, nan]
%%%         RESET [0,  -1,  -2,  -3, nan]
%%%
clear all; close all;

data = load('./data/keysight_rram_1t1r.mat')

% get number of sequences and sweep directions (fwd/rev)
data_shape = size(data.i_d);
num_sequences = data_shape(1)
num_directions = data_shape(2)

% get sequence step names
step_names = data.step_names

% get number of points in each sequence step
step_num_points = data.num_points

% plots
h_id_vd = figure;
set(gca, 'yscale', 'log')
xlabel("v_{d} [V]")
ylabel("i_{d} [A]")

h_ig_vd = figure;
set(gca, 'yscale', 'log')
xlabel("v_{d} [V]")
ylabel("i_{g} [A]")

h_res_vd = figure; hold on;
set(gca, 'yscale', 'log')
xlabel("v_{d} [V]")
ylabel("R_{d} [Ohm]")

for s = 1:num_sequences
for d = 1:num_directions
    % unpack
    n = step_num_points(s)
    vs = data.v_s(s,d,1);
    vd = squeeze(data.v_d(s,d,1:n));
    vg = squeeze(data.v_g(s,d,1:n));
    id = squeeze(data.i_d_abs(s,d,1:n));
    ig = squeeze(data.i_g_abs(s,d,1:n));
    res = squeeze(data.res(s,d,1:n));
    
    % plot id vs. vd
    figure(h_id_vd);
    hold on;
    plot(vd, id);
    
    % plot res vs. vd (res = vd / id)
    figure(h_res_vd);
    hold on;
    plot(vd, res);
    
    % plot ig vs. vd
    figure(h_ig_vd);
    hold on;
    plot(vd, ig);
end
end