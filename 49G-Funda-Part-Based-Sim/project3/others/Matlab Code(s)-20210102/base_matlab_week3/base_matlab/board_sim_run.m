%% Set Parameters

sim_params.rx_center                = [0, 0, 0];
sim_params.rx_r_inMicroMeters       = 5;
sim_params.rx_tx_distance           = 5;
sim_params.tx_emission_pt           = sim_params.rx_center + [sim_params.rx_tx_distance+sim_params.rx_r_inMicroMeters, 0, 0];
sim_params.D_inMicroMeterSqrPerSecond = 100;

sim_params.tend                     = 0.4;
sim_params.delta_t                  = 0.0001;
sim_params.num_molecules            = 50000;

%% SIMULATE

fprintf('\nSimulation <sim_gaussianRW_Point2Spherical_FFP_3D> \t\t[START]')
tstart = tic;
[nrx_sim_timeline, time] = sim_gaussianRW_Point2Spherical_FFP_3D(sim_params);
fprintf('\nSimulation <sim_gaussianRW_Point2Spherical_FFP_3D> \t\t[End] \tDuration = %f\n', toc(tstart))


%% THEORETICAL NRX

fprintf('\nTheoretical Formula \t\t[START]')
tstart = tic;
[nrx_theory_timeline] = eval_theoretical_nrx_3d_Point2Spherical_FFP_3D(sim_params, time);
fprintf('\nTheoretical Formula  \t\t[End] \tDuration = %f\n', toc(tstart))


%% PLOT

merge_cnt = 10;
[nrx_sim_timeline_merged, time_merged] = helper_merge_timeline(merge_cnt, nrx_sim_timeline, time);
[nrx_theory_timeline_merged, ~] = helper_merge_timeline(merge_cnt, nrx_theory_timeline, time);


hFig = figure;
set(gcf,'PaperPositionMode','auto')
set(hFig, 'Position', [0 101 600 400])

plot(time_merged, nrx_sim_timeline_merged/sim_params.num_molecules, '-', 'LineWidth', 2)
hold on
plot(time_merged, nrx_theory_timeline_merged, '--', 'LineWidth', 2)
grid on
xlabel('Time - (s)')
ylabel('Average Fraction of Received Molecules in \Delta t')
legend('Simulation', 'Theory');
title(['\Deltat=', num2str(merge_cnt*sim_params.delta_t), '; r_{rx}=', num2str(sim_params.rx_r_inMicroMeters), '; dist=', num2str(sim_params.rx_tx_distance), '; D=', num2str(sim_params.D_inMicroMeterSqrPerSecond)])




