function [ frx_t ] = eval_theoretical_nrx_3d_Point2Spherical_FFP_3D( sim_params, time )

dist            = sim_params.rx_tx_distance;
rx_r            = sim_params.rx_r_inMicroMeters;
D               = sim_params.D_inMicroMeterSqrPerSecond;

part1 = rx_r / (dist + rx_r);
part2_sub1 = exp(-dist*sqrt(sim_params.lambda_degRate/D)) * erfc(dist./sqrt(4*D*time) - sqrt(sim_params.lambda_degRate*time));
part2_sub2 = exp(dist*sqrt(sim_params.lambda_degRate/D)) * erfc(dist./sqrt(4*D*time) + sqrt(sim_params.lambda_degRate*time));
nrx_cumulative = (1/2) * part1 * (part2_sub1 + part2_sub2);

% If you subtract 1-shifted version from it self, you end up with nrx at each simulation step
frx_t = nrx_cumulative - [0 nrx_cumulative(1:end-1)];

end

