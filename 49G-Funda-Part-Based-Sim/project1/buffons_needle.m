needle_size = 5;
line_distance = 5;
n_needle = 10^6;


needle_matrix = rand(n_needle, 2); % center height, angle

needle_matrix(:,1) = needle_matrix(:,1)*(line_distance/2);
needle_matrix(:,2) = needle_matrix(:,2)*pi;

crossing_mask  = needle_matrix(:,1) <= (needle_size/2)*sin(needle_matrix(:,2));

n_crossing = sum(crossing_mask);

est_pi = (2 * needle_size * n_needle) / (line_distance * n_crossing);

disp(abs(est_pi - pi))


%% Plot

cum_cross = cumsum(crossing_mask);

cum_est_pi = (2 * needle_size * cumsum(ones(n_needle,1))) ./ (line_distance * cum_cross);

cum_est_pi = abs(cum_est_pi - pi);

figure('Name', 'Matrix')
subplot(5,1,1)
plot(cum_est_pi(1:100))
xlabel("Number of Needles")
ylabel("ABS. Error")
title("10^2 Needles")
subplot(5,1,2)
plot(cum_est_pi(1:10^3))
xlabel("Number of Needles")
ylabel("ABS. Error")
title("10^3 Needles")
subplot(5,1,3)
plot(cum_est_pi(1:10^4))
xlabel("Number of Needles")
ylabel("ABS. Error")
title("10^4 Needles")
subplot(5,1,4)
plot(cum_est_pi(1:10^5))
xlabel("Number of Needles")
ylabel("ABS. Error")
title("10^5 Needles")
subplot(5,1,5)
plot(cum_est_pi(1:10^6))
xlabel("Number of Needles")
ylabel("ABS. Error")
title("10^6 Needles")







