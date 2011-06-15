% Source code for the paper:
% Distributed Sensor Selection via a Truncated Newton Method
% by Danny Bickson and Danny Dolev
% submitted for publication
% http://arxiv.org/abs/0907.0931

% Original simulation generates the numerical example in the paper
% Sensor Selection via Convex Optimization
% www.stanford.edu/~boyd/papers/sensor_selection.html
% May 2008 Siddharth Joshi & Stephen Boyd
%
% Modified and created a distributed version: July 2009 by Danny Bickson
% Source code is available on http://www.cs.huji.ac.il/labs/danss/p2p/gabp/

addpath('..');

clear all;
m = 100; % number of sensors
n = 20; % dimension of x to be estimated
ks = 20:1:40; % target number of sensors to use
%ks = 20;

randn('state', 0);
A = randn(m,n)/sqrt(n);

fprintf('\nSensor selection problem:\nNumber of sensors: m = %d\nNumber of parameters: n = %d\n', m, n);
L = []; Utilde = []; L_loc = [];  L_loc2 = []; threshold = 0.4;
i = 0;
for k = ks
    fprintf('\n\nSelecting k = %d sensors...\n', k);
    i = i + 1;
    [zhat L(i) zast Utilde(i)] = sens_sel_approxnt(A, k); % centralized Newton method
    [zhat3 L3(i) zast3 Utilde3(i)] = sens_sel_approxnt_dist(A, k);% distributed Newton method
    [zhat2 L2(i) zast2 Utilde2(i)] = sens_sel_gabp(A, k); % fast approximation
    [z_loc L_loc(i)] = sens_sel_loc(A, zhat); % local optimization
    [z_loc2 L_loc2(i)] = sens_sel_locr(A, k, zast, threshold);
    fprintf('\nUtilde: %.3e, L: %.3e,  L_loc: %.3e, L_loc2: %.3e\n', Utilde(i), L(i), L_loc(i), L_loc2(i));
end
delta = Utilde - L; delta_loc = Utilde - L_loc; delta_loc2 =  Utilde - L_loc2;
delta2 = Utilde - L2;

figure; hold on;
set(gca,'FontName','times', 'FontSize', 16);
xlabel('k'); ylabel('bounds'); 
plot(ks, L, 'b-', 'LineWidth', 2);
plot(ks, L3, 'g-', 'LineWidth', 2);
plot(ks, L2, 'g-', 'LineWidth', 2);
plot(ks, L_loc, 'k--', 'LineWidth', 2);
plot(ks, Utilde, 'r:', 'LineWidth', 2);

legend('Cholesky', 'Dist-IP', 'GaBP', 'GaBP+Greedy', 'Upper bound','FontSize', 16,'Location','SouthEast');

figure; hold on;
set(gca,'FontName','times', 'FontSize', 16);
xlabel('k'); ylabel('gaps');
plot(ks, delta, 'b-', 'LineWidth', 2);
plot(ks, delta2, 'g-', 'LineWidth', 2);
plot(ks, delta_loc2, 'm-.', 'LineWidth', 2);
legend('Cholesky', 'GaBP','GaBP+Greedy','FontSize', 16);
