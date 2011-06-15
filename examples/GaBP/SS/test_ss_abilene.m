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

clear all;
load aug-dec2006.mat;
[m n]=size(A);

ks = 120:140;
iter1=zeros(1,length(ks));
iter2=zeros(1,length(ks));

fprintf('\nSensor selection problem:\nNumber of sensors: m = %d\nNumber of parameters: n = %d\n', m, n);
L = []; Utilde = []; L_loc = [];  L_loc2 = []; threshold = 0.4;
i = 0;
for k = ks
    fprintf('\n\nSelecting k = %d sensors...\n', k);
    i = i + 1;
    [zhat L(i) zast Utilde(i) iter1(i)] = sens_sel_approxnt(A, k); %centralized Newton method
    [zhat2 L2(i) zast2 Utilde2(i) iter2(i)] = sens_sel_gabp(A, k); % distributed fast approximation
    [z_loc L_loc(i)] = sens_sel_loc(A, zhat2);
    [z_loc2 L_loc2(i)] = sens_sel_locr(A, k, zast2, threshold);
    fprintf('\nUtilde: %.3e, L: %.3e,  L_loc: %.3e, L_loc2: %.3e\n', Utilde(i), L(i), L_loc(i), L_loc2(i));
end
delta = Utilde - L; delta_loc = Utilde - L_loc; delta_loc2 =  Utilde - L_loc2;
delta2 = Utilde - L2;

figure; hold on;

set(gca,'FontName','times', 'FontSize', 16);
xlabel('k'); ylabel('bounds'); 
plot(ks, L, 'b-', 'LineWidth', 2);
plot(ks, L2, 'g-', 'LineWidth', 2);
plot(ks, L_loc, 'k--', 'LineWidth', 2);
plot(ks, L_loc2, 'm-.', 'LineWidth', 2);
plot(ks, Utilde, 'r:', 'LineWidth', 2);
legend('Newton method', 'GaBP', 'GaBP+Local search', 'local2','Upper bound','FontSize', 16,'Location','SouthEast');


figure; hold on;
set(gca,'FontName','times', 'FontSize', 16);
xlabel('k'); ylabel('gaps');
plot(ks, delta, 'b-', 'LineWidth', 2);
plot(ks, delta2, 'g-', 'LineWidth', 2);
plot(ks, delta_loc, 'k--', 'LineWidth', 2);
plot(ks, delta_loc2, 'm-.', 'LineWidth', 2);
legend('Newton', 'gabp', 'Local search','Local r search','FontSize', 16);

figure; hold on;
set(gca,'FontName','times', 'FontSize', 16);
xlabel('Newton Steps'); ylabel('k');
bar(ks,[iter1' iter2']);
legend('Newton', 'GaBP');

