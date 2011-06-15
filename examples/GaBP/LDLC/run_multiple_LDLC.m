

% example script that runs the extended LDLC decoder for solving
% the multiuser detection problem
% Written by Danny Bickson.
% updated: 18-Dec-2008
%
% Supplamentary material of the paper: 
% "Low density lattice decoder via non-parametric belief propagation"
% By D. Bickson, A. T. Ihler, and D. Dolev,
% Submitted to ISIT 2009.
%
function [pscore,lscore] = run_multiple_LDLC()

N = 20;

pscore = 0;
lscore = 0;
for i=1:N
    [x0,x1,y,trans] = run_LDLC(2);
    if (sum(x' ~= trans) == 0) 
        pscore = pscore +1;
    end
    if (sum(y' ~= trans) == 0) 
        lscore = lscore + 1;
    end
end

disp(['total LDLC : ', num2str(pscore0/N), ' total extended LDLC: ', num2str(pscore1/N), ' total GaBP: ', num2str(lscore/N)]);
end

