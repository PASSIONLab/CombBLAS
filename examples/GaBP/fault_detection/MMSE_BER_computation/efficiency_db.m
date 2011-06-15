% computing eta through fix-point equation,
% which minimizes the constraint

function eta= efficiency_cal(e,mu,snr)

F = @(x) 1+snr/(e*mu)*mmse(e,x*snr)-1/x - 100000*min(x,0);
eta_temp = zeros(1,4);
fval_temp = zeros(1,4);
exitflag = zeros(1,4);
constr = zeros(1,4);
for i = 1:4
    [eta_temp(i) fval_temp(i) exitflag(i)] = fsolve(F,0.25*i-0.2);
    constr(i) = constraint(e,mu,snr,eta_temp(i));
end

eta_temp

index = find(exitflag == 1);
eta = eta_temp(min(find(constr == min(constr(index)))));

return;