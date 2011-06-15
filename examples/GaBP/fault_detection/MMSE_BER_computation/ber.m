% compute BER

function z=ber(e,snr);
if snr<=0
    z=1;
    return;
end;

z=e*qfunc(0.5*sqrt(snr)+1./sqrt(snr)*log(e/(1-e)))...
    +(1-e)*qfunc(0.5*sqrt(snr)-1./sqrt(snr)*log(e/(1-e)));

