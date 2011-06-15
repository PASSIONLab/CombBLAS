function [out] = integral_ekde(in, weight,single)
    if (nargin < 3)
        single = 0;
    end
    vij = -weight^2.*unique(getBW(in)).^2;  
    mij = -weight*getPoints(in);
    out = kde(mij./vij, 1./sqrt(vij));
    verify_kde(out,single);
end