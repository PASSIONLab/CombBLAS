function [] = verify_kde(current,single)
    if (nargin < 2)
        single = 0;
    end
   assert (sum(isnan(getPoints(current))) == 0);
%    if (length(getPoints(current)) > 1) 
%        assert(length(unique(getPoints(current))) ~= 1);
%    end
   if (single == 1)
       assert (length(unique(getBW(current))) == 1);
   end
   uni = unique(getBW(current));
   for i=1:length(uni)
      if (~isreal(uni(i)))
          assert(uni(i).^2<0)
      end
   end
end