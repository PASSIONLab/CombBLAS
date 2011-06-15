function []=verify_pdf(pdf)
   assert(sum(pdf < 0)==0);
   assert(sum(isnan(pdf))==0);
   assert(sum(isinf(pdf))==0);
   assert(length(unique(pdf)) > 1);
   assert(sum(~isreal(pdf)) == 0);
end