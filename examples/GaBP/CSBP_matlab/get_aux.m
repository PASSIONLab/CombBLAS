function [aux, sum_aux]=get_aux(phi, phisign, n, l, r)

[maxphirow,tmp]=size(phi);
aux=zeros(r, n);
sum_aux=zeros(1,n);

for measlist=1:maxphirow
   indset=phi(measlist,:); indset=setdiff(indset, 0);
   signset=phisign(measlist,:); signset=signset(1:length(indset));
   sum_aux(indset)=sum_aux(indset) + ones(size(indset));;
   for gg=indset
      aux(sum_aux(gg), gg)=measlist;
   end
end
