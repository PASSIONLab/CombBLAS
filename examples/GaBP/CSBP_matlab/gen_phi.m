function [phi]=gen_phi (n, l, r)

maxphirow=floor(n*r/l);
done=0;
attempt=0;

while (~done)
   attempt=attempt+1;
   disp(sprintf('  attempt: %d',attempt));
   phi=zeros(maxphirow,l);
   nperm=[];
   for gg=1:r
      nperm=[nperm randperm(n)];
   end
   done=1;
   for measlist=1:maxphirow
      ind1=1+(measlist-1)*l;
      ind2=ind1+l-1;
      indset=nperm(ind1:ind2);
      indset=unique(indset);
      if (length(indset) < l)
         %% option 1
         if (0)
            done=0;
            break;
         end
         %% option 2
         %if (0)
         while (length(indset) < l)
            ll=l-length(indset);
            bigset=setdiff((1:n), indset);
            scrambleind=randperm(n-length(indset));
            bigset=bigset(scrambleind);
            indsetappend=bigset(1:ll);
            indset=[indset indsetappend];
         end
         indset=sort(indset);
      end
      phi(measlist,1:length(indset))=indset;
   end
end
