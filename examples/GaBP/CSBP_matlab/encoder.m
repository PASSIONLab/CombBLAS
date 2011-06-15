function measvec=encoder(phiind, phisign, x)

[maxphirow, tmp]=size(phiind);
measvec=zeros(maxphirow,1);
for measlist=1:maxphirow
  indset=phiind(measlist, :);
  signset=phisign(measlist,:);
  indset=setdiff(indset,0);
  signset=signset(1:length(indset));
  measvec(measlist)=sum(x(indset).*signset);
end


