function [self_indexN, self_indexM]=GetSelfIndices(phi, aux);

[m,l]=size(phi);
[r,n]=size(aux);
self_indexN=[];
self_indexM=[];

for i=1:n
   neighbors=aux(:,i)';
   %neighbors=setdiff(neighbors,0);
   neighbors=setdiff_shri(neighbors,0);
   ln=length(neighbors);
   self_index=[];
   for j=1:ln 
      sindex=find(phi(neighbors(j), :)==i);
      self_index=[self_index sindex];
   end
   while (ln<r)
      self_index=[self_index 0];
      ln=ln+1;
   end
   self_indexN=[self_indexN; self_index];
end

for i=1:m
   neighbors=phi(i,:);
   neighbors=setdiff_shri(neighbors,0);
   ln=length(neighbors);
   self_index=[];
   for j=1:ln 
      sindex=find(aux(:, neighbors(j))==i);
      self_index=[self_index sindex];
   end
   while (ln<l)
      self_index=[self_index 0];
      ln=ln+1;
   end
   self_indexM=[self_indexM; self_index];
end