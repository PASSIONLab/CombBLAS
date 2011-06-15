function sol=setdiff_shri(vec, elem)

ind=find(vec~=elem);
sol=vec(ind); 
