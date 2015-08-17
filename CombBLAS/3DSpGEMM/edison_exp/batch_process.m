
%tl = sprintf('RMAT-24 matrix (approx. %d cores)', cores);
folder = 'G500/';
files = dir(sprintf('%s*.txt',folder));
for i=1:length(files)
    fname = sprintf('%s%s',folder, files(i).name);
    
    disp(fname);
    data = dlmread(fname);
    splits1 = strsplit(fname, '.');
    splits = strsplit(splits1{1}, '\_');
    
    title = sprintf('%s-%s', splits{2}, splits{3});
    cores = str2num(splits{4});
    disp (title);
    hf = mybarplot1(data,5,cores, title);
    
    oname = sprintf('%splots/%s-%d.pdf',folder, title, cores);
    saveas(hf,oname,'pdf')
end



