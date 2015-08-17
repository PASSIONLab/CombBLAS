function hf = mybarplot1(time, repeat, cores, class)
dim = size(time);
time_avg = zeros(dim(1)/repeat,dim(2));
for i=1:repeat:dim(1)
    time_avg(ceil(i/repeat),:) = mean(time((i+1):(i+(repeat-1)),:),1);
end

threads = unique(time_avg(:,4));
nthreads = length(threads);
layers = unique(time_avg(:,3));
nlayers = length(layers);
time_thread = zeros(nthreads, nlayers);
for t = 1:nthreads
    time_thread(t,:) = time_avg(time_avg(:,4)==threads(t),12);
end

thr_text = {'t=1', 't=2', 't=4', 't=8', 't=16'};
lay_text = {'c=1', 'c=2', 'c=4', 'c=8', 'c=12', 'c=16'};


time_avg(:,7) = time_avg(:,7) + time_avg(:,8);
time_3D = zeros(nlayers, nthreads, 5);
for l = 1:nlayers
    time_3D(l,:,:) = time_avg(time_avg(:,3)==layers(l),[5,6,7,9,11]);
end

hf = plotBarStackGroups(time_3D, lay_text, class, cores);




%fig=gcf;
%set(findall(fig,'-property','FontSize'),'FontSize',14)


