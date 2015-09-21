function hf = mybarplot1(time, cores, class)
dim = size(time);
for i=1:dim(1)
    actual_cores = time(i,1) * time(i,2) * time(i,3) * time(i,4);
    time(i,5:12) = time(i,5:12) * (cores/actual_cores);
end
time_avg = time;
%threads = unique(time_avg(:,4));
%nthreads = length(threads);
%layers = unique(time_avg(:,3));
%nlayers = length(layers);

thr_text = {'t=1', 't=3', 't=6', 't=12'};
lay_text = {'c=1', 'c=2', 'c=4', 'c=8', 'c=16'};
threads = [1 3 6 12];
nthreads = length(threads);
layers = [1 2 4 8 16];
nlayers = length(layers);


%time_avg(:,7) = time_avg(:,7) + time_avg(:,8);
time_3D = zeros(nlayers, nthreads, 6);
for l = 1:nlayers
    for t = 1:nthreads
        if(isempty(time_avg(time_avg(:,3)==layers(l) & time_avg(:,4)==threads(t),:))~=1)
            time_3D(l,t,:) = time_avg(time_avg(:,3)==layers(l) & time_avg(:,4)==threads(t),[5,6,7,8, 9,11]);
        end
    end
end

hf = plotBarStackGroups(time_3D, lay_text, thr_text, class, cores);




%fig=gcf;
%set(findall(fig,'-property','FontSize'),'FontSize',14)


