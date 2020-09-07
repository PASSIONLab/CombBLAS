function [hf] = plotBarStackGroups(stackData, groupLabels, barLabels, class, cores)
%% Plot a set of stacked bars, but group them according to labels provided.
%%
%% Params: 
%%      stackData is a 3D matrix (i.e., stackData(i, j, k) => (Group, Stack, StackElement)) 
%%      groupLabels is a CELL type (i.e., { 'a', 1 , 20, 'because' };)
%%
%% Copyright 2011 Evan Bollig (bollig at scs DOT fsu ANOTHERDOT edu
%%
%% 
NumGroupsPerAxis = size(stackData, 1);
NumStacksPerGroup = size(stackData, 2);


% Count off the number of bins
groupBins = 1:NumGroupsPerAxis;
MaxGroupWidth = 0.65; % Fraction of 1. If 1, then we have all bars in groups touching
groupOffset = MaxGroupWidth/NumStacksPerGroup;
hf = figure('visible','off');
    hold on; 
for i=1:NumStacksPerGroup

    Y = squeeze(stackData(:,i,:));
    
    % Center the bars:
    
    internalPosCount = i - ((NumStacksPerGroup+1) / 2);
    
    % Offset the group draw positions:
    groupDrawPos = (internalPosCount)* groupOffset + groupBins;
    
    h(i,:) = bar(Y, 'stacked');
    set(h(i,1),'facecolor',[0/255,50/255,255/255],'edgecolor','k');
    set(h(i,2),'facecolor',[135/255,206/255,255/255],'edgecolor','k');
    set(h(i,3),'facecolor',[0/255,128/255,0/255],'edgecolor','k');
    set(h(i,4),'facecolor',[255/255,210/255,0/255],'edgecolor','k');
    set(h(i,5),'facecolor',[200/255,0/255,0/255],'edgecolor','k');
    
    set(h(i,:),'BarWidth',groupOffset*1);
    set(h(i,:),'XData',groupDrawPos);
    rsum = sum(Y,2);
    
    h1 = text(groupDrawPos,rsum+.2, barLabels(i));
    set(h1, 'rotation', 90);
end
legend_txt = {'Broadcast','Scatter','Local Multiply','Merge Layer','Merge Fiber','Other'};
legend(legend_txt, 'Location', 'Northeast');
ylabel('Time (sec)');
tl = sprintf('%s (on %d cores of Edison)', class, cores);
title(tl);
hold off;
set(gca,'XTickMode','manual');
set(gca,'XTick',1:NumGroupsPerAxis);
set(gca,'XTickLabelMode','manual');
set(gca,'XTickLabel',groupLabels);

end 
