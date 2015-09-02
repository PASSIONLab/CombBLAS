function strongScaleRop(folder, startrow)

files = dir(sprintf('%s/*.txt',folder));
all_data = zeros(0,15);

for i=1:length(files)
    fname = sprintf('%s/%s',folder, files(i).name);
    disp(fname);
    data = dlmread(fname);
    data = horzcat(zeros(size(data,1),2), data);
    %nrow = size(data(:,1));
    %data1= vertcat(data1, data([1:3:nrow],:));
    %data2= vertcat(data2, data([2:3:nrow],:));
    all_data= vertcat(all_data, data);
end

all_data(:,2) = all_data(:,3) .* all_data(:,4) .* all_data(:,5) .* all_data(:,6);


L1T1 = all_data((all_data(:,5)==1 & all_data(:,6)==1),[2, 14]);
[val, order] = sort(L1T1(:,1));
L1T1 = log2(L1T1(order,:));
L1T8 = all_data((all_data(:,5)==1 & all_data(:,6)==6),[2, 14]);
[val, order] = sort(L1T8(:,1));
L1T8 = log2(L1T8(order,:));
L8T1 = all_data((all_data(:,5)==8 & all_data(:,6)==1),[2, 14]);
[val, order] = sort(L8T1(:,1));
L8T1 = log2(L8T1(order,:));
L8T8 = all_data((all_data(:,5)==8 & all_data(:,6)==6),[2, 14]);
[val, order] = sort(L8T8(:,1));
L8T8 = log2(L8T8(order,:));
L16T8 = all_data((all_data(:,5)==16 & all_data(:,6)==6),[2, 14]);
[val, order] = sort(L16T8(:,1));
L16T8 = log2(L16T8(order,:));

nrows = size(L1T1,1);
selrows = [startrow:3:nrows];
plot1 = plot(L1T1(selrows,1), L1T1(selrows,2),L1T8(selrows,1), L1T8(selrows,2),L8T1(selrows,1), L8T1(selrows,2),...
    L8T8(selrows,1), L8T8(selrows,2), L16T8(selrows,1), L16T8(selrows,2),...
    'LineWidth',2, 'MarkerFaceColor',[1 1 1],'MarkerSize',8);

%plot1 = plot(log2(mat(:,1)), log2(mat(:,2:6)),'LineWidth',2, 'MarkerFaceColor',[1 1 1],'MarkerSize',8)

set(gca,'XLim',[9 15.5])
xt = get(gca, 'XTick');
set(gca, 'XTick', min(xt):1:max(xt));
xt = get(gca, 'XTick');
set (gca, 'XTickLabel', 2.^xt);

%xt = get(gca, 'XTick');
yt = get(gca, 'YTick');
set(gca, 'YTick', min(yt):1:max(yt));
yt = get(gca, 'YTick');
%set(gca,'YLim',[min(yt) max(yt)])
set (gca, 'YTickLabel', 2.^yt);
%set (gca, 'YTickLabel', 2.^[min(yt):1:max(yt)]);

set(gca, 'YGrid','on','XGrid','on', 'FontSize',14, 'LineWidth',2);
xlabel('Number of Cores'); ylabel('Time (sec)'); 
box(gca,'on');
set(plot1(1),'Marker','o','DisplayName','2D (t=1)');
set(plot1(2),'Marker','^','DisplayName','2D (t=6)');
set(plot1(3),'MarkerSize',10,'Marker','x','DisplayName','3D (c=8, t=1)');
set(plot1(4),'MarkerSize',10,'Marker','s','DisplayName','3D (c=8, t=6)');
set(plot1(5),'MarkerSize',8,'Marker','*','DisplayName','3D (c=16, t=6)');
legend('show','Location','SouthWest')

if(startrow==1)
    aux_title = 'AA';
elseif(startrow==2)
     aux_title = 'R''A';
else
    aux_title = '(R''A)R';
end
    
%title(sprintf('%s (%s)', folder , aux_title));
title(sprintf('%s x %s (on Edison)', folder , folder));




