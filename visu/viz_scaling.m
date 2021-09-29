clear

% load scaling_data
fid = fopen('../output/out_diff_1D_lin3.txt'         ,'r'); diff_1D_lin3 = fscanf(fid, '%d %d %d',       [3 Inf]); fclose(fid);
fid = fopen('../output/out_diff_2D_lin3.txt'         ,'r'); diff_2D_lin3 = fscanf(fid, '%d %d %d %d',    [4 Inf]); fclose(fid);
fid = fopen('../output/out_diff_3D_lin3_multixpu.txt','r'); diff_3D_lin3 = fscanf(fid, '%d %d %d %d %d', [5 Inf]); fclose(fid);

fid = fopen('../output/out_diff_1D_linstep3.txt'         ,'r'); diff_1D_linstep3 = fscanf(fid, '%d %d %d',       [3 Inf]); fclose(fid);
fid = fopen('../output/out_diff_2D_linstep3.txt'         ,'r'); diff_2D_linstep3 = fscanf(fid, '%d %d %d %d',    [4 Inf]); fclose(fid);
fid = fopen('../output/out_diff_3D_linstep3_multixpu.txt','r'); diff_3D_linstep3 = fscanf(fid, '%d %d %d %d %d', [5 Inf]); fclose(fid);

fid = fopen('../output/out_diff_1D_nonlin3.txt'         ,'r'); diff_1D_nonlin3 = fscanf(fid, '%d %d %d',       [3 Inf]); fclose(fid);
fid = fopen('../output/out_diff_2D_nonlin3.txt'         ,'r'); diff_2D_nonlin3 = fscanf(fid, '%d %d %d %d',    [4 Inf]); fclose(fid);
fid = fopen('../output/out_diff_3D_nonlin3_multixpu.txt','r'); diff_3D_nonlin3 = fscanf(fid, '%d %d %d %d %d', [5 Inf]); fclose(fid);

FS = 20;
mylim = [0.02 1.04];
ylab = 0.14;

fig1 = 1;
fig2 = 0;

%%
if fig1==1
figure(2),clf,set(gcf,'color','white','pos',[1400 10 1000 400])
% 1D
sp1 = subplot(131);
st = 2;
semilogx(diff_1D_lin3(1,st:end), diff_1D_lin3(2,st:end)./diff_1D_lin3(1,st:end)./diff_1D_lin3(3,st:end),'-o', ...
         diff_2D_lin3(1,st:end), diff_2D_lin3(3,st:end)./diff_2D_lin3(1,st:end)./diff_2D_lin3(4,st:end),'-o', ...
         diff_3D_lin3(1,1:end),  diff_3D_lin3(4,1:end)./ diff_3D_lin3(1,1:end)./ diff_3D_lin3(5,1:end),'-o', ...
         'linewidth',3, 'MarkerFaceColor','k'), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
ylabel({' ';'\bf{iter_{tot}/nt/nx}'}, 'fontsize',FS)
ylim(mylim)
% lg=legend('1D', '2D', '3D'); set(lg,'box','off')
set(gca, 'XTick',diff_1D_lin3(1,st:end))
xtickangle(45)
xlabel('\bf{nx}', 'fontsize',FS)
title({'linear'; 'diffusion'})
set(gca,'fontname','Courier')
text(85,ylab,'a)','fontsize',FS+2,'fontname','Courier')

sp2 = subplot(132);
st = 2;
semilogx(diff_1D_linstep3(1,st:end), diff_1D_linstep3(2,st:end)./diff_1D_linstep3(1,st:end)./diff_1D_linstep3(3,st:end),'-o', ...
         diff_2D_linstep3(1,st:end), diff_2D_linstep3(3,st:end)./diff_2D_linstep3(1,st:end)./diff_2D_linstep3(4,st:end),'-o', ...
         diff_3D_linstep3(1,1:end),  diff_3D_linstep3(4,1:end)./diff_3D_linstep3(1,1:end)./diff_3D_linstep3(5,1:end),'-o', ...
         'linewidth',3, 'MarkerFaceColor','k'), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
title({'linear step'; 'diffusion'})
ylim(mylim)
set(gca, 'XTick',diff_1D_linstep3(1,st:end),'YTicklabel',[])
xtickangle(45)
xlabel('\bf{nx}', 'fontsize',FS)
set(gca,'fontname','Courier')
text(85,ylab,'b)','fontsize',FS+2,'fontname','Courier')

sp3 = subplot(133);
st = 2;
semilogx(diff_1D_nonlin3(1,st:end), diff_1D_nonlin3(2,st:end)./diff_1D_nonlin3(1,st:end)./diff_1D_nonlin3(3,st:end),'-o', ...
         diff_2D_nonlin3(1,st:end), diff_2D_nonlin3(3,st:end)./diff_2D_nonlin3(1,st:end)./diff_2D_nonlin3(4,st:end),'-o', ...
         diff_3D_nonlin3(1,1:end), diff_3D_nonlin3(4,1:end)./diff_3D_nonlin3(1,1:end)./diff_3D_nonlin3(5,1:end),'-o', ...
         'linewidth',3, 'MarkerFaceColor','k'), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
lg=legend('1D', '2D', '3D'); set(lg,'box','off')
title({'nonlinear'; 'diffusion'})
ylim(mylim)
set(gca, 'XTick',diff_1D_nonlin3(1,st:end),'YTicklabel',[])
xtickangle(45)
xlabel('\bf{nx}', 'fontsize',FS)
set(gca,'fontname','Courier')
text(85,ylab,'c)','fontsize',FS+2,'fontname','Courier')

pos1 = get(sp1,'position'); set(sp1,'position',[pos1(1)*0.9   pos1(2) pos1(3)*1.1 pos1(4)*1.1])
pos2 = get(sp2,'position'); set(sp2,'position',[pos2(1)*0.97  pos2(2) pos2(3)*1.1 pos2(4)*1.1])
pos3 = get(sp3,'position'); set(sp3,'position',[pos3(1)*0.985 pos3(2) pos3(3)*1.1 pos3(4)*1.1])
fig = gcf;
fig.PaperPositionMode = 'auto';
% print('fig_scale123D','-dpng','-r300')
end
%%
if fig2==1
figure(1),clf,set(gcf,'color','white','pos',[1400 10 1000 950])
% 1D
sp1 = subplot(331);
st = 2;
semilogx(diff_1D_lin(1,st:end), diff_1D_lin(2,st:end)./diff_1D_lin(1,st:end),'-o', ...
         diff_1D_lin3(1,st:end), diff_1D_lin3(2,st:end)./diff_1D_lin3(1,st:end),'-o', ...
         'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
ylabel({'\bf{linear}';'diffusion';' ';'iter_{tot}/nx'}, 'fontsize',FS)
set(gca, 'XTicklabel', [])
title('1D')
ylim([2 8])
set(gca,'fontname','Courier')

sp4 = subplot(334);
st = 2;
semilogx(diff_1D_linstep(1,st:end), diff_1D_linstep(2,st:end)./diff_1D_linstep(1,st:end),'-o', ...
         diff_1D_linstep3(1,st:end), diff_1D_linstep3(2,st:end)./diff_1D_linstep3(1,st:end),'-o', ...
         'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
ylabel({'\bf{linear step}';'diffusion';' ';'iter_{tot}/nx'}, 'fontsize',FS)
set(gca, 'XTicklabel', [])
ylim([2 8])
set(gca,'fontname','Courier')

sp7 = subplot(337);
st = 2;
semilogx(diff_1D_nonlin(1,st:end), diff_1D_nonlin(2,st:end)./diff_1D_nonlin(1,st:end),'-o', ...
         diff_1D_nonlin3(1,st:end), diff_1D_nonlin3(2,st:end)./diff_1D_nonlin3(1,st:end),'-o', ...
         'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
set(gca, 'XTick',diff_1D_nonlin(1,st:end))
xtickangle(45)
ylim([0.5 5])
xlabel('\bf{nx}', 'fontsize',FS)
ylabel({'\bf{nonlinear}';'diffusion';'';'iter_{tot}/nx'}, 'fontsize',FS)
set(gca,'fontname','Courier')

% 2D
sp2 = subplot(332);
st = 2;
semilogx(diff_2D_lin(1,st:end), diff_2D_lin(3,st:end)./diff_2D_lin(1,st:end),'-o', ...
         diff_2D_lin3(1,st:end), diff_2D_lin3(3,st:end)./diff_2D_lin3(1,st:end),'-o', ...
         'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
xlim([diff_2D_nonlin(1,st), diff_2D_nonlin(1,end)])
title('2D')
set(gca, 'XTicklabel', [],'YTicklabel', [])
ylim([2 8])
set(gca,'fontname','Courier')

sp5 = subplot(335);
st = 2;
semilogx(diff_2D_linstep(1,st:end), diff_2D_linstep(3,st:end)./diff_2D_linstep(1,st:end),'-o', ...
         diff_2D_linstep3(1,st:end), diff_2D_linstep3(3,st:end)./diff_2D_linstep3(1,st:end),'-o', ...
         'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
xlim([diff_2D_nonlin(1,st), diff_2D_nonlin(1,end)])
set(gca,'XTicklabel', [],'YTicklabel', [])
ylim([2 8])
set(gca,'fontname','Courier')

sp8 = subplot(338);
st = 2;
semilogx(diff_2D_nonlin(1,st:end), diff_2D_nonlin(3,st:end)./diff_2D_nonlin(1,st:end),'-o', ...
         diff_2D_nonlin3(1,st:end), diff_2D_nonlin3(3,st:end)./diff_2D_nonlin3(1,st:end),'-o', ...
         'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
xlim([diff_2D_nonlin(1,st), diff_2D_nonlin(1,end)])
set(gca, 'XTick',diff_2D_nonlin(1,st:end),'YTicklabel',[])
xtickangle(45)
ylim([0.5 5])
xlabel('\bf{nx}', 'fontsize',FS)
set(gca,'fontname','Courier')

% 3D
sp3 = subplot(333);
st = 1;
semilogx(diff_3D_lin(1,st:end),  diff_3D_lin(4,st:end)./diff_3D_lin(1,st:end),'-o', ...
         diff_3D_lin3(1,st:end), diff_3D_lin3(4,st:end)./diff_3D_lin3(1,st:end),'-o', ...
         'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
set(gca, 'XTicklabel', [],'YTicklabel', [])
title('3D')
ylim([2 8])
lg = legend('accel. 1','accel. 2'); set(lg,'box','off');
set(gca,'fontname','Courier')

sp6 = subplot(336);
st = 1;
semilogx(diff_3D_linstep(1,st:end),  diff_3D_linstep(4,st:end)./diff_3D_linstep(1,st:end),'-o', ...
         diff_3D_linstep3(1,st:end), diff_3D_linstep3(4,st:end)./diff_3D_linstep3(1,st:end),'-o', ...
         'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
set(gca, 'XTicklabel', [],'YTicklabel', [])
ylim([2 8])
set(gca,'fontname','Courier')

sp9 = subplot(339);
st = 1;
semilogx(diff_3D_nonlin(1,st:end),  diff_3D_nonlin(4,st:end)./diff_3D_nonlin(1,st:end),'-o', ...
         diff_3D_nonlin3(1,st:end), diff_3D_nonlin3(4,st:end)./diff_3D_nonlin3(1,st:end),'-o', ...
         'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
set(gca, 'XTick', diff_3D_nonlin(1,st:end),'YTicklabel', [])
xtickangle(45)
ylim([0.5 5])
xlabel('\bf{nx}', 'fontsize',FS)
set(gca,'fontname','Courier')

pos1 = get(sp1,'position'); set(sp1,'position',[pos1(1)*1.2 pos1(2)     pos1(3) pos1(4)])
pos4 = get(sp4,'position'); set(sp4,'position',[pos4(1)*1.2 pos4(2)*1.1 pos4(3) pos4(4)])
pos7 = get(sp7,'position'); set(sp7,'position',[pos7(1)*1.2 pos7(2)*1.7 pos7(3) pos7(4)])

pos2 = get(sp2,'position'); set(sp2,'position',[pos2(1) pos2(2)     pos2(3) pos2(4)])
pos5 = get(sp5,'position'); set(sp5,'position',[pos5(1) pos5(2)*1.1 pos5(3) pos5(4)])
pos8 = get(sp8,'position'); set(sp8,'position',[pos8(1) pos8(2)*1.7 pos8(3) pos8(4)])

pos3 = get(sp3,'position'); set(sp3,'position',[pos3(1)*0.96 pos3(2)     pos3(3) pos3(4)])
pos6 = get(sp6,'position'); set(sp6,'position',[pos6(1)*0.96 pos6(2)*1.1 pos6(3) pos6(4)])
pos9 = get(sp9,'position'); set(sp9,'position',[pos9(1)*0.96 pos9(2)*1.7 pos9(3) pos9(4)])

fig = gcf;
fig.PaperPositionMode = 'auto';
% print('fig_scale','-dpng','-r300')
end
