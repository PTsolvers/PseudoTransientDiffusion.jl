clear

data_1D = load('../out_visu/ini_1D.mat');
H0_1D = data_1D.H0_1D;
xc_1D = data_1D.xc_1D;

data_2D = load('../out_visu/ini_2D.mat');
H0_2D = data_2D.H0_2D;
xc_2D = data_2D.xc_2D;
yc_2D = data_2D.yc_2D;

data_3D = load('../out_visu/ini_3D.mat');
H0_3D = data_3D.H0_3D;
dx = data_3D.dx_3D; xc_3D = dx+dx/2:dx:size(H0_3D,1)*dx-dx-dx/2;
dy = data_3D.dy_3D; yc_3D = dy+dy/2:dy:size(H0_3D,2)*dy-dy-dy/2;
dz = data_3D.dz_3D; zc_3D = dz+dz/2:dz:size(H0_3D,3)*dz-dz-dz/2;

FS = 20;

% figure(1),clf,set(gcf,'color','white')
figure(1),clf,set(gcf,'color','white','pos',[1400 500 1000 400])


sp1 = subplot(131);
plot(xc_1D, H0_1D, 'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
xlabel('\bf{lx}', 'fontsize',FS)
ylabel('\bf{H_{initial}}', 'fontsize',FS)
set(gca,'fontname','Courier')
text(0.5,0.9,'a)','fontsize',FS+2,'fontname','Courier')

sp2 = subplot(132); imagesc(xc_2D, yc_2D, H0_2D'), axis xy equal tight, set(gca, 'fontsize',FS, 'linewidth',1.3)
set(gca, 'XTick', [0.1 9.9], 'XTicklabel', [0 10], 'fontsize',FS)
set(gca, 'YTick', [0.1 9.9], 'YTicklabel', [0 10], 'fontsize',FS)
set(gca,'fontname','Courier')
xlabel('\bf{lx}', 'fontsize',FS)
ylabel('\bf{ly}', 'fontsize',FS)
text(0.5,9,'b)','fontsize',FS+2,'fontname','Courier','Color','w')

sp3 = subplot(133);
nx = size(H0_3D,1); ny = size(H0_3D,2); nz = size(H0_3D,3);
D = H0_3D;

s1 = fix( nx/2.0);
s2 = fix( ny/2.0);
s3 = fix( nz/3  );
s4 = fix( nz/2  );
s5 = fix( nz    );

hold on
slice(D( :  , :  ,1:s5),[],s2,s3),shading flat
slice(D( :  , :  ,1:s3),ny,[],[]),shading flat
slice(D(1:s2, :  ,1:s5),ny,[],[]),shading flat
slice(D( :  , :  ,1:s3),[],nx,[]),shading flat
slice(D( :  ,1:s1,1:s4),[],nx,[]),shading flat
slice(D(1:s2, :  , :  ),[],[],s5),shading flat
slice(D( :  ,1:s1, :  ),[],[],s4),shading flat
slice(D( :  , :  ,1:s4),s1,[],[]),shading flat
%%%
% is1 = isosurface(tp1, 0.02);
% is2 = isosurface(tp1,-0.02);
% his1 = patch(is1); set(his1,'CData',4000,'Facecolor','Flat','Edgecolor','none')
% his2 = patch(is2); set(his2,'CData',2600,'Facecolor','Flat','Edgecolor','none')
hold off
set(gca, 'linewidth',1.4)
set(gca, 'Ticklength', [0 0])
set(gca, 'XTick', [10 nx-5], 'XTicklabel', [10 0], 'fontsize',FS)
set(gca, 'YTick', [10 ny-5], 'YTicklabel', [10 0], 'fontsize',FS)
set(gca, 'ZTick', [10 nz-5], 'ZTicklabel', [0 10], 'fontsize',FS)

% xlabel('Lz = 3000','fontsize',FS,'interpreter','latex')

% title('a) \quad\quad [x, y, z] correlation length = [5, 5, 5]','fontsize',FS+2,'interpreter','latex')
% title('b) \quad\quad [x, y, z] correlation length = [3, 3, 20]','fontsize',FS+2,'interpreter','latex')
% title('c) \quad\quad [x, y, z] correlation length = [20, 20, 3]','fontsize',FS+2,'interpreter','latex')

text(150,315,0,'\bf{ly}','fontsize',FS,'fontname','Courier') %xlabel('lx','fontsize',FS)
text(350,160,0,'\bf{lx}','fontsize',FS,'fontname','Courier') %ylabel('ly','fontsize',FS)
zlabel('\bf{lz}','fontsize',FS)
text(350,160,340,'c)','fontsize',FS+2,'fontname','Courier')

cb = colorbar;
cb.Location = 'southoutside';
cb.Limits = [0 1];
cb.Ticks = [0.05 0.95];
cb.TickLabels = {'0' '1'};
posCB = get(cb,'position');
cb.Position = [posCB(1)*0.9 posCB(2)*0.6 posCB(3)*0.5 posCB(4)*1.1];
set(cb,'fontsize',FS)
set(gca,'fontname','Courier')
text(340,180,-100,'\bf{H_{initial}}','fontsize',FS,'fontname','Courier')

box on
axis image
view(145,22)
camlight
% light
camproj perspective
light('position',[0.6 -1 1]);
% light('position',[1 -1 1]);
light('position',[-1.5 0.5 -0.5], 'color', [.6 .2 .2]);

pos1 = get(sp1,'position'); set(sp1,'position',[pos1(1)*.6   pos1(2) pos1(3) pos1(4)])
pos2 = get(sp2,'position'); set(sp2,'position',[pos2(1)      pos2(2) pos2(3) pos2(4)])
pos3 = get(sp3,'position'); set(sp3,'position',[pos3(1)*1.05 pos3(2) pos3(3) pos3(4)])

fig = gcf;
fig.PaperPositionMode = 'auto';
% print('fig_ini','-dpng','-r300')
