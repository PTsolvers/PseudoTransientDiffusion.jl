clear

data_1D = load('../out_visu/ini_1D.mat');           H0_1D = data_1D.H0_1D;
data_1D = load('../out_visu/diff_1D_lin.mat');     lin_1D = data_1D.H_1D; xc_1D = data_1D.xc_1D - data_1D.xc_1D(1);
data_1D = load('../out_visu/diff_1D_linstep.mat'); linstep_1D = data_1D.H_1D;
data_1D = load('../out_visu/diff_1D_nonlin.mat');  nonlin_1D = data_1D.H_1D;

data_2D = load('../out_visu/diff_2D_lin.mat');     lin_2D = data_2D.H_2D; xc_2D = data_2D.xc_2D - data_2D.xc_2D(1); yc_2D = data_2D.yc_2D  - data_2D.yc_2D(1);
data_2D = load('../out_visu/diff_2D_linstep.mat'); linstep_2D = data_2D.H_2D;
data_2D = load('../out_visu/diff_2D_nonlin.mat');  nonlin_2D = data_2D.H_2D;

data_3D = load('../out_visu/diff_3D_lin.mat');     lin_3D = data_3D.H_3D;
data_3D = load('../out_visu/diff_3D_linstep.mat'); linstep_3D = data_3D.H_3D;
data_3D = load('../out_visu/diff_3D_nonlin.mat');  nonlin_3D = data_3D.H_3D;
% dx = data_3D.dx_3D; xc_3D = dx+dx/2:dx:size(H0_3D,1)*dx-dx-dx/2;
% dy = data_3D.dy_3D; yc_3D = dy+dy/2:dy:size(H0_3D,2)*dy-dy-dy/2;
% dz = data_3D.dz_3D; zc_3D = dz+dz/2:dz:size(H0_3D,3)*dz-dz-dz/2;

FS = 20;

figure(1),clf,set(gcf,'color','white','pos',[1400 10 1000 950])
% 1D
sp1 = subplot(331);
plot(xc_1D, H0_1D, xc_1D, lin_1D, 'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
ylabel({'\bf{linear}';'diffusion'}, 'fontsize',FS)
set(gca, 'XTick', [])
set(gca,'fontname','Courier')
text(0.5,0.9,'(a)','fontsize',FS+2,'fontname','Courier')

sp4 = subplot(334);
plot(xc_1D, H0_1D, xc_1D, linstep_1D, 'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
ylabel({'\bf{linear step}';'diffusion'}, 'fontsize',FS)
set(gca, 'XTick', [])
set(gca,'fontname','Courier')
text(0.5,0.9,'(d)','fontsize',FS+2,'fontname','Courier')

sp7 = subplot(337);
plot(xc_1D, H0_1D, xc_1D, nonlin_1D, 'linewidth',3), axis square, set(gca, 'fontsize',FS, 'linewidth',1.4)
xlabel('\bf{lx}', 'fontsize',FS)
ylabel({'\bf{nonlinear}';'diffusion'}, 'fontsize',FS)
set(gca,'fontname','Courier')
text(0.5,0.9,'(g)','fontsize',FS+2,'fontname','Courier')

% 2D
sp2 = subplot(332); imagesc(xc_2D, yc_2D, lin_2D'), axis xy equal tight, set(gca, 'fontsize',FS, 'linewidth',1.3)
set(gca, 'XTick', [])
set(gca, 'YTick', [0.1 9.9], 'YTicklabel', [0 10], 'fontsize',FS)
set(gca,'fontname','Courier')
ylabel('\bf{ly}', 'fontsize',FS)
caxis([0 1])
text(0.5,9,'(b)','fontsize',FS+2,'fontname','Courier','Color','w')

sp5 = subplot(335); imagesc(xc_2D, yc_2D, linstep_2D'), axis xy equal tight, set(gca, 'fontsize',FS, 'linewidth',1.3)
set(gca, 'XTick', [])
set(gca, 'YTick', [0.1 9.9], 'YTicklabel', [0 10], 'fontsize',FS)
set(gca,'fontname','Courier')
ylabel('\bf{ly}', 'fontsize',FS)
caxis([0 1])
text(0.5,9,'(e)','fontsize',FS+2,'fontname','Courier','Color','w')

sp8 = subplot(338); imagesc(xc_2D, yc_2D, nonlin_2D'), axis xy equal tight, set(gca, 'fontsize',FS, 'linewidth',1.3)
set(gca, 'XTick', [0.1 9.9], 'XTicklabel', [0 10], 'fontsize',FS)
set(gca, 'YTick', [0.1 9.9], 'YTicklabel', [0 10], 'fontsize',FS)
set(gca,'fontname','Courier')
xlabel('\bf{lx}', 'fontsize',FS), ylabel('\bf{ly}', 'fontsize',FS)
caxis([0 1])
text(0.5,9,'(h)','fontsize',FS+2,'fontname','Courier','Color','w')

% 3D
tp = {lin_3D, linstep_3D, nonlin_3D};
for ip=1:length(tp)
    eval(['sp' num2str(ip*3) '= subplot(3,3,' num2str(ip*3) ');'])

    D  = tp{ip};
    nx = size(D,1); ny = size(D,2); nz = size(D,3);
    
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
    
    hold off
    set(gca, 'linewidth',1.4)
    set(gca, 'Ticklength', [0 0])
    if ip==3
        set(gca, 'XTick', [10 nx-5], 'XTicklabel', [10 0], 'fontsize',FS)
        set(gca, 'YTick', [10 ny-5], 'YTicklabel', [10 0], 'fontsize',FS)
        text(150,315,0,'\bf{ly}','fontsize',FS,'fontname','Courier')
        text(350,160,0,'\bf{lx}','fontsize',FS,'fontname','Courier')
    else
        set(gca, 'XTick', [])
        set(gca, 'YTick', [])
    end
    set(gca, 'ZTick', [10 nz-5], 'ZTicklabel', [0 10], 'fontsize',FS)
    zlabel('\bf{lz}','fontsize',FS,'fontname','Courier')
    caxis([0 1])
    if ip==3
        cb = colorbar;
        cb.Location = 'southoutside';
        cb.Limits = [0 1];
        cb.Ticks = [0.05 0.95];
        cb.TickLabels = {'0' '1'};
        posCB = get(cb,'position');
        cb.Position = [posCB(1)*0.9 posCB(2)*0.3 posCB(3)*0.5 posCB(4)*1.2];
        set(cb,'fontsize',FS)
        text(340,180,-130,'\bf{H}','fontsize',FS+2,'fontname','Courier')
    end
    if ip==1
        text(350,160,350,'(c)','fontsize',FS+2,'fontname','Courier')
    elseif ip==2
        text(350,160,350,'(f)','fontsize',FS+2,'fontname','Courier')
    else
        text(350,160,350,'(i)','fontsize',FS+2,'fontname','Courier')
    end
    
    set(gca,'fontname','Courier')
    box on
    axis image
    view(145,22)
    camlight
    % light
    camproj perspective
    light('position',[0.6 -1 1]);
    % light('position',[1 -1 1]);
    light('position',[-1.5 0.5 -0.5], 'color', [.6 .2 .2]);
end


pos1 = get(sp1,'position'); set(sp1,'position',[pos1(1)*.8 pos1(2) pos1(3) pos1(4)])
pos4 = get(sp4,'position'); set(sp4,'position',[pos4(1)*.8 pos4(2) pos4(3) pos4(4)])
pos7 = get(sp7,'position'); set(sp7,'position',[pos7(1)*.8 pos7(2) pos7(3) pos7(4)])

pos2 = get(sp2,'position'); set(sp2,'position',[pos2(1) pos2(2) pos2(3) pos2(4)])
pos5 = get(sp5,'position'); set(sp5,'position',[pos5(1) pos5(2) pos5(3) pos5(4)])
pos8 = get(sp8,'position'); set(sp8,'position',[pos8(1) pos8(2) pos8(3) pos8(4)])

pos3 = get(sp3,'position'); set(sp3,'position',[pos3(1)*1.05 pos3(2) pos3(3) pos3(4)])
pos6 = get(sp6,'position'); set(sp6,'position',[pos6(1)*1.05 pos6(2) pos6(3) pos6(4)])
pos9 = get(sp9,'position'); set(sp9,'position',[pos9(1)*1.05 pos9(2) pos9(3) pos9(4)])

fig = gcf;
fig.PaperPositionMode = 'auto';
% print('fig_final','-dpng','-r300')
