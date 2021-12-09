clear

% load scaling_data
fid = fopen('../output/out_diff_2D_nonlin_perf_octo.txt' ,'r'); diff_2D_octo  = fscanf(fid, '%d %d %d %f %f %f %f', [7 Inf]); fclose(fid); % nx ny ittot t_toc A_eff t_it T_eff
fid = fopen('../output/out_diff_2D_nonlin_perf_volta.txt','r'); diff_2D_volta = fscanf(fid, '%d %d %d %f %f %f %f', [7 Inf]); fclose(fid); % nx ny ittot t_toc A_eff t_it T_eff
fid = fopen('../output/out_diff_2D_nonlin_perf_daint.txt','r'); diff_2D_daint = fscanf(fid, '%d %d %d %f %f %f %f', [7 Inf]); fclose(fid); % nx ny ittot t_toc A_eff t_it T_eff
fid = fopen('../output/out_diff_3D_nonlin_perf_octo.txt' ,'r'); diff_3D_octo  = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../output/out_diff_3D_nonlin_perf_volta.txt','r'); diff_3D_volta = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../output/out_diff_3D_nonlin_perf_daint.txt','r'); diff_3D_daint = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../../PseudoTransientStokes/output/out_Stokes3D_ve_perf_octo.txt' ,'r'); stokes_3D_octo  = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../../PseudoTransientStokes/output/out_Stokes3D_ve_perf_volta.txt','r'); stokes_3D_volta = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../../PseudoTransientStokes/output/out_Stokes3D_ve_perf_daint.txt','r'); stokes_3D_daint = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff

fid = fopen('../output/out_diff_3D_nonlin_multixpu_perf_octo.txt' ,'r'); diff_3D_mxpu_octo  = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../output/out_diff_3D_nonlin_multixpu_perf_volta.txt','r'); diff_3D_mxpu_volta = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../output/out_diff_3D_nonlin_multixpu_perf_daint.txt','r'); diff_3D_mxpu_daint = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../../PseudoTransientStokes/output/out_Stokes3D_ve_xpu_perf_octo.txt' ,'r'); stokes_3D_mxpu_octo  = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../../PseudoTransientStokes/output/out_Stokes3D_ve_xpu_perf_volta.txt','r'); stokes_3D_mxpu_volta = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff
fid = fopen('../../PseudoTransientStokes/output/out_Stokes3D_ve_xpu_perf_daint.txt','r'); stokes_3D_mxpu_daint = fscanf(fid, '%d %d %d %d %d %f %f %f %f', [9 Inf]); fclose(fid); % np nx ny nz ittot t_toc A_eff t_it T_eff

nrep  = 5; % number of repetitions of the experiment
nrep2 = 2; % number of repetitions of the experiment
nrep1 = 1; % number of repetitions of the experiment
my_type = "my_max";
% my_type = "my_mean";

diff_2D_octo_2  = average_exp(diff_2D_octo, nrep, my_type);
diff_2D_volta_2 = average_exp(diff_2D_volta, nrep, my_type);
diff_2D_daint_2 = average_exp(diff_2D_daint, nrep2, my_type);
diff_3D_octo_2  = average_exp(diff_3D_octo, nrep, my_type);
diff_3D_volta_2 = average_exp(diff_3D_volta, nrep, my_type);
diff_3D_daint_2 = average_exp(diff_3D_daint, nrep2, my_type);
diff_3D_mxpu_octo_2  = average_exp(diff_3D_mxpu_octo, nrep, my_type);
diff_3D_mxpu_volta_2 = average_exp(diff_3D_mxpu_volta, nrep, my_type);
diff_3D_mxpu_daint_2 = average_exp(diff_3D_mxpu_daint, nrep1, my_type);

% stokes_3D_octo_2  = average_exp(stokes_3D_octo, nrep, my_type);
% stokes_3D_volta_2 = average_exp(stokes_3D_volta, nrep, my_type);
% stokes_3D_daint_2 = average_exp(stokes_3D_daint, nrep1, my_type);
stokes_3D_mxpu_octo_2  = average_exp(stokes_3D_mxpu_octo, nrep, my_type);
stokes_3D_mxpu_volta_2 = average_exp(stokes_3D_mxpu_volta, nrep, my_type);
stokes_3D_mxpu_daint_2 = average_exp(stokes_3D_mxpu_daint, nrep1, my_type);

T_peak_volta = 840;
T_peak_octo  = 254;
T_peak_daint = 561;

% no hide_comm perfs
single_daint_diff   = 315.6;
single_daint_stokes = 210.2;
single_volta_diff   = 481.1;
single_volta_stokes = 334.1;
sc = 100; % to get percent

FS = 20;
mylim = [0 870];
mylimx = [32 550];
ylab = 780;

mylim2  = [0.959 1.003].*sc;
mylimx2 = [0.8 3.e3];
ylab2 = 0.963*sc;

fig1 = 1;
fig2 = 1;

%%
if fig1==1
figure(1),clf,set(gcf,'color','white','pos',[1400 10 800 400])
sp1 = subplot(121);
%diff_2D_octo_2(1,:),diff_2D_octo_2(end,:), '-o', ...
%diff_2D_octo_2(1,:),T_peak_octo*ones(size(diff_2D_octo_2(1,:))), 'k--',...
semilogx(diff_2D_volta_2(1,:),diff_2D_volta_2(end,:), '-o', ...
         diff_2D_daint_2(1,:),diff_2D_daint_2(end,:), '-o', ...
         'linewidth',3, 'MarkerFaceColor','k'), set(gca, 'fontsize',FS, 'linewidth',1.4)
hold on
semilogx(diff_2D_octo_2(1,:),T_peak_volta*ones(size(diff_2D_octo_2(1,:))),'k-.', ...
         diff_2D_octo_2(1,:),T_peak_daint*ones(size(diff_2D_octo_2(1,:))),'k--', ...
         'linewidth',1.5, 'MarkerFaceColor','k')
hold off
title({'2D nonlinear'; 'diffusion'},'fontsize',FS-2)
ylabel({' ';'\bf{T_{eff} [GB/s]}'}, 'fontsize',FS)
ylim(mylim)
set(gca, 'XTick',diff_2D_octo_2(1,:))
xtickangle(45)
set(gca,'fontname','Courier')
xlabel('\bf{nx}', 'fontsize',FS)
text(35,ylab,'(a)','fontsize',FS+2,'fontname','Courier')


sp2 = subplot(122);
%diff_3D_octo_2(2,:),diff_3D_octo_2(end,:), '-o', ...
%diff_3D_volta_2(2,:),T_peak_octo*ones(size(diff_3D_volta_2(2,:))), 'k--',...
semilogx(diff_3D_volta_2(2,:),diff_3D_volta_2(end,:), '-o', ...
         diff_3D_daint_2(2,:),diff_3D_daint_2(end,:), '-o', ...
         'linewidth',3, 'MarkerFaceColor','k'), set(gca, 'fontsize',FS, 'linewidth',1.4)
hold on
semilogx(diff_3D_volta_2(2,:),T_peak_volta*ones(size(diff_3D_volta_2(2,:))),'k-.', ...
         diff_3D_volta_2(2,:),T_peak_daint*ones(size(diff_3D_volta_2(2,:))),'k--', ...
         'linewidth',1.5, 'MarkerFaceColor','k')
hold off
title({'3D nonlinear'; 'diffusion'},'fontsize',FS-2)
lg=legend('Tesla V100 SXM2', 'Tesla P100 PCIe'); set(lg,'box','off')
ylim(mylim)
xlim(mylimx)
set(gca, 'XTick',diff_3D_volta_2(2,:), 'YTicklabel',[])
xtickangle(45)
set(gca,'fontname','Courier')
xlabel('\bf{nx}', 'fontsize',FS)
text(33,ylab,'(b)','fontsize',FS+2,'fontname','Courier')

pos1 = get(sp1,'position'); set(sp1,'position',[pos1(1)*0.97  pos1(2)*2 pos1(3)*1.12 pos1(4)*0.8])
pos2 = get(sp2,'position'); set(sp2,'position',[pos2(1)*0.97  pos2(2)*2 pos2(3)*1.12 pos2(4)*0.8])
fig = gcf;
fig.PaperPositionMode = 'auto';
% print('fig_perf23D_diff_2','-dpng','-r300')
end
%%
if fig2==1

figure(2),clf,set(gcf,'color','white','pos',[1400 10 800 400])
sp1 = subplot(121);
% diff_3D_mxpu_octo_2(1,:),diff_3D_mxpu_octo_2(end,:)./diff_3D_mxpu_octo_2(end,1), '-o', ...
semilogx(diff_3D_mxpu_volta_2(1,:),diff_3D_mxpu_volta_2(end,:)./single_volta_diff.*sc, '-o', ...
         diff_3D_mxpu_daint_2(1,:),diff_3D_mxpu_daint_2(end,:)./single_daint_diff.*sc, '-o', ...
        'linewidth',3, 'MarkerFaceColor','k'), set(gca, 'fontsize',FS, 'linewidth',1.4)
title({'3D nonlinear'; 'diffusion'},'fontsize',FS-2)
ylabel({' ';'\bf{E}'}, 'fontsize',FS)
ytickformat('%g\%')
% lg=legend('Titan Xm', 'Tesla V100 SXM2'); set(lg,'box','off')
ylim(mylim2)
xlim(mylimx2)
set(gca, 'XTick',diff_3D_mxpu_daint_2(1,:))
xtickangle(45)
set(gca,'fontname','Courier')
xlabel('\bf{P (GPUs)}', 'fontsize',FS)
text(1.1,ylab2,'(a)','fontsize',FS+2,'fontname','Courier')

sp2 = subplot(122);
% stokes_3D_mxpu_octo_2(1,:),stokes_3D_mxpu_octo_2(end,:)./stokes_3D_mxpu_octo_2(end,1), '-o', ...
semilogx(stokes_3D_mxpu_volta_2(1,:),stokes_3D_mxpu_volta_2(end,:)./single_volta_stokes.*sc, '-o', ...
         stokes_3D_mxpu_daint_2(1,:),stokes_3D_mxpu_daint_2(end,:)./single_daint_stokes.*sc, '-o', ...
     'linewidth',3, 'MarkerFaceColor','k'),set(gca, 'fontsize',FS, 'linewidth',1.4)
title({'3D visco-elastic'; 'Stokes'},'fontsize',FS-2)
% ylabel({' ';'\bf{E}'}, 'fontsize',FS)
lg=legend('Tesla V100 SXM2', 'Tesla P100 PCIe'); set(lg,'box','off')
ylim(mylim2)
xlim(mylimx2)
set(gca, 'XTick',stokes_3D_mxpu_daint_2(1,:), 'YTicklabel',[])
xtickangle(45)
set(gca,'fontname','Courier')
xlabel('\bf{P (GPUs)}', 'fontsize',FS)
text(1.1,ylab2,'(b)','fontsize',FS+2,'fontname','Courier')

pos1 = get(sp1,'position'); set(sp1,'position',[pos1(1)*1.04  pos1(2)*2 pos1(3)*1.1 pos1(4)*0.8])
pos2 = get(sp2,'position'); set(sp2,'position',[pos2(1)*0.96  pos2(2)*2 pos2(3)*1.1 pos2(4)*0.8])

fig = gcf;
fig.PaperPositionMode = 'auto';
% print('fig_parperf3D_all','-dpng','-r300')

end

%%% support function
function B = average_exp(A, nrep, type)

nexp = size(A,2)/nrep;
B    = zeros(size(A,1),nexp);

if type == "my_mean"
    for i=1:nexp
        B(:,i) = mean(A(:,(i-1)*nrep+1:i*nrep),2);
    end
elseif type == "my_max"
    for i=1:nexp
        B(:,i) = max(A(:,(i-1)*nrep+1:i*nrep),[],2);
    end
end
end
