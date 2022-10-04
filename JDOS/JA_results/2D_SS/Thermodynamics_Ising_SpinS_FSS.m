clear
close all
%
% v1 - works, should be improved in <E> and <C> calculations
% v2 - correcting <E> and <C> calculations | works, should still be improved
% v3 - correcting for low T calculations using logs
% v4 - E and C calcs with log Z and log ZM
% v5 - cleanup, use only log data
% v6 - avg entropy and avg M from avg F 
% v7 - checking C from minF
% v8 - cleanup input, dS calcs, plots, only log_JDOS for calcs
% v9 - auto check for half or full JDOS, full JDOS calcs, plots
% v10 - calcs and plots for single H or T value
% v11 - including G_smooth using lagr
% v12 - includes core count
%
% LATTICE
%
L = 16;
Npos = 2;
%
N_atm = L^2;
NN = 4;
%
% FSS PARAMETERS
%
REP = 1E4;
skip = 1E2; % 1E2;
n_cores = 16;
%
J = 1;
T(:,1) = 0.05 : 0.05 : 5; % must be ascending!
H(:,1) = 0 : 0.002: 0.1; % must be ascending!
%
% END OF USER INPUT
%
if n_cores == 1
    if skip > 0
        JDOS_filename = ['JDOS_L', int2str(L),'_Nproj_', int2str(Npos),'_R1E', int2str(round(log10(REP))),'_skip_1E', int2str(round(log10(skip))),'.dat'];
    else
        JDOS_filename = ['JDOS_L', int2str(L),'_Nproj_', int2str(Npos),'_R1E', int2str(round(log10(REP))),'_skip_0.dat'];
    end
else
    %
    if skip > 0
        JDOS_filename = ['JDOS_L', int2str(L),'_Nproj_', int2str(Npos),'_R1E', int2str(round(log10(REP))),'_skip_1E', int2str(round(log10(skip))),'_x', int2str(n_cores),'.dat'];
    else
        JDOS_filename = ['JDOS_L', int2str(L),'_Nproj_', int2str(Npos),'_R1E', int2str(round(log10(REP))),'_x', int2str(n_cores),'.dat'];
    end
end
%
E_list_filename = ['E_list_L', int2str(L),'_Nproj_', int2str(Npos),'.dat'];
M_list_filename = ['M_list_L', int2str(L),'_Nproj_', int2str(Npos),'.dat'];
%
JDOS_aprox = load(JDOS_filename);
E_list = load(E_list_filename);
M_list = load(M_list_filename);
%
% T_max = 5; % maximum T value
% nT = 64; % number of T values
% %
% H_max = 0; % maximum H value
% nH = 1; % number of H values
% %
% % END OF USER INPUT
% %
% if nT > 1
%     %
%     T(:,1) = T_max/nT : T_max/nT : T_max;
%     %
% elseif nT == 1
%     %
%     T(:,1) = T_max;
%     %
% end
% %
% if nH > 1
%     %
%     H(:,1) = 0 : H_max/(nH-1) : H_max; 
%     %
% elseif nH == 1
%     %
%     H(:,1) = H_max;
%     %
% end 
%
% NORMALIZATION OF MAGNETIZATION AND ENERGY (imposes unitary length spin vector)
%
M_list = M_list ./ max(M_list) .* N_atm;
E_list = E_list ./ max(E_list) .* 1/2 .* N_atm .* NN; 
%
% CLEANUP JDOS, IF NEEDED
%
if sum(JDOS_aprox(:,end-1)) == 0
    %
    JDOS_aprox(:,(length(M_list)+1)/2 + 1 : end) = 0;
    %
end
%
log_JDOS_aprox = log(JDOS_aprox);
clear JDOS_aprox;
%
% CHECK FOR HALF OR FULL JDOS
%
index_M0 = (length(M_list)-1)/2 + 1;
%
if all(isinf(log_JDOS_aprox(:,index_M0+1))) % half JDOS
    %
    disp('Half JDOS detected - mirroring.')
    log_JDOS_aprox(:,index_M0+1:length(M_list)) = log_JDOS_aprox(:,index_M0-1:-1:1);
    %
else
    %
    disp('Full JDOS detected - no mirroring.')
    %
end
%
log_Z_M = nan(length(H), length(T), length(M_list));
%
q = 1;
hits = find(~isinf(log_JDOS_aprox(:,q)));
log_Z_M(:,:,q) = log_JDOS_aprox(hits(1),q) + ((E_list(hits(1)) - M_list(q).*H)*(-1./T'));
%
for k = 2:length(hits)
    %
    log_Z_M(:,:,q) = log_Z_M(:,:,q) + log(1 + exp( log_JDOS_aprox(hits(k),q) + ((E_list(hits(k)) - M_list(q).*H)*(-1./T') ) - log_Z_M(:,:,q)));
    %
end
%
log_Z = log_Z_M(:,:,q);
G_temp(:,:,q) = repmat(-T', [length(H),1]).*(log_Z_M(:,:,q));
%
for q = 2:length(M_list(:,1)) %index_M0
    %
    hits = find(~isinf(log_JDOS_aprox(:,q)));
    %
    log_Z_M(:,:,q) = log_JDOS_aprox(hits(1),q) + ((E_list(hits(1)) - M_list(q).*H)*(-1./T')) ;
    %
    for k = 2:length(hits)
        %
        log_Z_M(:,:,q) = log_Z_M(:,:,q) + log(1 + exp( log_JDOS_aprox(hits(k),q) + ((E_list(hits(k)) - M_list(q).*H)*(-1./T') ) - log_Z_M(:,:,q)));
        %
    end
    %
    log_Z = log_Z + log(1 + exp(log_Z_M(:,:,q) - log_Z));
    G_temp(:,:,q) = repmat(-T', [length(H),1]).*(log_Z_M(:,:,q));
    %
end
%
G = permute(G_temp,[3,2,1]);
%
disp('Free Energy calculations completed')
%
% CALCULATION OF AVERAGE THERMODYNAMIC VALUES
%
avg_M = nan(length(H),length(T));
avg_abs_M = nan(length(H),length(T));
avg_M2 = nan(length(H),length(T));
%
for i=1:length(T)
    %
    for j=1:length(H)
        %
        avg_M(j,i) = 0;
        avg_abs_M(j,i) = 0;
        avg_M2(j,i) = 0;
        %
        for q = 1:length(M_list(:,1)) %index_M0
            %
            avg_M(j,i) = avg_M(j,i) + M_list(q)*exp(log_Z_M(j,i,q) - log_Z(j,i));
            avg_abs_M(j,i) = avg_abs_M(j,i) + abs(M_list(q))*exp(log_Z_M(j,i,q) - log_Z(j,i));
            avg_M2(j,i) = avg_M2(j,i) + (M_list(q)^2)*exp(log_Z_M(j,i,q) - log_Z(j,i));
        end
        %
    end
end
%
% FIND MINIMUM OF Gibbs FREE ENERGY
%
temp = nan(length(M_list), 1);
M_minG = nan(length(H), length(T));
index_M_minG = nan(length(H), length(T));
minG = nan(length(H), length(T));
%
for i = 1:length(T)
    for j = 1:length(H)
        for q = index_M0:length(M_list(:,1)) % only M >= 0
            temp(q,1) = G(q,i,j);
        end
        [u,v] = min(temp(:,1));
        M_minG(j,i) = M_list(v);
        index_M_minG(j,i) = v;
        minG(j,i) = u;
        clear v
        %
    end
end
%
% DATA NORMALIZATION
%
avg_abs_M = avg_abs_M/N_atm;
avg_M = avg_M/N_atm;
avg_M2 = avg_M2/(N_atm^2);
M_minG = M_minG/N_atm;
minG = minG/N_atm;
G = G/N_atm;
%
abs_avg_M = abs(avg_M);
%
% INTERPOLATE G TO GET M_minG
%
M_minG_smooth = M_minG; %nan(length(H), length(T));
minG_smooth = minG;
%
for i = 1:length(T)
    %
    for j = 1:length(H)
        %
        G_temp = G(:,i,j);
        %
%         if index_M_minG(j,i) == 1
%             %
%             aux = lagr(M_list(index_M_minG(j,i):index_M_minG(j,i)+2), -G_temp(index_M_minG(j,i):index_M_minG(j,i)+2));
%             if aux(1) <= 1 && aux(1) >= 0
%                 M_minG_smooth(j,i) = aux(1)/N_atm;
%                 minG_smooth(j,i) = -aux(2);
%             end
%             %
%         end
        %
        if index_M_minG(j,i) < length(M_list)
            %
            aux = lagr(M_list(index_M_minG(j,i)-1:index_M_minG(j,i)+1), -G_temp(index_M_minG(j,i)-1:index_M_minG(j,i)+1));
            M_minG_smooth(j,i) = aux(1)/N_atm;
            minG_smooth(j,i) = -aux(2);
            %
        end
        %
        if index_M_minG(j,i) == length(M_list)
            %
            aux = lagr(M_list(index_M_minG(j,i)-2:index_M_minG(j,i)), -G_temp(index_M_minG(j,i)-2:index_M_minG(j,i)));
            if G_temp(index_M_minG(j,i)-1)-(G_temp(index_M_minG(j,i)-2)+G_temp(index_M_minG(j,i)))/2 < 0 && aux(1) <= 1 && aux(1) >= 0
                M_minG_smooth(j,i) = aux(1)/N_atm;
                minG_smooth(j,i) = -aux(2);
            end
            %
        end
        %
    end
    %
end
%
if length(H) > 1 || (length(H) == 1 && H(1) > 0)
    %
    M2_minG = M_minG.^2;
    M2_minG_smooth = M_minG_smooth.^2;
    avg_abs_M2 = avg_abs_M.^2;
    abs_avg_M2 = abs(avg_M2);
    %
    H_sob_M_minG = H ./ M_minG;
    H_sob_M_minG_smooth = H ./ M_minG_smooth;
    H_sob_avg_M = H ./ avg_M;
    H_sob_avg_abs_M = H ./ avg_abs_M;
    H_sob_abs_avg_M = H ./ abs_avg_M;
    %
end
%
% ENTROPY AND SPECIFIC HEAT FROM G DERIVATIVES (+G_smooth)
%
if length(T) > 1
    %
    T_dS = T(1:(end-1)) + (diff(T)/2);
    S_minG = nan(length(H), length(T_dS));
    S_minG_smooth = nan(length(H), length(T_dS));
    C_minG = nan(length(H), length(T_dS) - 1);
    C_minG_smooth = nan(length(H), length(T_dS) - 1);
    %
    for j = 1:length(H)
        %
        S_minG(j,:) = -diff(minG(j,:)) ./ diff(T)';
        C_minG(j,:) = -T(2:(end-1))' .* diff(-S_minG(j,:)) ./ diff(T_dS)';
        S_minG_smooth(j,:) = -diff(minG_smooth(j,:)) ./ diff(T)';
        C_minG_smooth(j,:) = -T(2:(end-1))' .* diff(-S_minG_smooth(j,:)) ./ diff(T_dS)';
        %
    end
    %
    % ENTHALPY FROM GIBBS FREE ENERGY MINIMA
    %
    S_minG_T_interp = nan(length(H), length(T));
    S_minG_smooth_T_interp = nan(length(H), length(T));
    %
    for j = 1:length(H)
        %
        S_minG_T_interp(j,:) = interp1(T_dS, S_minG(j,:), T, 'linear', 'extrap');
        S_minG_smooth_T_interp(j,:) = interp1(T_dS, S_minG_smooth(j,:), T, 'linear', 'extrap');
        %
    end
    %
    E_from_min_G = minG + T' .* S_minG_T_interp;
    E_from_min_G_smooth = minG_smooth + T' .* S_minG_smooth_T_interp;
    %
    % DATA NORMALIZATION
    %
%     S_minG = S_minG/N_atm;
%     C_minG = C_minG/N_atm;
%     E_from_min_G = E_from_min_G/N_atm;
    %
end
%
% END OF CALCULATIONS FROM MIN G AND FROM AVG M 
%
% ------------------------------------------------------------------------
% ------------------------------------------------------------------------
%
% PLOT FOR SINGLE T AND H VALUE
%
if length(T) == 1 && length(H) == 1
    %
    figure('Name','Average M','NumberTitle','off')
    subplot(3,2,1)
    plot(T, avg_abs_M, '.-'), xlabel('T'), ylabel('<|M|>'), grid
    subplot(3,2,2)
    plot(H, avg_abs_M, '.-'), xlabel('H'), ylabel('<|M|>'), grid
    subplot(3,2,3)
    plot(T, avg_M, '.-'), xlabel('T'), ylabel('<M|'), grid
    subplot(3,2,4)
    plot(H, avg_M, '.-'), xlabel('H'), ylabel('<|M|>'), grid
    subplot(3,2,5)
    plot(T, abs_avg_M, '.-'), xlabel('T'), ylabel('<M|'), grid
    subplot(3,2,6)
    plot(H, abs_avg_M, '.-'), xlabel('H'), ylabel('<|M|>'), grid
    %
    figure('Name','M from G minima','NumberTitle','off')
    subplot(2,3,1)
    plot(M_list, G(:,:,1),'.-'), xlabel('M'), ylabel('G'), grid
    subplot(2,3,2)
    plot(T, minG,'.-'), xlabel('T'), ylabel('G min'), grid
    subplot(2,3,3)
    plot(T, M_minG,'.-'), xlabel('T'), ylabel('M from G min'), grid
    subplot(2,3,4)
    plot(H, minG,'.-'), xlabel('H'), ylabel('G min'), grid
    subplot(2,3,5)
    plot(H, M_minG,'.-'), xlabel('H'), ylabel('M from G min'), grid
    %
    figure('Name','M from G smooth minima','NumberTitle','off')
    subplot(2,3,1)
    plot(M_list, G(:,:,1),'.-'), xlabel('M'), ylabel('G'), grid
    subplot(2,3,2)
    plot(T, minG_smooth,'.-'), xlabel('T'), ylabel('G min smooth'), grid
    subplot(2,3,3)
    plot(T, M_minG_smooth,'.-'), xlabel('T'), ylabel('M from G min smooth'), grid
    subplot(2,3,4)
    plot(H, minG_smooth,'.-'), xlabel('H'), ylabel('G min smooth'), grid
    subplot(2,3,5)
    plot(H, M_minG_smooth,'.-'), xlabel('H'), ylabel('M from G min smooth'), grid
    %
end
%
%
% PLOTS FOR T DEPENDENCE
%
if length(T) > 1 && length(H) == 1
    %
    figure('Name','Average M','NumberTitle','off')
    subplot(1,3,1)
    plot(T, avg_abs_M, '.-'), xlabel('T'), ylabel('<|M|>'), grid
    subplot(1,3,2)
    plot(T, avg_M, '.-'), xlabel('T'), ylabel('<M>'), grid
    subplot(1,3,3)
    plot(T, abs_avg_M, '.-'), xlabel('T'), ylabel('|<M>|'), grid
    %
    figure('Name','from G minima','NumberTitle','off')
    subplot(2,3,1)
    plot(M_list, G(:,:,1),'.-'), xlabel('M'), ylabel('G'), grid
    subplot(2,3,2)
    plot(T_dS, S_minG, '.-'), xlabel('T'), ylabel('S from G min'), grid
    subplot(2,3,3)
    plot(T(2:(end-1)), C_minG, '.-'), xlabel('T'), ylabel('C from G min'), grid
    subplot(2,3,4)
    plot(T, E_from_min_G, '.-'), xlabel('T'), ylabel('E from min G'), grid
    subplot(2,3,5)
    plot(T, minG,'.-'), xlabel('T'), ylabel('G min'), grid
    subplot(2,3,6)
    plot(T, M_minG,'.-'), xlabel('T'), ylabel('M from G min'), grid
    %
    figure('Name','from G minima smooth','NumberTitle','off')
    subplot(2,3,1)
    plot(M_list, G(:,:,1),'.-'), xlabel('M'), ylabel('G'), grid
    subplot(2,3,2)
    plot(T_dS, S_minG_smooth, '.-'), xlabel('T'), ylabel('S from G min smooth'), grid
    subplot(2,3,3)
    plot(T(2:(end-1)), C_minG_smooth, '.-'), xlabel('T'), ylabel('C from G min smooth'), grid
    subplot(2,3,4)
    plot(T, E_from_min_G_smooth, '.-'), xlabel('T'), ylabel('E from min G smooth'), grid
    subplot(2,3,5)
    plot(T, minG_smooth,'.-'), xlabel('T'), ylabel('G min smooth'), grid
    subplot(2,3,6)
    plot(T, M_minG_smooth,'.-'), xlabel('T'), ylabel('M from G min smooth'), grid
    %
end
%
% PLOTS FOR H DEPENDENCE
%
if length(T) == 1 && length(H) > 1
    %
    figure('Name','Average M','NumberTitle','off')
    subplot(1,3,1)
    plot(H,avg_abs_M,'.-'), xlabel('H'), ylabel('<|M|>'), grid
    subplot(1,3,2)
    plot(H,avg_M,'.-'), xlabel('H'), ylabel('<M>'), grid
    subplot(1,3,3)
    plot(H,abs_avg_M,'.-'), xlabel('H'), ylabel('<M>'), grid
    %
    figure('Name','from G minima','NumberTitle','off')
    subplot(1,3,1)
    plot(M_list, G(:,:,1),'.-'), xlabel('M'), ylabel('G'), grid
    subplot(1,3,2)
    plot(H, minG,'.-'), xlabel('H'), ylabel('G min'), grid
    subplot(1,3,3)
    plot(H, M_minG,'.-'), xlabel('H'), ylabel('M from G min'), grid
    %
    figure('Name','from G minima smooth','NumberTitle','off')
    subplot(1,3,1)
    plot(M_list, G(:,:,1),'.-'), xlabel('M'), ylabel('G'), grid
    subplot(1,3,2)
    plot(H, minG_smooth,'.-'), xlabel('H'), ylabel('G min smooth'), grid
    subplot(1,3,3)
    plot(H, M_minG_smooth,'.-'), xlabel('H'), ylabel('M from G min smooth'), grid
    %
    %
end
%
% PLOTS FOR (H,T) DEPENDENCE
%
if length(H)>1 && length(T)>1
    %
    % MAGNETOCALORIC EFFECT FROM MAXWELL RELATION
    %
    T_dS = T(1:(end-1),1) + diff(T)/2;
    %
    dM_minG_dT = nan(length(H), length(T_dS));
    M_minG_dS = nan(length(H), length(T_dS));
    %
    dM_minG_smooth_dT = nan(length(H), length(T_dS));
    M_minG_smooth_dS = nan(length(H), length(T_dS));
    %
    d_avg_M_dT = nan(length(H), length(T_dS));
    d_avg_abs_M_dT = nan(length(H), length(T_dS));
    d_abs_avg_M_dT = nan(length(H), length(T_dS));
    %
    avg_M_dS = nan(length(H), length(T_dS));
    avg_abs_M_dS = nan(length(H), length(T_dS));
    abs_avg_M_dS = nan(length(H), length(T_dS));
    %
    for j = 1:length(H)
        %
        dM_minG_dT(j,:) = diff(M_minG(j,:)) ./ diff(T)';
        M_minG_dS(j,:) = interp1(T, M_minG(j,:), T_dS, 'linear');
        %
        dM_minG_smooth_dT(j,:) = diff(M_minG_smooth(j,:)) ./ diff(T)';
        M_minG_smooth_dS(j,:) = interp1(T, M_minG_smooth(j,:), T_dS, 'linear');
        %
        d_avg_M_dT(j,:) = diff(avg_M(j,:)) ./ diff(T)';
        d_avg_abs_M_dT(j,:) = diff(avg_abs_M(j,:)) ./ diff(T)';
        d_abs_avg_M_dT(j,:) = diff(abs_avg_M(j,:)) ./ diff(T)';
        %
        avg_M_dS(j,:) = interp1(T, avg_M(j,:), T_dS, 'linear');
        avg_abs_M_dS(j,:) = interp1(T, avg_abs_M(j,:), T_dS, 'linear');
        abs_avg_M_dS(j,:) = interp1(T, abs_avg_M(j,:), T_dS, 'linear');
        %
    end
    %
    dS_M_minG = -cumtrapz(H, dM_minG_dT);
    M2_minG_dS = M_minG_dS.^2;
    %
    dS_M_minG_smooth = -cumtrapz(H, dM_minG_smooth_dT);
    M2_minG_smooth_dS = M_minG_smooth_dS.^2;
    %
    dS_avg_M = -cumtrapz(H, d_avg_M_dT);
    dS_avg_abs_M = -cumtrapz(H, d_avg_abs_M_dT);
    dS_abs_avg_M = -cumtrapz(H, d_abs_avg_M_dT);
    %
    avg_M2_dS = avg_M_dS.^2;
    avg_abs_M2_dS = avg_abs_M_dS.^2;
    abs_avg_M2_dS = abs_avg_M_dS.^2;
    %
    figure('Name','Average thermodynamic properties','NumberTitle','off')
    subplot(3,5,1)
    plot(T, avg_abs_M, '.-'), xlabel('T'), ylabel('<|M|>'), grid
    subplot(3,5,2)
    plot(H, avg_abs_M, '.-'), xlabel('H'), ylabel('<|M|>'), grid
    subplot(3,5,3)
    plot(avg_abs_M2, H_sob_avg_abs_M, '.-'), xlabel('<|M|> ^2'), ylabel('H / <|M|>'), grid
    subplot(3,5,4)
    plot(T_dS, dS_avg_abs_M, '.-'), xlabel('T'), ylabel('dS from <|M|>'), grid
    subplot(3,5,5)
    plot(avg_abs_M2_dS, dS_avg_abs_M, '.-'), xlabel('<|M|> ^2'), ylabel('dS from <|M|>'), grid
    subplot(3,5,6)
    plot(T, avg_M, '.-'), xlabel('T'), ylabel('<M>'), grid
    subplot(3,5,7)
    plot(H, avg_M, '.-'), xlabel('H'), ylabel('<M>'), grid
    subplot(3,5,8)
    plot(avg_M2, H_sob_avg_M, '.-'), xlabel('<M> ^2'), ylabel('H / <M>'), grid
    subplot(3,5,9)
    plot(T_dS, dS_avg_M, '.-'), xlabel('T'), ylabel('dS from <M>'), grid
    subplot(3,5,10)
    plot(avg_M2_dS, dS_avg_M, '.-'), xlabel('<M> ^2'), ylabel('dS from <M>'), grid
    subplot(3,5,11)
    plot(T, abs_avg_M, '.-'), xlabel('T'), ylabel('|<M>|'), grid
    subplot(3,5,12)
    plot(H, abs_avg_M, '.-'), xlabel('H'), ylabel('|<M>|'), grid
    subplot(3,5,13)
    plot(abs_avg_M2, H_sob_abs_avg_M, '.-'), xlabel('|<M>| ^2'), ylabel('H / |<M>|'), grid
    subplot(3,5,14)
    plot(T_dS, dS_abs_avg_M, '.-'), xlabel('T'), ylabel('dS from |<M>|'), grid
    subplot(3,5,15)
    plot(abs_avg_M2_dS, dS_abs_avg_M, '.-'), xlabel('|<M>| ^2'), ylabel('dS from |<M>|'), grid
    %
    figure('Name','from G minima','NumberTitle','off')
    subplot(3,4,1)
    plot(M_list, G(:,:,1),'.-'), xlabel('M'), ylabel('zero-field G'), grid
    subplot(3,4,2)
    plot(M_list, G(:,:,length(H)),'.-'), xlabel('M'), ylabel('max field G'), grid
    subplot(3,4,3)
    plot(T_dS, S_minG, '.-'), xlabel('T'), ylabel('S from G min'), grid
    subplot(3,4,4)
    plot(T(2:(end-1)), C_minG, '.-'), xlabel('T'), ylabel('C from G min'), grid
    subplot(3,4,5)
    plot(T, E_from_min_G, '.-'), xlabel('T'), ylabel('E from min G'), grid
    subplot(3,4,6)
    plot(T, minG,'.-'), xlabel('T'), ylabel('G min'), grid
    subplot(3,4,7)
    plot(T, M_minG,'.-'), xlabel('T'), ylabel('M from G min'), grid
    subplot(3,4,8)
    plot(H, M_minG,'.-'), xlabel('H'), ylabel('M from G min'), grid
    subplot(3,4,9)
    plot(M2_minG, H_sob_M_minG, '.-'), xlabel('M from G min^2'),ylabel('H / M from G min'), grid
    subplot(3,4,10)
    plot(T_dS, dS_M_minG, '.-'),xlabel('T'),ylabel('dS from M from G min'), grid
    subplot(3,4,11)
    plot(M2_minG_dS, dS_M_minG, '.-'), xlabel('M from G min^2'), ylabel('dS from M from G min'), grid
    %
    figure('Name','from G minima smooth','NumberTitle','off')
    subplot(3,4,1)
    plot(M_list, G(:,:,1),'.-'), xlabel('M'), ylabel('zero-field G'), grid
    subplot(3,4,2)
    plot(M_list, G(:,:,length(H)),'.-'), xlabel('M'), ylabel('max field G'), grid
    subplot(3,4,3)
    plot(T_dS, S_minG_smooth, '.-'), xlabel('T'), ylabel('S from G min smooth'), grid
    subplot(3,4,4)
    plot(T(2:(end-1)), C_minG_smooth, '.-'), xlabel('T'), ylabel('C from G min smooth'), grid
    subplot(3,4,5)
    plot(T, E_from_min_G_smooth, '.-'), xlabel('T'), ylabel('E from min G smooth'), grid
    subplot(3,4,6)
    plot(T, minG_smooth,'.-'), xlabel('T'), ylabel('G min smooth'), grid
    subplot(3,4,7)
    plot(T, M_minG_smooth,'.-'), xlabel('T'), ylabel('M from G min smooth'), grid
    subplot(3,4,8)
    plot(H, M_minG_smooth,'.-'), xlabel('H'), ylabel('M from G min smooth'), grid
    subplot(3,4,9)
    plot(M2_minG, H_sob_M_minG_smooth, '.-'), xlabel('M from G min^2'),ylabel('H / M from G min smooth'), grid
    subplot(3,4,10)
    plot(T_dS, dS_M_minG_smooth, '.-'),xlabel('T'),ylabel('dS from M from G min smooth'), grid
    subplot(3,4,11)
    plot(M2_minG_dS, dS_M_minG_smooth, '.-'), xlabel('M from G min^2'), ylabel('dS from M from G min smooth'), grid
    %
end
%
drawnow()
%
disp('Average energy and specific heat calculations...')
%
avg_C = nan(length(H),length(T));
%
% E_temp = E_list(sum(JDOS_aprox,2)~=0);
E_temp = E_list(sum(exp(log_JDOS_aprox),2)~=0);
%
log_parte1_final_E = cell(1,length(E_temp));
%
for b = 1:length(E_temp(:,1))
    %
    counter = 0;
    %
    for q = 1:length(M_list(:,1)) %index_M0 
        %
        for z = 1:length(E_list(:,1))
            %
            if E_list(z) == E_temp(b,1)
                %
                counter = counter + 1;
                %
                log_parte1_final_E{b}(counter,1) = M_list(q);
                log_parte1_final_E{b}(counter,2) = log_JDOS_aprox(z,q);
                %
            end
        end
    end
end
%
log_avg_E = nan(length(H), length(T));
log_avg_E2 = nan(length(H), length(T));
%
for j = 1:length(H)
    %
    for i = 1:length(T)
        %
        b = 1;
        k = 1;
        rehits = find(~isinf(log_parte1_final_E{b}(:,2)));
        %
        log_avg_E(j,i) = log((E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j))) + log_parte1_final_E{b}(rehits(k),2) + (-1/T(i) * (E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j)));
        log_avg_E2(j,i) = log((E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j))^2) + log_parte1_final_E{b}(rehits(k),2) + (-1/T(i) * (E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j)));
        %
        if length(rehits) > 1
            %
            for k = 2:length(rehits) %Q(E)
                %
                log_avg_E(j,i) = log_avg_E(j,i) + log(1 + exp(log((E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j))) + log_parte1_final_E{b}(rehits(k),2) + (-1/T(i) * (E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j))) - log_avg_E(j,i)));
                log_avg_E2(j,i) = log_avg_E2(j,i) + log(1 + exp(log((E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j))^2) + log_parte1_final_E{b}(rehits(k),2) + (-1/T(i) * (E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j))) - log_avg_E2(j,i)));
                %
            end
            %
        end
        %
        for b = 2:length(E_temp(:,1))
            %
            rehits = find(~isinf(log_parte1_final_E{b}(:,2)));
            %
            for k = 1:length(rehits) %Q(E)
                %
                log_avg_E(j,i) = log_avg_E(j,i) + log(1 + exp(log((E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j))) + log_parte1_final_E{b}(rehits(k),2) + (-1/T(i) * (E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j))) - log_avg_E(j,i)));
                log_avg_E2(j,i) = log_avg_E2(j,i) + log(1 + exp(log((E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j))^2) + log_parte1_final_E{b}(rehits(k),2) + (-1/T(i) * (E_temp(b) - log_parte1_final_E{b}(rehits(k),1) * H(j))) - log_avg_E2(j,i)));
                %
            end
            %
        end
        %
        log_avg_E(j,i) = log_avg_E(j,i) - log_Z(j,i);
        log_avg_E2(j,i) = log_avg_E2(j,i) - log_Z(j,i);
        %
        avg_C(j,i) = 1/(T(i)^2).*(exp(log_avg_E2(j,i)) - exp(log_avg_E(j,i)).^2);
        %
    end
    %
end
%
% DATA NORMALIZATION
%
log_avg_E = log_avg_E - log(N_atm);
log_avg_E2 = log_avg_E2 - log(N_atm);
avg_C = real(avg_C / N_atm);
%
avg_E = real(exp(log_avg_E));
avg_E2 = real(exp(log_avg_E2));
%
disp('Average energy and specific heat calculations completed')
%
if length(T) == 1
    %
    if length(H) == 1
        %
        figure('Name','Average E, C','NumberTitle','off')
        subplot(2,3,1)
        plot(T, avg_E, '.-'), xlabel('T'), ylabel('<E>'), grid
        subplot(2,3,2)
        plot(T, avg_E2, '.-'), xlabel('T'), ylabel('<E^2>'), grid
        subplot(2,3,3)
        plot(T, avg_C, '.-'), xlabel('T'), ylabel('<C>'), grid
        subplot(2,3,4)
        plot(H, avg_E, '.-'), xlabel('H'), ylabel('<E>'), grid
        subplot(2,3,5)
        plot(H, avg_E2, '.-'), xlabel('H'), ylabel('<E^2>'), grid
        subplot(2,3,6)
        plot(H, avg_C, '.-'), xlabel('H'), ylabel('<C>'), grid
        %
    elseif length(H) > 1
        %
        figure('Name','Average E, C','NumberTitle','off')
        subplot(1,3,1)
        plot(H, avg_E, '.-'), xlabel('H'), ylabel('<E>'), grid
        subplot(1,3,2)
        plot(H, avg_E2, '.-'), xlabel('H'), ylabel('<E^2>'), grid
        subplot(1,3,3)
        plot(H, avg_C, '.-'), xlabel('H'), ylabel('<C>'), grid
        %
    end
    %
elseif length(T) > 1
    %
    avg_S = nan(length(H), length(T));
    avg_G = nan(length(H), length(T));
    %
    for j = 1:length(H)
        %
        avg_S(j,:) = cumtrapz(T, avg_C(j,:) ./ T');
        avg_G(j,:) = avg_E(j,:) - T' .* avg_S(j,:);
        %
    end
    %
    if length(H) == 1
        %
        figure('Name','Average E, C, S, G','NumberTitle','off')
        subplot(2,3,1)
        plot(T, avg_E, '.-'), xlabel('T'), ylabel('<E>'), grid
        subplot(2,3,2)
        plot(T, avg_E2, '.-'), xlabel('T'), ylabel('<E^2>'), grid
        subplot(2,3,3)
        plot(T, avg_C, '.-'), xlabel('T'), ylabel('<C>'), grid
        subplot(2,3,4)
        plot(T, avg_S, '.-'), xlabel('T'), ylabel('<S>'), grid
        subplot(2,3,5)
        plot(T, avg_G, '.-'), xlabel('T'), ylabel('<G>'), grid
        %
    elseif length(H) > 1
        %
        % SCALAR M FROM dG/dH
        %
        H_dS = H(1:(end-1)) + (diff(H)/2);
        avg_M_avg_G = nan(length(H_dS), length(T));
        avg_M_avg_G_extrap = nan(length(H), length(T));
        %
        for i = 1:length(T)
            %
            avg_M_avg_G(:,i) = -diff(avg_G(:,i)) ./ diff(H);
            avg_M_avg_G_extrap(:,i) = interp1(H_dS, avg_M_avg_G(:,i), H, 'linear', 'extrap');
            %
        end
        %
        d_avg_M_avg_G_extrap_dT = nan(length(H), length(T_dS));
        avg_M_avg_G_extrap_dS = nan(length(H), length(T_dS));
        %
        for j = 1:length(H)
            %
            d_avg_M_avg_G_extrap_dT(j,:) = diff(avg_M_avg_G_extrap(j,:)) ./ diff(T)';
            avg_M_avg_G_extrap_dS(j,:) = interp1(T, avg_M_avg_G_extrap(j,:), T_dS, 'linear');
            %
        end
        %
        dS_avg_M_avg_G_extrap = -cumtrapz(H, d_avg_M_avg_G_extrap_dT);
        avg_M2_avg_G_extrap = avg_M_avg_G_extrap.^2;
        avg_M2_avg_G_extrap_dS = avg_M_avg_G_extrap_dS.^2;
        H_sob_avg_M_avg_G_extrap = H ./ avg_M_avg_G_extrap;
        %
        figure('Name','Average E, C, S , G, and M from G','NumberTitle','off')
        subplot(2,5,1)
        plot(T, avg_E, '.-'), xlabel('T'), ylabel('<E>'), grid
        subplot(2,5,2)
        plot(T, avg_E2, '.-'), xlabel('T'), ylabel('<E^2>'), grid
        subplot(2,5,3)
        plot(T, avg_C, '.-'), xlabel('T'), ylabel('<C>'), grid
        subplot(2,5,4)
        plot(T, avg_S, '.-'), xlabel('T'), ylabel('<S>'), grid
        subplot(2,5,5)
        plot(T, avg_G, '.-'), xlabel('T'), ylabel('<G>'), grid
        subplot(2,5,6)
        plot(T, avg_M_avg_G_extrap, '.-'), xlabel('T'), ylabel('<M> from <G>'), grid
        subplot(2,5,7)
        plot(H, avg_M_avg_G_extrap, '.-'), xlabel('H'), ylabel('<M> from <G>'), grid
        subplot(2,5,8)
        plot(avg_M2_avg_G_extrap, H_sob_avg_M_avg_G_extrap, '.-'), xlabel('<M>2 from <G>'), ylabel('H / <M> from <G>'), grid
        subplot(2,5,9)
        plot(T_dS, dS_avg_M_avg_G_extrap,'.-'), xlabel('T'), ylabel('dS from <M> from <G>'), grid
        subplot(2,5,10)
        plot(avg_M2_avg_G_extrap_dS, dS_avg_M_avg_G_extrap, '.-'), xlabel('<M>2 from <G>'), ylabel('dS from <M> from <G>'), grid
        %
    end
    %
end
%
function lagr=lagr(xm,ym)
% determinacao de o ma'ximo de uma funcao discreta
%
% input: coordenadas de 3 pontos vizinhos de ordenadas maiores
%            matrizes xm e ym
% output: coordenadas do ponto ma'ximo (xmax,ymax)
%

xab=xm(1)-xm(2);
xac=xm(1)-xm(3);
xbc=xm(2)-xm(3);

a=ym(1)/(xab*xac);
b=-ym(2)/(xab*xbc);
c=ym(3)/(xac*xbc);

xml=(b+c)*xm(1)+(a+c)*xm(2)+(a+b)*xm(3);
xmax=0.5*xml/(a+b+c);

xta=xmax-xm(1);
xtb=xmax-xm(2);
xtc=xmax-xm(3);

ymax=a*xtb*xtc+b*xta*xtc+c*xta*xtb;

lagr(1)=xmax;
lagr(2)=ymax;
%
end