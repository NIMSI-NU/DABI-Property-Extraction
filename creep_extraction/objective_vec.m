function [cost, evalTime] = objective_vec(CreepParams,in,idx)

% Implemented for IN718 by June 13th

% in: all the input parameters

% CreepParams: Initial sets of creep parameters during optimization (1-by-4
% vector). CreepParams(1~4) are, in hyperbolic-sine law, the coefficient A,
% stress coefficient B, power n, and activation energy Eact.

tStart = tic;

%% initialization
mesh = in.mesh;
saveDir = in.saveDir;
res = in.res;
lb = in.LBori;
ub = in.UBori;
CPUs=1;

%% Compute results of current iteration
% Write and run ABAQUS input files in parallel
% Abaqus will run input files in working directory but will save file in
% the current folder. To save disk space, a line to delete the ODB and lock file
% is used here in the loop. But .dat and .sat file will be kept for all results.
Elasticity = in.Modulus;
TjTend = in.PressJump;
Pressure = TjTend(:,1:3);
Duration = TjTend(:,4);
Temperature = in.Temperature;
R = 8.314;

Loss_temp = [];

for itemp = 1:length(Elasticity)
%% scale back
% log(A), n0, ninf, Abump, taur, taud, Qe5
Q = ((CreepParams(7))*(ub(7)-lb(7))+lb(7)) * 1e5;
A4 = 10^((CreepParams(1))*(ub(1)-lb(1))+lb(1));
Acorrected = A4 * exp(-Q / R / Temperature(itemp)); % temperature term

Params = [Acorrected,...
    (CreepParams(2))*(ub(2)-lb(2))+lb(2),...
    (CreepParams(3))*(ub(3)-lb(3))+lb(3),...
    (CreepParams(4))*(ub(4)-lb(4))+lb(4),...
    (CreepParams(5))*(ub(5)-lb(5))+lb(5),...
    (CreepParams(6))*(ub(6)-lb(6))+lb(6),...
    0.0]; % The model allows for strain hardening term

%%
JobName = strcat("DABI_nt_",num2str(idx),'_',num2str(itemp));
InFileName = strcat(JobName,".inp");
SubFileName = strcat(JobName,".f");
lckname = strcat(JobName,".lck"); 
if exist(lckname,'file')==2; delete(lckname); end % Remove lock file

%% Write and run
writeAbaqusInputFile(InFileName, saveDir, mesh, Pressure(itemp,:), Elasticity(itemp)*1000,Duration(itemp)); % unchanged writer

% --- Write subroutine ---
writeSubroutine_nt(SubFileName, saveDir, Params);

% --- Run Abaqus job (same pattern as yours) ---
cmd_str = strcat('abaqus job=', JobName, ...
                 ' input=', InFileName, ...
                 ' user=',  SubFileName, ...
                 ' interactive cpus=', num2str(CPUs), ' > NUL 2>&1');
system(cmd_str);

if idx > 2000
    disp('test simulation')
else
ODBFileName = strcat(saveDir,JobName,'.odb');
if exist(ODBFileName, 'file') == 2
    delete(ODBFileName);
end
end

%% compute loss
costtmp = ComputeCost(JobName, in, itemp);
Loss_temp = [Loss_temp;costtmp];

end

%% ending
cost = mean(Loss_temp);
evalTime = toc(tStart);

end