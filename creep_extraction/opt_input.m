function in = opt_input(F_mesh,F_truth)

%% Sparsity regularization parameters
withinRange = 5.0;
in.Temperature = [715.76, 765.90, 793.30, 803.53, 712.27, 770.79, 800.17, 810.57, 743.05, 780.32, 802.45, 810.48] + 273.15;
in.Modulus = [108.425, 103.41, 100.67, 99.647, 108.773, 102.921,99.9827, 98.94281, 105.195, 101.968, 99.755, 98.952]; % tensile tests
in.PressJump = [8 8 3 4;
    8 8 3 4;
    8 8 3 4;
    8 8 3 4;
    8 8 3 4;
    8 8 3 4;
    8 8 3 4;
    8 8 3 4;
    8 8 3 4;
    8 10 4.05 8;
    8 10 4.05 8;
    8 8 3 4;
    ];

%% Read simulation config.
in.mesh.nodes = readmatrix(strcat(F_mesh,"nodes.txt"));
ElemArray = readmatrix(strcat(F_mesh,"elements.txt"),"Delimiter",",");
in.mesh.elem = ElemArray;

in.mesh.BulgeNodes = readmatrix(strcat(F_mesh,"TOPDIMPLENODES.txt"));
in.mesh.BulgeNodes = in.mesh.BulgeNodes(:);
nonNaNIndex = ~isnan(in.mesh.BulgeNodes);
in.mesh.BulgeNodes = in.mesh.BulgeNodes(nonNaNIndex);
in.mesh.BulgeNodes = sortrows(in.mesh.BulgeNodes,1);

%% BCs
% fixed
in.mesh.FixedBCset = readmatrix(strcat(F_mesh,"FIXEDBC.txt"));
in.mesh.FixedBCset = in.mesh.FixedBCset(:);
nonNaNIndex = ~isnan(in.mesh.FixedBCset);
in.mesh.FixedBCset = in.mesh.FixedBCset(nonNaNIndex);

%% pressure
in.mesh.P800Surf1 = readmatrix(strcat(F_mesh,"Surf_1_S1.txt"));
in.mesh.P800Surf1 = in.mesh.P800Surf1(:);
nonNaNIndex = ~isnan(in.mesh.P800Surf1);
in.mesh.P800Surf1 = in.mesh.P800Surf1(nonNaNIndex);

in.mesh.P800Surf3 = readmatrix(strcat(F_mesh,"Surf_1_S3.txt"));
in.mesh.P800Surf3 = in.mesh.P800Surf3(:);
nonNaNIndex = ~isnan(in.mesh.P800Surf3);
in.mesh.P800Surf3 = in.mesh.P800Surf3(nonNaNIndex);

in.mesh.P800Surf4 = readmatrix(strcat(F_mesh,"Surf_1_S4.txt"));
in.mesh.P800Surf4 = in.mesh.P800Surf4(:);
nonNaNIndex = ~isnan(in.mesh.P800Surf4);
in.mesh.P800Surf4 = in.mesh.P800Surf4(nonNaNIndex);

in.mesh.P800Surf2 = readmatrix(strcat(F_mesh,"Surf_1_S2.txt"));
in.mesh.P800Surf2 = in.mesh.P800Surf2(:);
nonNaNIndex = ~isnan(in.mesh.P800Surf2);
in.mesh.P800Surf2 = in.mesh.P800Surf2(nonNaNIndex);

%% Get bulge node coordinates
in.bulgeNodeCoordinate = in.mesh.nodes(in.mesh.BulgeNodes,:);
x2eval = in.bulgeNodeCoordinate(:,2);
in.x2eval = x2eval;
%% Gaussian GT
GTGaussian = load(strcat(F_truth,"MatExtractionDABI250818.mat"));
inv_samples = GTGaussian.inv_samplesEXPDABIQEiextract;

%% construct GT based on Gaussian fit on the experimental data
x = x2eval(:);                 % xDim x 1
xDim   = numel(x);
nCells = numel(inv_samples);

Truth = cell(nCells, 1);

for i = 1:nCells
    ts = inv_samples{i}.time_series;   % [t, D_amp, Sd, Sp]
    t    = ts(:,1);                    % tDim x 1
    Damp = ts(:,2).';                  % 1 x tDim
    Sd   = ts(:,3).';                  % 1 x tDim
    Sp   = ts(:,4).';                  % 1 x tDim
    tDim = numel(t);

    % Evaluate D(x,t) on grid (xDim x tDim) using implicit expansion :contentReference[oaicite:1]{index=1}
    D = Damp .* exp( - ( (x ./ Sd) .^ Sp ) );

    % --- Add the t=0 column with D=0 ---
    if isempty(t) || abs(t(1)) > 0   % if time doesn't already start at 0
        t_aug = [0; t(:)];
        D_aug = [zeros(xDim,1), D];
    else
        t_aug = t(:);
        D_aug = D;
    end

    % Build matrix with headers: (xDim+1) x (numel(t_aug)+1)
    M = NaN(xDim+1, numel(t_aug)+1);
    M(1,1)         = 0;           % top-left corner
    M(1,2:end)     = t_aug.';      % time header row
    M(2:end,1)     = in.mesh.BulgeNodes; % x;           % x header column
    M(2:end,2:end) = D_aug;       % values

    Truth{i} = M;
end

%% Get out of bound nodes to remove and interpolate
TruthMod = cell(nCells, 1);

for i = 1:length(Truth)

current = Truth{i};
TimeVec = current(1,2:end);
data = current(2:end,:);

%% columns and rows to remove
if i == 1

resultArray = sortrows(data,1);
in.indicesdata = resultArray;
% Get distance between center and all points
Distance2Center = x2eval;

IDX = Distance2Center>withinRange;
NodeIDX = resultArray(IDX,1);

[~, indices] = ismember(NodeIDX, resultArray(:,1));
end

%%
InputArray = [TimeVec;data(:,2:end)];
tottime = in.PressJump(i,end) * 3600;
ind = find(TimeVec > tottime, 1, 'first');
InputArrayTruc = InputArray(:,1:ind);

res = 72;
in.res = res;
endtime = tottime;
currentTruth = Postprocess_ABAQ_outputs(InputArrayTruc, res, endtime);

TruthMod{i} = currentTruth';

if i == 1
in.zeroColumnIndices = find(all(TruthMod{i} == 0, 1));
in.DiscardedPoints = indices; % Discard those points closed to periphery which usually has bad data due to interpolation
in.AllDiscardedPoints = unique([in.zeroColumnIndices-1,in.DiscardedPoints']);
in.Row2Remove = [];
end

end

in.TruthMod = TruthMod;
%%

% in.Row2Remove = []; % Discard the beginning which may be incorrect due to DIC dataprocessing.

end