function writeAbaqusInputFile(InFileName, saveDir, mesh, Pressure, MODULUS, DURATION)

fname = strcat(saveDir,InFileName);
fid = fopen(fname, 'w');

% Write Heading
fprintf(fid, '*Heading\n');
fprintf(fid, '** Job name=HTDABIdimple\n');
fprintf(fid, '** PARTS\n');

%% 
fprintf(fid, '*Part, name=Dimple\n');
fprintf(fid, '*Node\n');

% Write Nodes
for ino = 1:size(mesh.nodes, 1)
    fprintf(fid, '    %d, %.6f, %.6f\n', mesh.nodes(ino, 1), mesh.nodes(ino, 2), mesh.nodes(ino, 3));
end
% Write Elements
fprintf(fid, '*Element, type=CAX4R\n');
% writematrix(mesh.elem,fname,'WriteMode','append','FileType','text');
for iel = 1:size(mesh.elem, 1)
    fprintf(fid, '    %d, %d, %d, %d, %d\n', mesh.elem(iel, :));
end

max_indices_per_line = 16;

% Generate all nodes and elements set
fprintf(fid, '*Nset, nset=AllNodes, generate\n');
fprintf(fid, '%4d, %5d, %5d\n',     1,  length(mesh.nodes),     1);
fprintf(fid, '*Elset, elset=AllElem, generate\n');
fprintf(fid, '%4d, %5d, %5d\n',     1,  length(mesh.elem),     1);

fprintf(fid, '*Solid Section, elset=AllElem, material=ElasticCreep\n');
fprintf(fid, ',\n');

fprintf(fid, '*End Part\n');
fprintf(fid, '**\n');

%% Assembly
fprintf(fid, '*Assembly, name=Assembly\n');
fprintf(fid, '*Instance, name=Dimple-1, part=Dimple\n');
fprintf(fid, '*End Instance\n');
fprintf(fid, '**\n');

%% Other Node and Element Sets

% Bulge nodes
fprintf(fid, '*Nset, nset=BulgeNodes, instance=Dimple-1\n');
write_indices(fid, mesh.BulgeNodes, max_indices_per_line);

% Fixed node set
fprintf(fid, '*Nset, nset=FixedBC, instance=Dimple-1\n');
write_indices(fid, mesh.FixedBCset, max_indices_per_line);

% temperature
% fprintf(fid, '*Nset, nset=AllNodesTemp, instance=Plate2D-1, generate\n');
% fprintf(fid, '%4d, %5d, %5d\n',     1,  length(mesh.nodes),     1);
% *Elset, elset=Set-3, instance=Plate2D-1, generate
%     1,  1290,     1

fprintf(fid, '*Elset, elset=_Surf-1_S1, internal, instance=Dimple-1\n');
write_indices(fid, mesh.P800Surf1, max_indices_per_line);
fprintf(fid, '*Elset, elset=_Surf-1_S2, internal, instance=Dimple-1\n');
write_indices(fid, mesh.P800Surf2, max_indices_per_line);
fprintf(fid, '*Elset, elset=_Surf-1_S3, internal, instance=Dimple-1\n');
write_indices(fid, mesh.P800Surf3, max_indices_per_line);
fprintf(fid, '*Elset, elset=_Surf-1_S4, internal, instance=Dimple-1\n');
write_indices(fid, mesh.P800Surf4, max_indices_per_line);

fprintf(fid, '*Surface, type=ELEMENT, name=Surf-1\n');
fprintf(fid, '_Surf-1_S1, S1\n');
fprintf(fid, '_Surf-1_S2, S2\n');
fprintf(fid, '_Surf-1_S4, S4\n');
fprintf(fid, '_Surf-1_S3, S3\n');

fprintf(fid, '*End Assembly\n');

% Loading amplitude
fprintf(fid, '*Amplitude, name=Amp-1\n');
% fprintf(fid, '%4d, %4d, %4d, %4d, %4d, %4d, %4d, %4d, %4d, %4d\n', ...
%     0, 0, ...
%     5, Pressure(2), ...
%     Pressure(1)*3600, Pressure(2), ...
%     Pressure(1)*3600 + 5, Pressure(3), ...
%     12*3600, Pressure(3));
Amplitude = [0,0,...
    5,Pressure(1), ...
    Pressure(3)*3600, Pressure(1), ...
    Pressure(3)*3600 + 5, Pressure(2), ...
    DURATION*3600, Pressure(2)];

write_Amp(fid, Amplitude);
% write_Amp(fid, Config.Amplitude);

% Material
fprintf(fid, '*Material, name=ElasticCreep\n');
fprintf(fid, '*Creep, law=USER\n');
fprintf(fid, '*Elastic\n');
fprintf(fid, '%4d, %5d\n',MODULUS,0.3);

%% Step
fprintf(fid, '*Step, name=Creep, nlgeom=YES, inc=10000000\n');
fprintf(fid, '*Visco, cetol=0.001\n');
fprintf(fid, '%5d, %5d, %5d, %5d\n', 0.1, DURATION*3600, 0.01, DURATION*3600);

% Boundary conditions
fprintf(fid, '*Boundary\n');
fprintf(fid, 'FixedBC, ENCASTRE\n');
fprintf(fid, '*Dsload, amplitude=Amp-1\n');
fprintf(fid, 'Surf-1, P, %f\n', 1.0);

%% Output
fprintf(fid, '*Output, field, variable=PRESELECT, number interval=72\n');
fprintf(fid, '*Output, history, variable=PRESELECT, number interval=72\n');
fprintf(fid, '*Node print, nset=BulgeNodes\n');
fprintf(fid, 'U2,\n');

fprintf(fid, '*End Step');

fclose(fid);

%%
% Function to write indices with a limit per line
function write_indices(fid, indices, max_per_line)
    num_indices = length(indices);
    for i = 1:max_per_line:num_indices
        if i + max_per_line - 1 <= num_indices
            fprintf(fid, '%d, ', indices(i:i + max_per_line - 1));
        else
            fprintf(fid, '%d, ', indices(i:end));
        end
        fprintf(fid, '\n');
    end
end

% Function to write indices with a limit per line
function write_Amp(fid, Amp)
    max_per_line_Amp = 8;
    num_Amp = length(Amp);
    for i = 1:max_per_line_Amp:num_Amp
        if i + max_per_line_Amp - 1 <= num_Amp
            fprintf(fid, '%d, ', Amp(i:i + max_per_line_Amp - 1));
        else
            fprintf(fid, '%d, ', Amp(i:end));
        end
        fprintf(fid, '\n');
    end
end

end