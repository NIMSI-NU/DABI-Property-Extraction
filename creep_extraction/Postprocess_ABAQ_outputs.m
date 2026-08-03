function T_Disp_m=Postprocess_ABAQ_outputs(T_Disp,res,endtime)
%%
% A function to postprocess data obtained from abaqus via sorting, interpolation
% and select the location points for optimization
num_points_new = res; % This should be 1 more than the frequency specified in the abaqus simulation

Time_original = T_Disp(1,:);
%T_new = linspace(0, T_Disp(1,end), num_points_new);
T_new = linspace(0, endtime, num_points_new);
% T_new = T_Disp(1,:);
T_Disp_m = T_new;
for i = 1:size(T_Disp(2:end,:),1)
    Disp_orig = T_Disp(i+1,:);
    [uniqueX, ~, idx] = unique(Time_original, 'stable');
    uniqueData = zeros(2,length(uniqueX));
    for j = 1:length(uniqueX)
        correspondingY = Disp_orig(idx' == j);
        uniqueData(:, j) = [uniqueX(j);correspondingY(end)];
    end
    Disp_new = interp1(uniqueData(1,:), uniqueData(2,:), T_new, 'linear');
    T_Disp_m = [T_Disp_m;Disp_new];
end
T_Disp_m(:,1) = 0;

end
