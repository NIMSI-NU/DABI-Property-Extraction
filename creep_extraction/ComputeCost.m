function C = ComputeCost(JobName, in, itemp)
%%
    fileDAT = strcat(JobName,'.dat');
    fileSTA = strcat(JobName,'.sta');
    endtime = in.PressJump(itemp,end)*3600;

    T_Disp=Read_ABAQ_outputs(fileDAT,fileSTA,endtime);

    %%
    if isstring(T_Disp)
        C = inf;
    else
        %%
        T_Disp_m=Postprocess_ABAQ_outputs(T_Disp,in.res,endtime);
        %%
        T_Disp_m = T_Disp_m';
        %%
        T = in.TruthMod{itemp};
        T = T(2:end,2:end);
        P = T_Disp_m(2:end,2:end);

        %%
        Columns2Calculate = true(1,size(T,2));
        Columns2Calculate(in.AllDiscardedPoints) = false;
        Row2Calculate = true(1,size(T,1));
        Row2Calculate(in.Row2Remove) = false;

        %% compute cost
        % Extract the subset of data based on row/column filters
        T_subset = T(Row2Calculate, Columns2Calculate);
        P_subset = P(Row2Calculate, Columns2Calculate);

        % Create a mask to identify values where abs(T) is large enough
        % This returns logical 1 (true) for values >= a specified value,
        % and 0 for outliers, remove extremely small displacements in the
        % identifications if needed.
        validElementsMask = abs(T_subset) >= 0.005;

        % Apply the mask to extract only the valid elements
        T_final = T_subset(validElementsMask);
        P_final = P_subset(validElementsMask);

        % RMSRE
        C = (P_final - T_final) ./ (T_final + 1e-6);
        N_s = size(T_subset, 1);
        C = sqrt(sum(C(:).^2)) / N_s;
        % C = sqrt(sum(C(:).^2)) / length(C);
        
    end
end