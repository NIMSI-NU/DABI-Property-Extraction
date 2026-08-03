function T_Disp=Read_ABAQ_outputs(fileDAT,fileSTA, endtime)
% A function to read both STA and DAT files
D_sim = [];
NodeLabel = [];

if exist(fileSTA, 'file') == 2

STATUS = readmatrix(fileSTA,'FileType','text');
STATUS(any(isnan(STATUS),2),:) = []; 

if isempty(STATUS)
    T_Disp = "false";
else
    Time = STATUS(1:end,8)'; % TimeInc = STATUS(1:end,9)';
    if isempty(Time)
        T_Disp = "false";
    elseif Time(end) < endtime-1.0
        T_Disp = "false";
    else
        Time = []; % Clear the previous
        % pattern = 'STEP TIME COMPLETED\s+[\d\.E\-]+\s*,\s*TOTAL TIME COMPLETED\s+([\d\.E\-]+)';
        fidd = fopen(fileDAT);
        numSteps = 0;
        while (~feof(fidd))
            tline = fgetl(fidd);
            % j=0;
            D_sim_tmp = [];
        
            % Get the time from DAT file instead of STA file
            if contains(tline, 'STEP TIME COMPLETED')
            % Find the last number in the line
                pattern = '[-+]?[0-9]*\.?[0-9]+([eE][-+]?[0-9]+)?';
                numbers = regexp(tline, pattern, 'match');
                
                if ~isempty(numbers)
                    % Convert the last number to a double
                    last_number_str = numbers{end};
                    Timetmp = str2double(last_number_str);
                    Time = [Time,Timetmp];
                end
            end
            
            % For elements, replace 'N O D E   O U T P U T'  by 'E L E M E N T   O U T P U T'
            if (regexpi(tline, 'N O D E   O U T P U T')>0)
                
                numSteps = numSteps + 1;
                tline = fgetl(fidd);
                
                while(isempty(str2num(tline)))
                    tline = fgetl(fidd);
                end
                while(~isempty(str2num(tline)))
                    % % j=j+1;
                    % data_f = sscanf(tline, '%d %e %e', [1,3]);
                    data_f = sscanf(tline, '%d %e', [1,2]);
                    if numSteps == 1
                        node_number=data_f(1);
                        NodeLabel=[NodeLabel;node_number];
                    end
                    D_sim_tmp=[D_sim_tmp;data_f(2)];
                    tline = fgetl(fidd);
                end
                D_sim=[D_sim,D_sim_tmp];
            end
    
        end
        fclose(fidd);
        
        if Time(end) < endtime-1.0
            T_Disp = "false";
        else
            T_Disp = [Time;D_sim];
        end
    end
end
else
    T_Disp = "false";
end
