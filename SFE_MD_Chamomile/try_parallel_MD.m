startup;
delete(gcp('nocreate'));

%%
P       = 100;
N_exp   = 4;
N_core  = 4;

MD(P, N_exp, N_core);

%%
% Set time of the simulation
PreparationTime         = 0;
ExtractionTime          = 600;
timeStep                = 5;                                                % Minutes
OP_change_Time          = 10; 
Sample_Time             = 5;
    
simulationTime          = PreparationTime + ExtractionTime;
    
timeStep_in_sec         = timeStep * 60;                                    % Seconds
Time_in_sec             = (timeStep:timeStep:simulationTime)*60;            % Seconds
Time                    = [0 Time_in_sec/60];                               % Minutes

N_Time                  = length(Time_in_sec);

SAMPLE                  = Sample_Time:Sample_Time:ExtractionTime;

OP_change               = OP_change_Time:OP_change_Time:ExtractionTime;

% Check if the number of data points is the same for both the dataset and the simulation
N_Sample                = [];
for i = 1:numel(SAMPLE)
    N_Sample            = [N_Sample ; find(round(Time,3) == round(SAMPLE(i))) ];
end
if numel(N_Sample) ~= numel(SAMPLE)
    keyboard
end

