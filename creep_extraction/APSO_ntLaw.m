% This script is to use nt_law and APSO to run creep identifications based
% on 12 DABI experiments
clear all
close all
clc

%% ---------------------------  Mesh & Paths  ---------------------------
F_mesh  = '...\DABI_mesh\';
F_truth = '...\GroundTruth\';
in      = opt_input(F_mesh, F_truth);
in.saveDir = '...\Simulations\';
if ~exist(in.saveDir, 'dir'); mkdir(in.saveDir); end

%% Lower and upper bounds, normalized to [0,1]
% log(A), n0, ninf, Abump, taur, taud, Qe5
lb = [1, 1.0, 3.0, 0, 1000, 2000, 2];
ub = [10, 7.0, 7.0, 3.5, 10000, 15000, 6];
lblog = [0, 0, 0, 0, 0, 0, 0];
ublog = [1, 1, 1, 1, 1, 1, 1];

in.LBori = lb;
in.UBori = ub;
in.UB = ublog;
in.LB = lblog;

%% Run with some solution
fun_vec = def_objective_vec(in,5000);
Sol = [6.2956	5.1436	5.1592	2.37895	6546.1	10347.3	4.7284];
%Sol = [5.624, 4.434, 4.30126, 1.29426, 4753.9359, 7818.379, 4.22572];
SolLog = [(Sol(1)-lb(1))/(ub(1)-lb(1)),...
    (Sol(2)-lb(2))/(ub(2)-lb(2)),...
    (Sol(3)-lb(3))/(ub(3)-lb(3)),...
    (Sol(4)-lb(4))/(ub(4)-lb(4)),...
    (Sol(5)-lb(5))/(ub(5)-lb(5)),...
    (Sol(6)-lb(6))/(ub(6)-lb(6)),...
    (Sol(7)-lb(7))/(ub(7)-lb(7))];

tic;
[fitness, evalTime] = fun_vec(SolLog)
toc

%% Problem Definition
% Ground truth creep parameters
% IN718: Creep strain and creep-life prediction for alloy 718 using the omega method
% 2-coefficient power law: A,n,Eact
numParticles = 70;
numVariables = 7;
maxIterations = 1000;
numCPUs = 70;

maxFuncEvalsPerParticle = maxIterations; % Maximum function evaluations per particle
evalThreshold = 1;                  % Display information after every evalThreshold evaluations
Fraction = 1;
targetFraction = Fraction * numParticles;  % 80% of particles reaches maxIterations evaluation will end the process
maxGlobalEvals = 0;                   % Total number of global evaluations

waittime = 1200;
AsynchWaitTime = 0;

% Usually c1+c2 ~ 4, take c1+c2=4.1
% Make the same
c1 = 2.05; % Cognitive (personal) coefficient
c2 = 2.05; % Social (global) coefficient

% Constriction factor method (method adopted)
% Ref1: Comparing Inertia Weights and Constriction Factors in Particle Swarm Optimization
% Ref2: The Particle Swarm: Explosion, Stability, and Convergence in a Multidimensional Complex Space
% Constriction factor K is for phi > 4
phi = c1 + c2;
K = 2/abs(2-phi-sqrt(phi^2-4*phi));

% Bounds for velocity
% Avoid explosion at the beginning which results in meaningless search
vBoundsScaleFactor = 0.15;
v_max = abs(vBoundsScaleFactor*(ublog-lblog)); % the lower bound is -v_max

%% Initial guess generation
% Initialize Sobol sequence generators
positionSobol = sobolset(numVariables, 'Skip', 1e3); % For initial positions

% Generate initial positions using Sobol sequence
positions = net(positionSobol, numParticles*100); % Generate Sobol sequence for positions
indices = randperm(length(positions), numParticles); % Get random indices
positions = positions(indices,:);

% Transform the positions to the bounds
for i = 1:numParticles
    for j = 1:numVariables
        positions(i, j) = lblog(j) + ...
            (ublog(j) - lblog(j)) * positions(i, j);
    end
end

% Pre-generation of r1 and r2
cognitiveSobol = sobolset(1, 'Skip', 3e3); % For cognitive coefficients
cognitiveRand = net(cognitiveSobol, numParticles * maxIterations*100);
indices1 = randperm(length(cognitiveRand), numParticles*maxIterations);
cognitiveRand = cognitiveRand(indices1);
cognitiveRand = cognitiveRand(:);

socialSobol = sobolset(1, 'Skip', 4e3); % For social coefficients
socialRand = net(socialSobol, numParticles * maxIterations*100);
indices2 = randperm(length(socialRand), numParticles*maxIterations);
socialRand = socialRand(indices2);
% socialRand = reshape(socialRand,[numParticles,maxIterations]);
socialRand = socialRand(:);

% Regarding the discussion of limiting velocity
% https://www.scielo.org.mx/scielo.php?script=sci_arttext&pid=S1405-55462016000400635#B11
% Ref: Particle Swarm Optimization: Velocity Initialization
% "This leads to the main disadvantages of random initialization, namely much slower
% convergence and much more wasted effort searching outside the bounds of the optimization
% problem. The conclusion is therefore that random initialization should be avoided in favor
% of small random or zero initialization. Note that the latter two strategies 
% exhibited similar behavior."
% Zero initial velocity is adopted
velocities = zeros([numParticles,numVariables]);

%% Asynchronic PSO main loop through iterations
%% Initiation of storages
% Initialize particles (positions, velocities, personal bests, etc.)
personalBestPositions = positions;
personalBestScores = inf(numParticles, 1);
globalBestPosition = rand(1, numVariables);
globalBestScore = inf;

% Initialize counters and data storage
evalCounts = zeros(numParticles, 1);  % Track the number of evaluations per particle
evalTimes = cell(numParticles, 1);  % Cell array to store time for each evaluation
allCosts = cell(numParticles, 1);     % Store all costs per particle
allSolutions = cell(numParticles, 1); % Store all positions per particle

%%
EachEvalminCost = [];
EachEvalElapsedTime = [];
EachEvalminCostPos = [];

%%
% Elapsed time and evaluation count tracking for specific thresholds
thresholds = [0.1, 0.01, 0.005, 0.0025, 0.001, 1e-4, 1e-6];
thresholdTimes = inf * ones(1, length(thresholds));
thresholdEvalCounts = inf * ones(1, length(thresholds));

%%
startTime = tic;  % Start timing the entire process

%% Initiation evaluation
% Open a parallel pool with numCPUs workers
parpool("Processes",numCPUs);

%%
% Initial evaluations using parfeval (submit jobs asynchronously)
disp('Submitting initial evaluations to the background...');
futures = parallel.FevalFuture.empty(numParticles, 0); % Store future objects

for i = 1:numParticles
    % Submit each particle's evaluation job to the background
    func_obj = def_objective_vec(in,i);
    futures(i) = parfeval(func_obj, 2, positions(i, :));
end

disp('Initial guesses evaluations: Waiting for some time before checking progress...');
pause(waittime);

%%
numFinished = 0;
for i = 1:numParticles
    if strcmp(futures(i).State, 'finished')
        numFinished = numFinished + 1;
    end
end
fprintf('%d out of %d initial evaluations have finished after wait time.\n', numFinished, numParticles);

% Fetch results of the finished evaluations
for i = 1:numParticles
    if strcmp(futures(i).State, 'finished')
        
        [score, evalTime] = fetchOutputs(futures(i));  % Get the result        
        % Set personal best scores
        personalBestScores(i) = score;
        personalBestPositions(i, :) = positions(i, :);
        
        % Update global best
        if score < globalBestScore
            globalBestScore = score;
            globalBestPosition = positions(i, :);
        end
    end
end

fprintf('Best fitness = %.6f' , globalBestScore);

save('DABIntLaw_RMSRE_2D_init.mat');

%% Asynchronous particle evaluation using parfeval
startimeofAsynchLoop = toc(startTime)
%%
% Main loop with stopping criteria and asynchronous evaluation
% while maxGlobalEvals < numParticles * maxFuncEvalsPerParticle
while nnz(evalCounts >= maxFuncEvalsPerParticle) < targetFraction

    % Fetch results as they become available
    for i = 1:numParticles
        if strcmp(futures(i).State, 'finished')
            if evalCounts(i) < maxFuncEvalsPerParticle

            [score, evalTime] = fetchOutputs(futures(i));  % Get the result

            evalTimes{i} = [evalTimes{i}; evalTime];
            evalCounts(i) = evalCounts(i) + 1;  % Increment the particle's evaluation count
            maxGlobalEvals = maxGlobalEvals + 1; % Increment total evaluations
            
            % Store cost and solution
            allCosts{i} = [allCosts{i}; score];
            allSolutions{i} = [allSolutions{i}; positions(i, :)];
            
            % Update personal bests
            if score < personalBestScores(i)
                personalBestScores(i) = score;
                personalBestPositions(i, :) = positions(i, :);
            end
            
            % Update global best
            if score < globalBestScore
                globalBestScore = score;
                globalBestPosition = positions(i, :);
                
                % Check for threshold crossing, log time and evaluation count
                elapsedTime = toc(startTime);
                for tIdx = 1:length(thresholds)
                    if globalBestScore <= thresholds(tIdx) && isinf(thresholdTimes(tIdx))
                        thresholdTimes(tIdx) = elapsedTime;
                        thresholdEvalCounts(tIdx) = maxGlobalEvals;
                        fprintf('Best cost <= %.6f at time %.2f seconds, after %d evaluations.\n', ...
                                thresholds(tIdx), elapsedTime, maxGlobalEvals);
                    end
                end
            end

            % Display best fitness so far after certain evaluations
            if mod(maxGlobalEvals, evalThreshold) == 0
                elapsedTime = toc(startTime);
                EachEvalminCost = [EachEvalminCost;globalBestScore];
                EachEvalElapsedTime = [EachEvalElapsedTime;elapsedTime];
                EachEvalminCostPos = [EachEvalminCostPos;globalBestPosition(:)];

                fprintf('After %d evaluations and %.2f time: Best fitness = %.6f, Particle = %d, Iteration = %d\n', ...
                    maxGlobalEvals, elapsedTime, globalBestScore, i, evalCounts(i));
            end
            if mod(maxGlobalEvals, 30) == 0
                disp(globalBestPosition);
            end

            % Update particle position and velocity
            shuffledcognitiveRand = cognitiveRand(randperm(length(cognitiveRand)));
            shuffledsocialRand = socialRand(randperm(length(socialRand)));
            randomIndex = randi(length(shuffledcognitiveRand));  % Random index

            velocities(i, :) = K * (velocities(i, :) + ...
                       c1 * shuffledcognitiveRand(randomIndex) * (personalBestPositions(i, :) - positions(i, :)) + ...
                       c2 * shuffledsocialRand(randomIndex) * (globalBestPosition - positions(i, :)));
            
            % Check velocity bounds
            velocities(i, :) = max(min(velocities(i, :), v_max), -v_max);
            
            % Update positions
            positions(i, :) = positions(i, :) + velocities(i, :);
            
            % Check position bounds
            positions(i, :) = max(min(positions(i, :), ublog), lblog);

            end
        end
    end
    
    % Launch asynchronous evaluations for each updated particle
    for i = 1:numParticles
        if isempty(futures(i)) || strcmp(futures(i).State, 'finished') % Check available worker
            if evalCounts(i) < maxFuncEvalsPerParticle
                func_obj = def_objective_vec(in,i);
                futures(i) = parfeval(func_obj, 2, positions(i, :));
            end
        end
    end

    % saving
    save('APSOloop.mat','-v7');

    % Check stopping criteria
    if globalBestScore < 1e-6
        fprintf('Stopping: Best fitness reached %.6f at %d evaluations\n', globalBestScore, maxGlobalEvals);
        break;
    end
    
    % Check if 80% of particles have reached the max evaluations
    if nnz(evalCounts >= maxFuncEvalsPerParticle) >= targetFraction
        fprintf('Stopping: %f of particles have reached %d evaluations\n', Fraction, maxFuncEvalsPerParticle);
        break;
    end

    pause(AsynchWaitTime);

end

%% Shutdown parallel pool
delete(gcp);
