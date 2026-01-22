%% Compare Baseline (8 params) vs Variant B (9 params)
% This script helps analyze and compare model performance
%
% Usage:
%   1. Run main_Parameter_Estimation_Chamomile with Variant B
%   2. Save results: save('results_variantB.mat', 'KOUT', 'obj_value', 'cond_number', 'std_errors')
%   3. Switch to baseline model in modelSFE.m and main file
%   4. Run optimization again
%   5. Save results: save('results_baseline.mat', 'KOUT', 'obj_value', 'cond_number', 'std_errors')
%   6. Run this script: compare_models

%% Load Results
fprintf('=== Model Comparison ===\n\n');

if ~exist('results_baseline.mat', 'file')
    error('Baseline results not found. Run optimization and save results first.');
end

if ~exist('results_variantB.mat', 'file')
    error('Variant B results not found. Run optimization and save results first.');
end

baseline = load('results_baseline.mat');
variantB = load('results_variantB.mat');

%% Basic Statistics
Ndata = 156;  % Total observations
Nparams_baseline = 8;
Nparams_variantB = 9;

SSE_baseline = baseline.obj_value;
SSE_variantB = variantB.obj_value;

fprintf('--- Sum of Squared Errors (SSE) ---\n');
fprintf('Baseline (8 params): %.6e\n', SSE_baseline);
fprintf('Variant B (9 params): %.6e\n', SSE_variantB);
fprintf('Improvement: %.2f%%\n\n', 100*(SSE_baseline - SSE_variantB)/SSE_baseline);

%% Information Criteria
% Akaike Information Criterion (AIC)
AIC_baseline = Ndata * log(SSE_baseline/Ndata) + 2*Nparams_baseline;
AIC_variantB = Ndata * log(SSE_variantB/Ndata) + 2*Nparams_variantB;

% Bayesian Information Criterion (BIC)
BIC_baseline = Ndata * log(SSE_baseline/Ndata) + Nparams_baseline*log(Ndata);
BIC_variantB = Ndata * log(SSE_variantB/Ndata) + Nparams_variantB*log(Ndata);

fprintf('--- Information Criteria ---\n');
fprintf('AIC Baseline:  %.2f\n', AIC_baseline);
fprintf('AIC Variant B: %.2f\n', AIC_variantB);
fprintf('ΔAIC:          %.2f ', AIC_variantB - AIC_baseline);
if AIC_variantB - AIC_baseline < -2
    fprintf('(STRONG evidence for Variant B)\n');
elseif AIC_variantB - AIC_baseline < 0
    fprintf('(Weak evidence for Variant B)\n');
elseif AIC_variantB - AIC_baseline < 2
    fprintf('(Models equivalent)\n');
else
    fprintf('(Baseline is better)\n');
end

fprintf('\nBIC Baseline:  %.2f\n', BIC_baseline);
fprintf('BIC Variant B: %.2f\n', BIC_variantB);
fprintf('ΔBIC:          %.2f ', BIC_variantB - BIC_baseline);
if BIC_variantB - BIC_baseline < -6
    fprintf('(VERY STRONG evidence for Variant B)\n');
elseif BIC_variantB - BIC_baseline < 0
    fprintf('(Evidence for Variant B)\n');
elseif BIC_variantB - BIC_baseline < 6
    fprintf('(Models equivalent)\n');
else
    fprintf('(Baseline is better)\n');
end
fprintf('\n');

%% Identifiability (Condition Number)
fprintf('--- Identifiability ---\n');
fprintf('Condition Number Baseline:  %.2e', baseline.cond_number);
if baseline.cond_number < 1e6
    fprintf(' (Excellent)\n');
elseif baseline.cond_number < 1e8
    fprintf(' (Good)\n');
elseif baseline.cond_number < 1e10
    fprintf(' (Acceptable)\n');
else
    fprintf(' (Poor - identifiability issues)\n');
end

fprintf('Condition Number Variant B: %.2e', variantB.cond_number);
if variantB.cond_number < 1e6
    fprintf(' (Excellent)\n');
elseif variantB.cond_number < 1e8
    fprintf(' (Good)\n');
elseif variantB.cond_number < 1e10
    fprintf(' (Acceptable)\n');
else
    fprintf(' (Poor - identifiability issues)\n');
end
fprintf('\n');

%% Parameter Uncertainty
fprintf('--- Parameter Uncertainty (Relative Std Error) ---\n');
fprintf('Baseline (8 params):\n');
rel_std_baseline = baseline.std_errors ./ abs(baseline.KOUT);
for i = 1:Nparams_baseline
    fprintf('  Param %d: %.1f%%', i, 100*rel_std_baseline(i));
    if rel_std_baseline(i) < 0.2
        fprintf(' (Well determined)\n');
    elseif rel_std_baseline(i) < 0.5
        fprintf(' (Acceptable)\n');
    elseif rel_std_baseline(i) < 1.0
        fprintf(' (Uncertain)\n');
    else
        fprintf(' (Poorly determined)\n');
    end
end

fprintf('\nVariant B (9 params):\n');
param_names_B = {'k_max_base', 'K_m_base', 'beta_Km', 'E_a', 'alpha_RE', 'RE_ref', 'rho_ref', 'k_diff', 'n_order'};
rel_std_variantB = variantB.std_errors ./ abs(variantB.KOUT);
for i = 1:Nparams_variantB
    fprintf('  %s: %.1f%%', param_names_B{i}, 100*rel_std_variantB(i));
    if rel_std_variantB(i) < 0.2
        fprintf(' (Well determined)\n');
    elseif rel_std_variantB(i) < 0.5
        fprintf(' (Acceptable)\n');
    elseif rel_std_variantB(i) < 1.0
        fprintf(' (Uncertain)\n');
    else
        fprintf(' (Poorly determined)\n');
    end
end
fprintf('\n');

%% Physical Validation (Variant B only)
fprintf('--- Physical Validation (Variant B) ---\n');

beta_Km = variantB.KOUT(3);
E_a = variantB.KOUT(4);
alpha_RE = variantB.KOUT(5);

fprintf('beta_Km:  %.3f ', beta_Km);
if beta_Km >= 0.5 && beta_Km <= 2.0
    fprintf('(Physical range OK)\n');
else
    fprintf('(Outside expected range 0.5-2.0)\n');
end

fprintf('E_a:      %.1f kJ/mol ', E_a);
if E_a >= 10 && E_a <= 40
    fprintf('(Physical range OK)\n');
else
    fprintf('(Outside expected range 10-40 kJ/mol)\n');
end

fprintf('alpha_RE: %.3f ', alpha_RE);
if alpha_RE >= 0.4 && alpha_RE <= 0.6
    fprintf('(Physical range OK - Sherwood theory)\n');
elseif alpha_RE >= 0.3 && alpha_RE <= 0.8
    fprintf('(Acceptable range)\n');
else
    fprintf('(Outside expected range 0.3-0.8)\n');
end
fprintf('\n');

%% Recommendation
fprintf('=== RECOMMENDATION ===\n');

% Count decision factors
favor_variantB = 0;
favor_baseline = 0;

% AIC comparison
if AIC_variantB - AIC_baseline < -2
    favor_variantB = favor_variantB + 2;
elseif AIC_variantB - AIC_baseline > 2
    favor_baseline = favor_baseline + 1;
end

% BIC comparison
if BIC_variantB - BIC_baseline < -6
    favor_variantB = favor_variantB + 2;
elseif BIC_variantB - BIC_baseline > 6
    favor_baseline = favor_baseline + 1;
end

% Condition number
if variantB.cond_number < baseline.cond_number && variantB.cond_number < 1e10
    favor_variantB = favor_variantB + 1;
elseif variantB.cond_number > 1e10
    favor_baseline = favor_baseline + 2;
end

% Parameter uncertainty (check if new params are well determined)
if rel_std_variantB(3) < 0.5 && rel_std_variantB(4) < 0.5  % beta_Km and E_a
    favor_variantB = favor_variantB + 1;
elseif rel_std_variantB(3) > 1.0 || rel_std_variantB(4) > 1.0
    favor_baseline = favor_baseline + 2;
end

% Physical ranges
physical_ok = (beta_Km >= 0.5 && beta_Km <= 2.0) && ...
              (E_a >= 10 && E_a <= 40) && ...
              (alpha_RE >= 0.3 && alpha_RE <= 0.8);
if physical_ok
    favor_variantB = favor_variantB + 1;
else
    favor_baseline = favor_baseline + 1;
end

fprintf('Score: Variant B = %d, Baseline = %d\n\n', favor_variantB, favor_baseline);

if favor_variantB > favor_baseline + 2
    fprintf('✅ RECOMMENDATION: Adopt Variant B\n');
    fprintf('   - Better fit and/or identifiability\n');
    fprintf('   - Physically meaningful parameters\n');
    fprintf('   - New parameters (beta_Km, E_a) are well determined\n');
elseif favor_variantB > favor_baseline
    fprintf('⚡ RECOMMENDATION: Consider Variant B\n');
    fprintf('   - Slight improvement over baseline\n');
    fprintf('   - Verify physical trends before final decision\n');
elseif favor_baseline > favor_variantB
    fprintf('❌ RECOMMENDATION: Keep Baseline\n');
    fprintf('   - Variant B does not provide sufficient improvement\n');
    fprintf('   - Or identifiability issues detected\n');
    fprintf('   - Consider trying Variant C (K_m only) instead\n');
else
    fprintf('⚖️  RECOMMENDATION: Models are equivalent\n');
    fprintf('   - Choose based on preference:\n');
    fprintf('     • Variant B: More physical interpretation\n');
    fprintf('     • Baseline: Simpler, fewer parameters\n');
end

fprintf('\n');

%% Visualization Suggestion
fprintf('=== Next Steps ===\n');
fprintf('1. Check correlation matrix for new parameters\n');
fprintf('2. Plot K_m(ρ) and k_max(T,RE) vs experimental conditions\n');
fprintf('3. Validate monotonic trends:\n');
fprintf('   - K_m should DECREASE with increasing ρ\n');
fprintf('   - k_max should INCREASE with increasing T and RE\n');
fprintf('4. If all trends are physical, adopt Variant B\n');
fprintf('5. If E_a is poorly determined, try Variant C (K_m only)\n');
