clc, close all
clear all

%addpath('C:\dev\casadi-windows-matlabR2016a-v3.5.2');
addpath('\\home.org.aalto.fi\sliczno1\data\Documents\casadi-windows-matlabR2016a-v3.5.1');
import casadi.*

Parameters_table        = readtable('Parameters.csv') ;        % Table with prameters
Parameters_cell         = table2cell(Parameters_table(:,3));

%% introduce Casadi variables

P = 80;

h=[];
for T=[linspace(40,80)]+273
    Z           = Compressibility( T, P,         Parameters_cell );
    rho         = rhoPB_Comp(      T, P, Z,      Parameters_cell );
    h           =[h, SpecificEnthalpy(T, P, Z, rho, Parameters_cell )];
end

T_s             = MX.sym('T_s',numel(h),1);

Z               = Compressibility( T_s, P,         Parameters_cell );
rho             = rhoPB_Comp(      T_s, P, Z,      Parameters_cell );
h_sym           = SpecificEnthalpy(T_s, P, Z, rho, Parameters_cell );

H               = h' - h_sym;

g = Function('g',{T_s},{H});
G = rootfinder('G','newton',g);
disp(G)

tic
( G(40+273)-273 - linspace(40,80)' )
toc