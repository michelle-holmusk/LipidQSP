import pandas as pd
import numpy as np
import scipy
import os, sys, time, json, math
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from os.path import join
from datetime import datetime
from scipy.integrate import odeint
from numpy import loadtxt
from scipy.optimize import minimize
rootDir = os.path.abspath(os.path.curdir)
print(rootDir)
sys.path.insert(0, os.path.join(rootDir, 'lib'))
## use JD's optimizer
#from systemSolver import optimizer as optimizer
from optimizer import Differential_Evolution
from getPatientData import getPatientData

import copy
from matplotlib.font_manager import FontProperties

#riskConfig = json.load(open('amgen-risk-model/amgen-risk-model/config/riskModel_3y_LipidCxoOptimized.json'))
# classConfig = json.load(open(riskConfig['patientClassConfig']))
classConfig = json.load(open('../config/lipidoptimizing.json'))

def differentialequations(I, t, p):
    '''
    This function has the differential equations of the lipids (LDL,
    Total Cholesterol, Triglyceride, HDL)
    
    Inputs:
    
    I: Initial conditions
    t: timepoints
    p: parameters
    
    '''
    try:
        # Initial conditions
        Cldl, Cchol, Ctrig, Chdl = I
        
        # Parameters
        adherence, dose, Imaxldl, Imaxchol, Imaxtrig, Imaxhdl, Ic50, n, dx, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl = p
        
        t = np.round(t)
        t = t.astype(int)
        # print dose.shape  
        if t > (dose.shape[0] - 1):
            t = (dose.shape[0] - 1)
        
        div = (Ic50+(dose[t]*adherence[t])**n)
        h0 = ((dose[t] * adherence[t])**n)
        
        # llipid equation
        dCldldt = (Sx0ldl * (1 - np.sum((Imaxldl*h0)/div))) - (dx*Cldl)
        dCcholdt = (Sx0chol * (1 - np.sum((Imaxchol*h0)/div))) - (dx*Cchol)
        dCtrigdt = (Sx0trig * (1 - np.sum((Imaxtrig*h0)/div))) - (dx*Ctrig)
        dChdldt = (Sx0hdl * (1 + np.sum((Imaxhdl*h0)/div))) - (dx*Chdl)
        f = [dCldldt, dCcholdt, dCtrigdt, dChdldt]
        
        return f
    except Exception as e:
        # print 'There was some problem with the differentialequations function: {}'.format(e)
        print(dose.shape, t)
        raise

def differential_solve(adherence, t, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0, dose):

    '''
    This function solves the differential equations with odeint
    
    Inputs:

    adherence: patient's adherence for all the statins, 2-d numpy array
    t: timepoints
    Sx0: synthesis terms for all the lipids
    C0: baseline values for all the lipids
    dose: doses for all the statins, 2-d numpy array
    '''
    try:

        dx = math.log(2)/14

        ldl_eff  = np.load('../data/final/Efficacy/ldl_efficacy.npy')
        chol_eff = np.load('../data/final/Efficacy/tc_efficacy.npy')
        trig_eff = np.load('../data/final/Efficacy/trig_efficacy.npy')
        hdl_eff  = np.load('../data/final/Efficacy/hdl_efficacy.npy')

        Imaxldl = ldl_eff[0]
        Imaxchol = chol_eff[0]
        Imaxtrig = trig_eff[0]
        Imaxhdl = hdl_eff[0]

        # Imaxldl, Imaxchol, Imaxtrig, Imaxhdl = np.array([0,0,0,0,0,0]), np.array([0,0,0,0,0,0]), np.array([0,0,0,0,0,0]), np.array([0,0,0,0,0,0])
        Ic50 = ldl_eff[1]
        n = 0.7


        I0 = [Cldl0, Cchol0, Ctrig0, Chdl0]
        p = [adherence, dose, Imaxldl, Imaxchol, Imaxtrig, Imaxhdl, Ic50, n, dx, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl]

        sol = odeint(differentialequations, I0, t, args = (p,))
        # print(sol)
        Cldl = []
        Cchol = []
        Ctrig = []
        Chdl = []
        for s1 in sol:
            Cldl.append(s1[0])
            Cchol.append(s1[1])
            Ctrig.append(s1[2])
            Chdl.append(s1[3])
        # print(Cldl)

        return Cldl, Cchol, Ctrig, Chdl
    except Exception as e:
        # print('There was some problem with the differential_solve function: {}'.format(e))
        raise

def adherence_coding(adherence, periods):

    ''' This function takes the adherence and identifies where it is -1 and returns the pairs 
    of rows and columns, number of windows and the flag
    
    
    
    Parameters
    ----------
    adhrenece : {2-d numpy array for each patient}
        It has the adherence values for all the medications for each day
    periods_total : {1-d numpy array}
        It has the 
    
    Returns
    -------
    [type]
        [description]
    '''
    try:
    
        # print(periods_total)
        period_nonzero = periods[periods!=0]


        row, col = np.where(adherence==-1)
        pairs = list(map(list, zip(row, col)))

        windows = len(np.where(np.roll(period_nonzero,1)!=period_nonzero)[0])

        if windows == 0:
            windows = 1
        else:
            windows = windows

        return pairs, windows, period_nonzero

    except Exception as e:
        print('There was some problem with the adherence_coding function: {}'.format(e))

def adherence_guess(adherence, pairs, values, flag):
    try:

        for i in range(len(flag)):
            l = pairs[i]
            adherence[l[0]][l[1]] = values[flag[i]-1]

        return adherence

    except Exception as e:
        # print 'There was some problem with the adherence_guess function: {}'.format(e)
        raise

def h0_cal(dose, Imax, Ic50, n, adherence):
    try:


        h0 = (Imax*((dose*adherence)**n))/(Ic50 + ((dose*adherence)**n))

        if all(np.isnan(h0)):
            h0[:] = 0


        h0_dictionary = {'Atorvastatin':h0[0], 'Fluvastatin':h0[1], 'Lovastatin':h0[2], 
                             'Pravastatin':h0[3], 'Rosuvastatin':h0[4], 'Simvastatin':h0[5]}
        # print(h0_dictionary)

        return h0_dictionary


    except Exception as e:
        print('There was some problem with the h0_cal function: {}'.format(e))

def rmse_function(real_data,real_time,max_value, t, ode_solution):
    try:
        real_time = np.array(real_time)
        weight = (1/max_value)**2
        indices = []
        for j in real_time:
            k = np.where(t == j)[0][0]
            # print(k)
            indices.append(k)
    
        ode_final_values = np.array(ode_solution)[indices]
        # print(indices)
        # quit()
        # print(ode_final_values)
        rmse = np.average(weight*((ode_final_values - np.array(real_data))**2))
        return rmse
    
    except Exception as e:
        print('There was some problem with the rmse_function function: {}'.format(e))

def get_total_rmse_nonNorm(adherence, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0, dose, t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl,t):
    try:

        ldl_max = max(ldl)
        tc_max = max(tc)
        # if len(trig)>0:
        # 	trig_max = max(trig)
        # else:
        # 	trig_max = 1
        trig_max = 1 # max(trig)
        hdl_max = 1 # max(hdl)
        
        Cldl, Cchol, Ctrig, Chdl = differential_solve(adherence, t, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0, dose)
        
        rmse_ldl = rmse_function(ldl, t_ldl, 1, t, Cldl)
        rmse_tc = rmse_function(tc, t_tc, 1, t, Cchol)
        # rmse_trig = rmse_function(trig, t_trig, trig_max, t, Ctrig)
        rmse_trig = 0
        rmse_hdl = 0 #rmse_function(hdl, t_hdl, 1, t, Chdl)
        
        rmse_total = rmse_ldl + rmse_tc + (rmse_trig * 0) + rmse_hdl
        
        return rmse_total

    except Exception as e:
        # print 'There was some problem with the get_total_rmse function: {}'.format(e)
        raise

def get_total_rmse(x, pairs, windows, period_nonzero, adherence, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0, dose, t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl,t,count, biomarker,pre_adherence,prestatin, statintype, statin_dose):
    try:
        values_adherence = x[0:windows]
        if count > 0:
            values_biomarker = x[windows:]
            for i in range(count):
                if biomarker[i] == 'ldl':
                    Cldl0 = values_biomarker[i]
                if biomarker[i] == 'chol':
                    Cchol0 = values_biomarker[i]
                if biomarker[i] == 'trig':
                    Ctrig0 = values_biomarker[i]
                if biomarker[i] == 'hdl':
                    Chdl0 = values_biomarker[i]
                if biomarker[i] == 'pre_adherence':
                    pre_adherence = values_biomarker[i]
                if biomarker[i] == 'alpha':
                    alpha = values_biomarker[i]

            if 'alpha' in biomarker:
                Cldl0 = Cldl0 * alpha
                Cchol0 = Cchol0 * alpha

            Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0 = synthesis_calculation(Cldl0, Cchol0, Ctrig0, Chdl0, prestatin, statintype, statin_dose, pre_adherence)
        
        adherence = adherence_guess(adherence, pairs, values_adherence, period_nonzero)
        
        ldl_max = max(ldl)
        tc_max = max(tc)
        # if len(trig)>0:
        # 	trig_max = max(trig)
        # else:
        # 	trig_max = 1
        trig_max = 1 #max(trig)
        hdl_max = 1 #max(hdl)
        
        Cldl, Cchol, Ctrig, Chdl = differential_solve(adherence, t, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0, dose)
        
        rmse_ldl = rmse_function(ldl, t_ldl, ldl_max, t, Cldl)
        rmse_tc = rmse_function(tc, t_tc, tc_max, t, Cchol)
        # rmse_trig = rmse_function(trig, t_trig, trig_max, t, Ctrig)
        rmse_trig = 0
        rmse_hdl = 0 #rmse_function(hdl, t_hdl, hdl_max, t, Chdl)
        
        rmse_total = (1.2 * rmse_ldl) + rmse_tc + (rmse_trig * 0) +rmse_hdl
        
        return rmse_total
    
    except Exception as e:
        # print 'There was some problem with the get_total_rmse function: {}'.format(e)
        raise

def synthesis_calculation(Cldl0, Cchol0, Ctrig0, Chdl0, prestatin, statintype, statin_dose, pre_adherence):
    try:
        
        ldl_eff  = np.load('../data/final/Efficacy/ldl_efficacy.npy')
        chol_eff = np.load('../data/final/Efficacy/tc_efficacy.npy')
        trig_eff = np.load('../data/final/Efficacy/trig_efficacy.npy')
        hdl_eff  = np.load('../data/final/Efficacy/hdl_efficacy.npy')
        n        = 0.7

        dx = math.log(2)/14
        if pd.isnull(Cldl0) | pd.isnull(Cchol0) | pd.isnull(Ctrig0) | pd.isnull(Chdl0):
            print(Cldl0, Cchol0, Ctrig0, Chdl0, prestatin, statintype, statin_dose)
            Cldl0, Cchol0, Ctrig0, Chdl0 = baseline_map(Cldl0, Cchol0, Ctrig0, Chdl0, prestatin, statintype, statin_dose)

        if prestatin:
            Sx0ldl  = (dx*Cldl0)/(1-h0_cal(statin_dose, ldl_eff[0], ldl_eff[1], n, pre_adherence)[statintype])
            Sx0chol = (dx*Cchol0)/(1-h0_cal(statin_dose, chol_eff[0], chol_eff[1], n, pre_adherence)[statintype])
            Sx0trig = (dx*Ctrig0)/(1-h0_cal(statin_dose, trig_eff[0], trig_eff[1], n, pre_adherence)[statintype])
            Sx0hdl  = (dx*Chdl0)/(1-h0_cal(statin_dose, hdl_eff[0], hdl_eff[1], n, pre_adherence)[statintype])
        else:
            Sx0ldl  = (dx*Cldl0)
            Sx0chol = (dx*Cchol0)
            Sx0trig = (dx*Ctrig0)
            Sx0hdl  = (dx*Chdl0)
            # print(Cldl0, Cchol0, Ctrig0, Chdl0)

        return Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0

    except Exception as e:
        # print 'There was some problem with the synthesis_calculation function: {}'.format(e)
        raise

def baseline_map(Cldl0, Cchol0, Ctrig0, Chdl0, prestatin, statintype, statin_dose):
    try:

        

        ldl = {'Atorvastatin': {'5': 0.31, '10': 0.37, '15': 0.40, '20': 0.43, '30': 0.46,'40': 0.49, '45': 0.50, '50': 0.51, '60': 0.52, '70': np.nan, '80': 0.55},
                'Fluvastatin': {'5': 0.10, '10': 0.15, '15': np.nan, '20': 0.21, '30': np.nan, '40': 0.27, '45': np.nan, '50': np.nan, '60': np.nan, '70': np.nan, '80': 0.33},
                'Lovastatin': {'5': np.nan, '10': 0.21 , '15': np.nan, '20': 0.29, '30': 0.33, '40': 0.37, '45': np.nan, '50': np.nan, '60': np.nan, '70': np.nan, '80': 0.45},
                'Pravastatin': {'5': 0.15, '10': 0.2, '15': np.nan, '20': 0.24, '30': 0.27, '40': 0.29, '45': np.nan, '50': np.nan, '60': np.nan, '70': np.nan, '80': 0.33}, 
                'Rosuvastatin': {'5': 0.38, '10': 0.43, '15': 0.46, '20': 0.48, '30': 0.51, '40': 0.53, '45': np.nan, '50': np.nan, '60': np.nan, '70': np.nan, '80': 0.58},
                'Simvastatin': {'5': 0.23, '10': 0.27, '15': 0.3, '20': 0.32, '30': 0.35, '40': 0.37, '45': 0.38, '50': 0.38, '60': 0.4, '70': 0.41, '80': 0.42}}

        tc = {'Atorvastatin': {'5': 0.24, '10': 0.29, '15': 0.31, '20': 0.33, '30': 0.36, '40': 0.38, '45': 0.39, '50': 0.39, '60': 0.4, '70': np.nan, '80': 0.43},
                'Fluvastatin':  {'5': 0.07, '10': 0.12, '15': np.nan, '20': 0.17, '30': np.nan, '40': 0.21, '45': np.nan, '50': np.nan, '60': np.nan, '70':np.nan, '80': 0.26},
                'Lovastatin': {'5': np.nan, '10': 0.17, '15': np.nan, '20': 0.23, '30': 0.26, '40': 0.29, '45': np.nan, '50': np.nan, '60': np.nan, '70': np.nan, '80': 0.35},
                'Pravastatin': {'5': 0.12, '10': 0.15, '15': np.nan, '20': 0.19, '30': 0.21, '40': 0.22, '45': np.nan, '50': np.nan, '60': np.nan, '70': np.nan, '80': 0.26}, 
                'Rosuvastatin': {'5': 0.3, '10': 0.34, '15': 0.36, '20': 0.38, '30': 0.39, '40': 0.41, '45': np.nan, '50': np.nan, '60': np.nan, '70': np.nan, '80': 0.45},
                'Simvastatin': {'5': 0.17, '10': 0.21, '15': 0.23, '20': 0.25, '30': 0.27, '40': 0.29, '45': np.nan, '50': 0.3, '60': 0.31, '70': 0.32, '80': 0.33}}

        trig = {'Atorvastatin': {'5': 0.16, '10': 0.19, '15': 0.2, '20': 0.21, '30': 0.23, '40': 0.25, '45': 0.25, '50': 0.25, '60': 0.26, '70': np.nan, '80': 0.27},
                'Fluvastatin':  {'5': 0.05, '10': 0.08, '15': np.nan, '20': 0.11, '30': np.nan, '40': 0.14, '45': np.nan, '50': np.nan, '60': np.nan, '70': np.nan, '80': 0.16},
                'Lovastatin': {'5': np.nan, '10': 0.11, '15': np.nan, '20': 0.15, '30': 0.16, '40': 0.18, '45': np.nan, '50': np.nan, '60': np.nan, '70': np.nan, '80': 0.22},
                'Pravastatin': {'5': 0.08, '10': 0.10, '15': np.nan, '20': 0.12, '30': 0.13, '40': 0.14, '45': np.nan, '50': np.nan, '60': np.nan, '70': np.nan, '80': 0.17}, 
                'Rosuvastatin': {'5': 0.19, '10': 0.22, '15': 0.23, '20': 0.24, '30': 0.25, '40': 0.27, '45': np.nan, '50': np.nan, '60': np.nan, '70': np.nan, '80': 0.29},
                'Simvastatin': {'5': 0.11, '10': 0.14, '15': 0.15, '20': 0.16, '30': 0.17, '40': 0.18, '45': np.nan, '50': 0.19, '60': 0.20, '70': 0.20, '80': 0.21}}


        hdl = {'Atorvastatin': {'5': 1.0, '10': 1.0, '15': 1.0, '20': 1.0, '30': 1.0, '40': 1.0, '45': 1.0, '50': 1.0, '60': 1.0, '70':1.0, '80': 1.0},
                'Fluvastatin':  {'5': 1.0, '10': 1.0, '15': 1.0, '20': 1.0, '30': 1.0, '40': 1.0, '45': 1.0, '50': 1.0, '60': 1.0, '70': 1.0, '80': 1.0},
                'Lovastatin': {'5': 1.0, '10': 1.0, '15': 1.0, '20': 1.0, '30': 1.0, '40': 1.0, '45': 1.0, '50': 1.0, '60': 1.0, '70': 1.0, '80': 1.0},
                'Pravastatin': {'5': 1.0, '10': 1.0, '15': 1.0, '20': 1.0, '30': 1.0, '40': 1.0, '45': 1.0, '50': 1.0, '60': 1.0, '70': 1.0, '80': 1.0}, 
                'Rosuvastatin': {'5': 1.0, '10': 1.0, '15': 1.0, '20': 1.0, '30': 1.0, '40': 1.0, '45': 1.0, '50': 1.0, '60': 1.0, '70': 1.0, '80': 1.0},
                'Simvastatin': {'5': 1.0, '10': 1.0, '15': 1.0, '20': 1.0, '30': 1.0, '40': 1.0, '45': 1.0, '50': 1.0, '60': 1.0, '70': 1.0, '80': 1.0}}
        

        Cldl_prestatin = 4.78407034
        Cchol_prestatin = 6.77527799
        Ctrig_prestatin = 4.65168793
        Chdl_prestatin = 1.81018878
        if prestatin == False:
            if pd.isnull(Cldl0):
                Cldl0 = Cldl_prestatin
            if pd.isnull(Cchol0):
                Cchol0 = Cchol_prestatin
            if pd.isnull(Ctrig0):
                Ctrig0 = Ctrig_prestatin
            if pd.isnull(Chdl0):
                Chdl0 = Chdl_prestatin
        if prestatin:
            if ~(pd.isnull(statin_dose)):
                statin_dose = str(int(statin_dose))
            if pd.isnull(Cldl0):
                Cldl0 = Cldl_prestatin * ldl[statintype][statin_dose]
            if pd.isnull(Cchol0):
                Cchol0 = Cchol_prestatin * tc[statintype][statin_dose]
            if pd.isnull(Ctrig0):
                Ctrig0 = Ctrig_prestatin * trig[statintype][statin_dose]
            if pd.isnull(Chdl0):
                Chdl0 = Chdl_prestatin * hdl[statintype][statin_dose]

        return Cldl0, Cchol0, Ctrig0, Chdl0

    except Exception as e:
        print('There was some problem with the baseline_map function: {}'.format(e))

def optimize_callback(x, pairs, windows, period_nonzero, adherence, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0, dose, t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl,t,count, biomarker,pre_adherence,prestatin, statintype, statin_dose):
    try:
        values_adherence = x[0:windows]
        if count > 0:
            values_biomarker = x[windows:]
            for i in range(count):
                if biomarker[i] == 'ldl':
                    Cldl0 = values_biomarker[i]
                if biomarker[i] == 'chol':
                    Cchol0 = values_biomarker[i]
                if biomarker[i] == 'trig':
                    Ctrig0 = values_biomarker[i]
                if biomarker[i] == 'hdl':
                    Chdl0 = values_biomarker[i]
                if biomarker[i] == 'pre_adherence':
                    pre_adherence = values_biomarker[i]
                if biomarker[i] == 'alpha':
                    alpha = values_biomarker[i]

            if 'alpha' in biomarker:
                Cldl0 = Cldl0 * alpha
                Cchol0 = Cchol0 * alpha

            Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0 = synthesis_calculation(Cldl0, Cchol0, Ctrig0, Chdl0, prestatin, statintype, statin_dose, pre_adherence)
        
        adherence = adherence_guess(adherence, pairs, values_adherence, period_nonzero)
        
        ldl_max = max(ldl)
        tc_max = max(tc)
        # if len(trig)>0:
        # 	trig_max = max(trig)
        # else:
        # 	trig_max = 1
        trig_max = max(trig)
        hdl_max = max(hdl)
        
        Cldl, Cchol, Ctrig, Chdl = differential_solve(adherence, t, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0, dose)
        
        rmse_ldl = rmse_function(ldl, t_ldl, ldl_max, t, Cldl)
        rmse_tc = rmse_function(tc, t_tc, tc_max, t, Cchol)
        # rmse_trig = rmse_function(trig, t_trig, trig_max, t, Ctrig)
        rmse_trig = 0
        rmse_hdl = 0 #rmse_function(hdl, t_hdl, hdl_max, t, Chdl)
        
        rmse_total = (1.2 * rmse_ldl) + rmse_tc + (rmse_trig * 0) +rmse_hdl
        print(rmse_total)
        return rmse_total
    
    except Exception as e:
        # print 'There was some problem with the get_total_rmse function: {}'.format(e)
        raise

def optimize_params(pairs, windows, period_nonzero, adherence, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0, dose, t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl,t, count, biomarker, pre_adherence, prestatin, statintype, statin_dose):
    print('optimize_params')
    try:

        if (('ldl' not in biomarker) and ('chol' not in biomarker) and ('hdl' not in biomarker)):
            alpha_lower = np.nanmax((0.1, Chdl0 / (Cchol0 - Cldl0)))
        else:
            print('else statement')
            alpha_lower = 0.1
        alpha_upper = 3.0

        optimal_range = {'ldl'				: {'lo': 1.292, 'hi':5.171},
                         'chol'				: {'lo': 2.585, 'hi':9.05},
                          'trig'			: {'lo': 1.129, 'hi':5.645},
                          'hdl'				: {'lo': 0.775, 'hi':1.81},
                          'pre_adherence'	: {'lo': 0.01, 'hi': 1.0},
                          'alpha'			: {'lo': alpha_lower, 'hi': alpha_upper}
                          }
        print('optimal range done')

        low = []
        high = []
        for name in biomarker:
            low.append(optimal_range[name]['lo'])
            high.append(optimal_range[name]['hi'])

        print('setting bounds')

        npar = windows+count
        bounds = np.zeros([npar,2])
        bounds[:,0] = [0.01]*windows + low
        bounds[:,1] = [1]*windows + high

        # Convert bounds to list of tuples
        boundsList = [tuple(bounds[i,:]) for i in range(bounds.shape[0])]

        #solver = minimize(get_total_rmse, x0=np.mean(bounds, axis=1).tolist(), bounds=boundsList, 
        #    args = (pairs, windows, period_nonzero, adherence, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, 
        #        Cldl0, Cchol0, Ctrig0, Chdl0, dose, t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl, 
        #        t, count, biomarker, pre_adherence, prestatin, statintype, statin_dose))

        #best_rmse = np.inf
        #all_vals = {}

        #for i in range(1):

        #    result = solver.fun
        #    vals = solver.x

        #    if result < best_rmse:
        #        best_rmse, best_vals = result, vals

        #    all_vals[i] = {}
        #    all_vals[i]['Error'] = result
        #    all_vals[i]['params'] = list(vals)

        solver = Differential_Evolution(obj_fun=get_total_rmse, bounds=bounds, parallel= True, npar=npar, npool=npar*8, 
        									CR=0.85, strategy=2, fmin=0, fmax=2)
        print(solver.parameter_number)
        result = solver.optimize(args = [pairs, windows, period_nonzero, adherence, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, 
         									Cldl0, Cchol0, Ctrig0  , Chdl0, dose, t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl, 
         									t, count, biomarker, pre_adherence, prestatin, statintype, statin_dose])
        best_rmse, best_vals = result[:2]

        return best_rmse, best_vals

    except Exception as e:
        # print 'There was some problem with the optimize_params function: {}'.format(e)
        raise

def plotting(t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl, Cldl_f, Cchol_f, Ctrig_f, Chdl_f, t, p, tmax=1095):
    try:

        # fontP = FontProperties()
        plt.style.use(['seaborn-white', 'seaborn-talk'])
        sns.set_style('ticks', {'font.family': ['Times New Roman'], 'font.size': ['18']})
        sns.set_context('talk', font_scale=1)
        # fontP.set_size('24')

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.plot(t_ldl, ldl, '*', color='teal', label='Real data', markersize = '24')
        ax.plot(t, Cldl_f, color='darkblue', label='ODE Simulation')
        ax.set_xlabel('Days from baseline')
        ax.set_ylabel('LDL, mmol/L')
        ax.set_xlim(0, tmax)
        ax.legend(frameon=True, framealpha=0.7, fontsize = '18')
        fig.tight_layout()
        outdir = os.path.join(classConfig['outputPath'], 'LDL_Simulation')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fig.savefig('{}/Patient{}'.format(outdir, p))

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.plot(t_tc, tc, '*', color='teal', label='Real data', markersize = '24')
        ax.plot(t, Cchol_f, color='darkblue', label='ODE Simulation')
        ax.set_xlabel('Days from baseline')
        ax.set_ylabel('Total cholesterol, mmol/L')
        ax.set_xlim(0, tmax)
        ax.legend(frameon=True, framealpha=0.7, fontsize = '18')
        fig.tight_layout()
        outdir = os.path.join(classConfig['outputPath'], 'Chol_Simulation')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fig.savefig('{}/Patient{}'.format(outdir, p))

        # fig = plt.figure(figsize=(12,8))
        # ax = fig.add_subplot(111)
        # ax.plot(t_trig, trig, '*', color='teal', label='Real data', markersize = '24')
        # ax.plot(t, Ctrig_f, color='darkblue', label='ODE Simulation')
        # ax.set_xlabel('Days from baseline')
        # ax.set_ylabel('Triglycerides, mmol/L')
        # ax.set_xlim(0, 730)
        # ax.legend(frameon=True, framealpha=0.7, fontsize = '18')
        # fig.tight_layout()
        # outdir = os.path.join(classConfig['outputPath'], 'Trig_Simulation')
        # if not os.path.exists(outdir):
        #     os.makedirs(outdir)
        # fig.savefig('{}/Patient{}'.format(outdir, p))

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)
        ax.plot(t_hdl, hdl, '*', color='teal', label='Real data', markersize = '24')
        ax.plot(t, Chdl_f, color='darkblue', label='ODE Simulation')
        ax.set_xlabel('Days from baseline')
        ax.set_ylabel('HDL, mmol/L')
        ax.set_xlim(0, tmax)
        ax.legend(frameon=True, framealpha=0.7, fontsize = '18')
        fig.tight_layout()
        outdir = os.path.join(classConfig['outputPath'], 'HDL_Simulation')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fig.savefig('{}/Patient{}'.format(outdir, p))

        plt.close('all')

    except Exception as e:
        # print 'There was some problem with the plotting function: {}'.format(e)
        raise

def plotAdherence(adhData, scatterPoints, labels, outputFileName, tmax=1095):
    try:
        plt.style.use(['seaborn-white', 'seaborn-talk'])
        sns.set_style('ticks', {'font.family': ['Times New Roman']})
        sns.set_context('talk', font_scale=1)

        fig = plt.figure(figsize=(12,8))
        ax = fig.add_subplot(111)

        for m in np.arange(0, adhData.shape[1]):
            ax.plot(adhData[:,m], label=labels[m])
            ax.scatter(np.arange(0, scatterPoints[:,m].shape[0]), scatterPoints[:,m])
        ax.set_xlabel('Days from baseline')
        ax.set_ylabel('Adherence')
        ax.set_xlim(0, tmax)
        ax.set_ylim(bottom=0)
        ax.legend(frameon=True, framealpha=0.7)
        fig.tight_layout()

        outdir = os.path.join(classConfig['outputPath'], 'Adherence')
        if not os.path.exists(outdir):
            os.makedirs(outdir)
        fig.savefig('{}/Adherence_{}.png'.format(outdir, outputFileName))

        plt.close('all')
    except Exception as e:
        raise

def main_simulate(patientListPickleFile, **kwargs):
    try:
        tic = time.time()

        patientsList = pd.read_pickle(str(patientListPickleFile))
        patientList = np.array(patientsList['NRIC_X'].astype(str)) #Note the difference in names of the above two

        patSN = pd.read_pickle('../../data/intermediate/patientSN_info.pkl')
        patSN_dict = dict(patSN.apply(lambda x: tuple(x), axis=1).values)

       #  patientsToRun = ['1841', '1993', '2022', '2134', '2272', '2457', '2682', '3088', '3606', '3670',
                            # '2341', '2360', '2466', '2534', '2743', '2787', '2849', '3198', '4267', '4347']
        # patientsToRun = ['2326']

        rmseFinal = pd.DataFrame(columns=['NRIC_X', 'PatientSN', 'TotalRMSE_nonNorm'])
        for p in patientList:
            try:
                print(p, patSN_dict)
                p1 = patSN_dict[p]
                # if p1 not in patientsToRun:
                #     continue
                print('Loading data for patient {}'.format(p1))
                myPatient = getPatientData(p, **kwargs)
                myPatient.patientSN = p1
                myPatient.loadOrigMedications()

                adherence = myPatient.origMeds['statin']['adherence']
                periods = myPatient.origMeds['statin']['periods']
                dose = myPatient.origMeds['statin']['dose']

                t_ldl, ldl = myPatient.biomarker(['LDL'])
                t_tc, tc = myPatient.biomarker(['Cholesterol'])
                t_trig, trig = myPatient.biomarker(['Triglycerides'])
                t_hdl, hdl = myPatient.biomarker(['HDL'])

                t_ldl_1 = []
                ldl_1 = []
                for i in range(len(t_ldl)):
                    t_ldl_1.append(t_ldl[i][0])
                    ldl_1.append(ldl[i][0])

                t_tc_1 = []
                tc_1 = []
                for i in range(len(t_tc)):
                    t_tc_1.append(t_tc[i][0])
                    tc_1.append(tc[i][0])

                t_trig_1 = []
                trig_1 = []
                for i in range(len(t_trig)):
                    t_trig_1.append(t_trig[i][0])
                    trig_1.append(trig[i][0])

                t_hdl_1 = []
                hdl_1 = []
                for i in range(len(t_hdl)):
                    t_hdl_1.append(t_hdl[i][0])
                    hdl_1.append(hdl[i][0])

                t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl =  t_ldl_1, ldl_1, t_tc_1, tc_1, t_trig_1, trig_1, t_hdl_1, hdl_1
                # print(t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl)
                print('Loading data for patient {}'.format(p1))
                Cldl0 = myPatient.baseline(['LDL'])[0][0]
                Cchol0 = myPatient.baseline(['Cholesterol'])[0][0]
                Ctrig0 = myPatient.baseline(['Triglycerides'])[0][0]
                Chdl0 = myPatient.baseline(['HDL'])[0][0]
                
                # print(Cldl0, Chdl0, Cchol0, Ctrig0)

                prestatin = myPatient.baseline(['Statin_Prior'])[0][0]
                statintype = myPatient.baseline(['Statin_Prior_Type'])[0][0]
                statin_dose = myPatient.baseline(['Statin_Prior_Dose'])[0][0]
                # pre_adherence = myPatient.baseline(['Statin_Pre_Adherence'])[0][0]
                pre_adherence = 0
                
                
                ldl_pre = int(pd.isnull(Cldl0))
                chol_pre = int(pd.isnull(Cchol0))
                # trig_pre = int(pd.isnull(Ctrig0))
                hdl_pre = int(pd.isnull(Chdl0))

                if prestatin == 1:
                    pre_adherence = 1


                # Added this so that lipid sim will be evaluated for each day
                # t = np.sort(reduce(np.union1d, [t_ldl, t_tc, t_trig, t_hdl]))
                t_max = 1095
                t = np.arange(0, (t_max+1), 1)

                # Load optimized values
                myPatient.loadOptimizedMedications()
                currAdherence 	= myPatient.optimizedMeds['statin']['adherence']
                currDose		= myPatient.optimizedMeds['statin']['dose']
                currSxo 		= myPatient.optimizedMeds['statin']['Sxo']
                currCxo 		= myPatient.optimizedMeds['statin']['Cxo']

                total_rmse_nonNorm = get_total_rmse_nonNorm(currAdherence, currSxo['LDL'], currSxo['Cholesterol'], currSxo['Triglycerides'], currSxo['HDL'], 
                    currCxo['LDL'], currCxo['Cholesterol'], currCxo['Triglycerides'], currCxo['HDL'], currDose, 
                    t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl, t)

                rmseFinal = rmseFinal.append({'NRIC_X': p, 'PatientSN': p1, 'TotalRMSE_nonNorm': total_rmse_nonNorm}, ignore_index=True)


            except Exception as e:
                print('Patient cannot be processed: {}\n'.format(e))
                continue
                # raise

        rmseFinal.to_pickle('{}/{}/TotalRMSE.pkl'.format(classConfig['savePath']['final'], classConfig['savePath']['optimizedMeds']))

    except Exception as e:
        print('There was some problem with the main function: {}'.format(e))
        # raise

def main(patientList, **kwargs):
    try:
        tic = time.time()

        #patientsList = pd.read_pickle(str(patientListPickleFile))
        #print('patientsList', patientsList)
        #patientList = [np.array(patientsList['NRIC_X'].astype(str))] #Note the difference in names of the above two

        #patSN = pd.read_pickle('../../data/intermediate/patientSN_info.pkl')
        #patSN_dict = dict(patSN.apply(lambda x: tuple(x), axis=1).values)

       #  patientsToRun = ['1841', '1993', '2022', '2134', '2272', '2457', '2682', '3088', '3606', '3670',
                            # '2341', '2360', '2466', '2534', '2743', '2787', '2849', '3198', '4267', '4347']
        # patientsToRun = ['2326']
        #print(patSN_dict)
        print(patientList)

        for p in patientList:
            try:
                print('p iterated')
                p1 = p
                # if p1 not in patientsToRun:
                #     continue
                print('Loading data for patient {}'.format(p1))
                myPatient = getPatientData(p, **kwargs)
                print('initiated patient data')
                myPatient.patientSN = p1
                print('initiated patient SN')
                myPatient.loadOrigMedications()
                print('loadedOrigMeds')
                print('Loaded patient {} medications'.format(p1))

                adherence = myPatient.origMeds['statin']['adherence']
                periods = myPatient.origMeds['statin']['periods']
                dose = myPatient.origMeds['statin']['dose']
                print(adherence.shape)

                print('Loading biomarkers for patient {}'.format(p1))
                print(myPatient.biomarker(['LDL']))
                t_ldl, ldl = myPatient.biomarker(['LDL'])
                t_tc, tc = myPatient.biomarker(['Cholesterol'])
                t_trig, trig = myPatient.biomarker(['Triglycerides'])
                t_hdl, hdl = myPatient.biomarker(['HDL'])
                print('Loaded biomarkers for patient{}'.format(p1))


                t_ldl_1 = []
                ldl_1 = []
                for i in range(len(t_ldl)):
                    t_ldl_1.append(t_ldl[i][0])
                    ldl_1.append(ldl[i][0])
                print('loaded ldl')

                t_tc_1 = []
                tc_1 = []
                for i in range(len(t_tc)):
                    t_tc_1.append(t_tc[i][0])
                    tc_1.append(tc[i][0])
                print('loaded tc')

                t_trig_1 = []
                trig_1 = []
                for i in range(len(t_trig)):
                    t_trig_1.append(t_trig[i][0])
                    trig_1.append(trig[i][0])
                print('loaded trig')

                t_hdl_1 = []
                hdl_1 = []
                for i in range(len(t_hdl)):
                    t_hdl_1.append(t_hdl[i][0])
                    hdl_1.append(hdl[i][0])
                print('loaded hdl')

                t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl =  t_ldl_1, ldl_1, t_tc_1, tc_1, t_trig_1, trig_1, t_hdl_1, hdl_1
                # print(t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl)
                print('Loading data for patient {} for baseline'.format(p1))
                Cldl0 = myPatient.baseline(['LDL'])[0][0]
                print('Cldl0', Cldl0)
                Cchol0 = myPatient.baseline(['Cholesterol'])[0][0]
                print('Cchol0', Cchol0)
                Ctrig0 = myPatient.baseline(['Triglycerides'])[0][0]
                print('Ctrig0', Ctrig0)
                Chdl0 = myPatient.baseline(['HDL'])[0][0]
                print('Chdl0', Chdl0)
                
                # print(Cldl0, Chdl0, Cchol0, Ctrig0)
                print('loading statins')
                prestatin = myPatient.baseline(['Statin_Prior'])[0][0]
                statintype = myPatient.baseline(['Statin_Prior_Type'])[0][0]
                statin_dose = myPatient.baseline(['Statin_Prior_Dose'])[0][0]
                # pre_adherence = myPatient.baseline(['Statin_Pre_Adherence'])[0][0]
                pre_adherence = 0
                print('loaded statins')
                
                print('loading prebiomarkers')
                ldl_pre = int(pd.isnull(Cldl0))
                chol_pre = int(pd.isnull(Cchol0))
                # trig_pre = int(pd.isnull(Ctrig0))
                hdl_pre = int(pd.isnull(Chdl0))
                print('loaded prebiomarkers')
                if prestatin == 1:
                    pre_adherence = 1

                # If baseline values are not present, don't optimize alpha
                if ((ldl_pre==1) & (chol_pre==1) & (hdl_pre==1)):
                    optAlpha = 0
                else:
                    optAlpha = 1
                
                print('optimizing')
                optimize_dict = {
                    'ldl': 0,  #ldl_pre 
                     'chol': 0, #chol_pre,
                      'trig': 0, 
                      'hdl': 0, #hdl_pre, 
                      'pre_adherence': 0, #pre_adherence, 
                      'alpha': 0 #optAlpha
                      }
                alpha = 1
                print('loading optimizing dict iterms')
                count = 0
                biomarker = []
                for bio, opt in optimize_dict.items():
                    if opt == 1:
                        count += 1
                        biomarker.append(bio)
                

                # Added this so that lipid sim will be evaluated for each day
                # t = np.sort(reduce(np.union1d, [t_ldl, t_tc, t_trig, t_hdl]))
                t_max = 1300
                t = np.arange(0, (t_max+1), 1)
                print('adherence coding')
                pairs, windows, period_nonzero = adherence_coding(adherence, periods)
                print('adherence coding done \nsynthesis calculation    ')
                Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0 = synthesis_calculation(Cldl0, Cchol0, Ctrig0, Chdl0, 
                                                                                                        prestatin, statintype, statin_dose, pre_adherence)
                print('synthesis calculation done')

                print('Starting optimisation: {} minutes elapsed'.format((time.time()-tic)/60.0))
                total_rmse, best_vals = optimize_params(pairs, windows, 
                                                        period_nonzero, 
                                                        adherence, Sx0ldl, 
                                                        Sx0chol, Sx0trig, 
                                                        Sx0hdl, Cldl0, Cchol0,
                                                        Ctrig0, Chdl0, dose, 
                                                        t_ldl, ldl, t_tc, 
                                                        tc, t_trig, trig, 
                                                        t_hdl, hdl, t, count,
                                                        biomarker, pre_adherence, 
                                                        prestatin, statintype, 
                                                        statin_dose)
                print('Total RMSE = {}'.format(total_rmse))
                values_adherence = best_vals[0:windows]
                if count > 0:
                    values_biomarker = best_vals[windows:]
                    for i in range(count):
                        if biomarker[i] == 'ldl':
                            Cldl0 = values_biomarker[i]
                        if biomarker[i] == 'chol':
                            Cchol0 = values_biomarker[i]
                        if biomarker[i] == 'trig':
                            Ctrig0 = values_biomarker[i]
                        if biomarker[i] == 'hdl':
                            Chdl0 = values_biomarker[i]
                        if biomarker[i] == 'pre_adherence':
                            pre_adherence = values_biomarker[i]
                        if biomarker[i] == 'alpha':
                            alpha = values_biomarker[i]

                if 'ldl' not in biomarker:
                    Cldl0 = Cldl0 * alpha
                if 'chol' not in biomarker:
                    Cchol0 = Cchol0 * alpha

                adherence_final = adherence_guess(adherence, pairs, best_vals[0:windows], period_nonzero)
                Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0 = synthesis_calculation(Cldl0, Cchol0, Ctrig0, Chdl0, prestatin, statintype, statin_dose, pre_adherence)
                total_rmse_nonNorm = get_total_rmse_nonNorm(adherence_final, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0, dose, 
                                            t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl, t)
                t_plotting = np.arange(0, t_max)
                Cldl_f, Cchol_f, Ctrig_f, Chdl_f = differential_solve(adherence_final, t_plotting, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0, dose)

                print('Saving patient optimized parameters: {} minutes elapsed\n'.format((time.time()-tic)/60.0))
                myPatient.saveOptimizedMedications(adherence_final, dose, np.array([Sx0ldl, Sx0chol, Sx0trig, Sx0hdl]), 
                                                    np.array([Cldl0, Cchol0, Ctrig0, Chdl0]), np.array(['LDL', 'Cholesterol', 'Triglycerides', 'HDL']), 
                                                    total_rmse_nonNorm, alpha)
                plotting(t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl, Cldl_f, Cchol_f, Ctrig_f, Chdl_f, t_plotting, p1)

                myPatient.loadOptimizedMedications()
                myPatient.loadOrigMedications()
                optimizedAdh = copy.deepcopy(myPatient.optimizedMeds['statin']['adherence'])
                originalAdh = copy.deepcopy(myPatient.origMeds['statin']['adherence'])
                optimizedAdh[originalAdh!=-1] = np.nan
                plotAdherence(myPatient.optimizedMeds['statin']['adherence'], optimizedAdh, classConfig['medicationsList']['statin'], myPatient.patientSN)

            except Exception as e:
                print('Patient cannot be processed: {}\n'.format( e))
                continue
                # raise

    except Exception as e:
        print('There was some problem with the main function: {}'.format(e))
        # raise

if __name__ == '__main__':
    try:
       
        # Run on 16.09.2019; new optimization, optimize Cxo 
        patientListPickleFile = '../data/intermediate/patientSN_info.pkl'
        classConfigFile = '../config/lipidoptimizing.json' # use the 3y directory so that it doesnt affect currently running risk model
        # classConfigFile = riskConfig['patientClassConfig']
        main(patientListPickleFile, classConfigFile=classConfigFile)



    except Exception as e:
        print("Cant call main function. ERROR: {}".format(str(e)))