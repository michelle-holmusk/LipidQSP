import pandas as pd
import numpy as np
import scipy
import os, sys, time, json
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from os.path import join
from datetime import datetime
sys.path.insert(0, 'E:\\YH_GF\\SingCLOUD_Data_Extract\\Holmusk\\AmgenRR\\Scripts\\ODESimulation')
from lipidSimulation import *

'''
This function generates plots of optimized lipid curve.
lipidSimulation.py has to be run first. 

Plots are saved into Results/{lipid}_Simulation/

'''

tic = time.time()

classConfig = json.load(open('Holmusk/AmgenRR/config/patientClass.json'))
ode_patient_list = json.load(open('Holmusk/AmgenRR/config/riskModel.json'))

## Please run lipidSimulation.py first

try:
	patientsList = pd.read_pickle(str(ode_patient_list['trainPatientsList']))
	patientList = np.array(patientsList['NRIC_X'].astype(str)) #Note the difference in names of the above two
	
	patSN = pd.read_pickle(join('Holmusk', 'AmgenRR', 'Data', 'Intermediate', 'patientSN_info.pkl'))
	patSN_dict = dict(patSN.apply(lambda x: tuple(x), axis=1).values)

	# patientsToRun = ['2326']
	
	for p in patientList:
		try:
			p1 = patSN_dict[p]
			# if p1 not in patientsToRun:
			# 	continue
			print('Loading optimized data for patient {}: {} minutes elapsed'.format(p1, (time.time()-tic)/60.0))
			myPatient = getPatientData(p)
			myPatient.patientSN = p1
			myPatient.loadOptimizedMedications()
			
			dose = myPatient.optimizedMeds['statin']['dose']
			adherence = myPatient.optimizedMeds['statin']['adherence']
			Sx0ldl = myPatient.optimizedMeds['statin']['Sxo']['LDL']
			Sx0chol = myPatient.optimizedMeds['statin']['Sxo']['Cholesterol']
			Sx0trig = myPatient.optimizedMeds['statin']['Sxo']['Triglycerides']
			Sx0hdl = myPatient.optimizedMeds['statin']['Sxo']['HDL']
			
			Cldl0 = myPatient.optimizedMeds['statin']['Cxo']['LDL']
			Cchol0 = myPatient.optimizedMeds['statin']['Cxo']['Cholesterol']
			Ctrig0 = myPatient.optimizedMeds['statin']['Cxo']['Triglycerides']
			Chdl0 = myPatient.optimizedMeds['statin']['Cxo']['HDL']
			
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
			t_plotting = np.arange(0, 731)
			
			Cldl, Cchol, Ctrig, Chdl = differential_solve(adherence, t_plotting, Sx0ldl, Sx0chol, Sx0trig, Sx0hdl, Cldl0, Cchol0, Ctrig0, Chdl0, dose)

			plotting(t_ldl, ldl, t_tc, tc, t_trig, trig, t_hdl, hdl, Cldl, Cchol, Ctrig, Chdl, t_plotting, p1)

			myPatient.loadOrigMedications()
			optimizedAdh = copy.deepcopy(myPatient.optimizedMeds['statin']['adherence'])
			originalAdh = copy.deepcopy(myPatient.origMeds['statin']['adherence'])
			optimizedAdh[originalAdh!=-1] = np.nan
			plotAdherence(myPatient.optimizedMeds['statin']['adherence'], optimizedAdh, classConfig['medicationsList']['statin'], myPatient.patientSN)

			print('Plotting completed for patient {}\n'.format(p1))
			
		except Exception as e:
			print('Unable to process patient {}: {}\n'.format(p1, e))
			continue
			# raise
except Exception as e:
	raise