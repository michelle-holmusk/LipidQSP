import json, pprint
import numpy as np
import pandas as pd
import os
from functools import reduce
from os.path import join

class getPatientData():

	def __init__(self, patientUID, classConfigFile='../config/lipidoptimizing.json'):
		'''
		This function initialize individual patient's class 

		Input arguments
		--------------
		patientUID : (str) patient's UID (NRIC_X)
		'''
		try:
			self.patientUID = patientUID
			print('patientUID type', type(patientUID))
			print('patientUID', patientUID)
			self.origMeds = dict()
			self.optimizedMeds = dict()
			self.classConfig = json.load(open(classConfigFile))
			print('__init__ happened')
		except Exception as e:
			print('Unable to initialize patient class :{}'.format(str(e)))
		return

	def replaceNaN(self, inp):
		# Replace empty values with nan
		try:
			if not isinstance(inp, list) and not isinstance(inp, str) and not isinstance(inp, np.ndarray):
				if inp is np.nan:
					return []
				elif np.isnan(inp):
					return []
				else:
					return inp
			else:
				return inp
			print('replaceNan happened')
		except Exception as e:
			print('Unable to convert to categorical :{}'.format(str(e)))
		return

	def convertCategoricalToInt(self, tbl, excludeCol=['NRIC_X', 'Statin_Prior_Type']):
		'''
		This function converts pandas DataFrame columns of dtype 'object' to categorical

		Input arguments
		--------------
		tbl : pandas DataFrame

		Output
		--------------
		tbl : pandas DataFrame (all columns are of dtype numbers, either int or float)
		'''
		try:
			tbl[[x for x in tbl.columns if x not in excludeCol]] = tbl.drop(excludeCol, axis=1).apply(lambda x: pd.Categorical(x).codes if x.dtype.hasobject else x)
			print('convertCat happened')
			return tbl

		except Exception as e:
			print('Unable to convert to categorical :{}'.format(str(e)))

		return

	def baseline(self, baselineVarList, baselineTable=None):
		'''
		This function pull individual patient's baseline data for all variables specified by baselineVarList
		If column's dtype is object, it will treat it as categorical and convert to integers

		Input arguments
		--------------
		baselineVarList : (list) list of baseline variables, e.g. ['Age', 'Gender', 'Race']

		Output
		--------------
		baselineData : (np array (1, len(baselineVarList)), e.g. [[45 0 3 2]]
		'''
		try:
			if baselineTable is None:
				tbl = pd.read_pickle(str(self.classConfig['baselineTable']))
			else:
				tbl = pd.read_pickle(str(baselineTable))
				
			if tbl.select_dtypes(include=['object']).size != 0:
				tbl = self.convertCategoricalToInt(tbl)
			baselineDataList = tbl[list(baselineVarList)].loc[tbl.NRIC_X==self.patientUID]
			self.baselineData = np.array(baselineDataList)
			print('baseline happened')
			return self.baselineData

		except Exception as e:
			print('Unable to get baseline data :{}'.format(str(e)))
			# raise

		return

	def saveToNPZ(self, obj, outFileName, outputDir=None):
		'''
		This function saves an object (in the class) to an npz file
		Default outputDir is specified in classConfig['savePath']['intermediate']
		Output filename is Patient{PatientUID}_{obj}_{keys}.npz

		Input arguments
		--------------
		obj	: (str) object name in the class to be saved, e.g. 'origMeds'
		outputDir	: (str) optional, full path to the output directory, e.g. '/Users/jesiscatandi/Document'
		'''
		try:
			if outputDir is None:
				outputDir = self.classConfig['savePath']['intermediate']

			outputDir = os.path.join(outputDir, outFileName)
			if not os.path.exists(outputDir):
				os.mkdir(outputDir)

			if (getattr(self, obj) is not None):
				for j in getattr(self, obj):
					savez_dict = dict()
					for k in getattr(self, obj)[j]:
						savez_dict[k] = getattr(self, obj)[j][k]
					np.savez(join(outputDir, 'Patient{}_{}_{}.npz'.format(self.patientSN, str(obj), j)), **savez_dict)
			print('saveToNPZ happened')

		except Exception as e:
			# print 'Unable to save data :{}'.format(str(e))
			raise

		return

	def processMedications(self, medicationsFamily="statin", biomarkersFamily="lipid"):
		'''
		This function processes individual patient's raw medication data, convert them into numpy arrays for subsequent analysis, and save them into self.origMeds
		Paths to the raw tables are specified in classConfig: 'medicationsTable' and 'biomarkersTable'
		List of medications and biomarkers are provided in classConfig['medicationsList'] and  classConfig['biomarkersList'] respectively

		Input arguments
		--------------
		medicationsFamily 	: (str) name of the medications family (corresponds to the key values in classConfig['medicationsList'])
								default: 'statin'
		biomarkersFamily	: (str) name of the biomarkers family (corresponds to the key values in classConfig['biomarkersList'])
								default: 'lipid'

		This function will generate 4 numpy arrays and save them into an .npz file: 
		1) dose 		: (day, M) dose[x,y] is the dose of medication y prescribed on day x. Value is equal to dose when drug is prescribed, otherwise 0 
		2) adherence 	: (day, M) adherence[x,y] is the adherence of medication y on day x. 
							Value is equal to :
								calculated adherence when drug is prescribed but no biomarker measure is available at that period, 
								-1 when drug is prescribed and when biomarker measure is available
								otherwise 0 (when drug is not prescribed)
		3) periods 		: (day, 1) periods[x,0] is the adherence period label that specify which adherence period does day x belong to
							Values : 
								0 when adherence does not need to be optimized
								{1, ..., A} when adherence needs to be optimized
		4) periodsTotal	: (day, 1) periodsTotal[x,0] is the period label that specify which period does day x belong to (new period label when there is any dose/medication changes)
							Values : {1, ..., P} 
		where 
		M = the number of medication types under this medication family,
		day = the last day that we have measurement (either medication or biomarker information)
		P = total number of periods,
		A = total number of periods with adherence to be estimated (A<=P)
		
		Output file will be saved into classConfig['savePath']['intermediate'] with filename Patient{PatientUID}_origMeds_medicationsFamily.npz

		'''
		try:
			# classConfig = json.load(open('Holmusk/AmgenRR/config/patientClass.json'))
			tblMed 				= pd.read_pickle(str(self.classConfig['medicationsTable']))
			tblBiomarker 		= pd.read_pickle(str(self.classConfig['biomarkersTable']))
			patientMed 			= tblMed[tblMed.NRIC_X==self.patientUID]
			patientBiomarker	= tblBiomarker[tblBiomarker.NRIC_X==self.patientUID]
			patientMed			= patientMed.applymap(self.replaceNaN)
			patientBiomarker	= patientBiomarker.applymap(self.replaceNaN)

			if patientMed.shape[0]==0:
				print('No follow up {} info for patient {}'.format(medicationsFamily, self.patientUID))

			# Added on 29.08.2019 : to impute those with missing data after dataAvailability day in cohortList_withDataAvailability.pkl
			tblAvail 		= pd.read_pickle(self.classConfig['dataAvailTbl'])
			patientAvail	= int(tblAvail[tblAvail.NRIC_X==self.patientUID].dataAvailability.values)

			# Set the last day : find the latest day with either dose or biomarker data (we might just hard code this in the future)
			# t_max = 1095 # 3y
			# Added +1 here so that if biomarker measured on StartDay, it'll be fitted to the previous period
			t_max = max(np.array(patientMed.filter(regex='EndDayFinal').apply(lambda x: np.max(np.concatenate(x)), axis=1))[0] + 1, 
			 	np.array(patientBiomarker.filter(regex='Time_').apply(lambda x: np.max(np.concatenate(x)), axis=1))[0])
			t_max = int(t_max)
			
			# Initialize arrays to store data
			adherence 				= np.zeros((t_max+1,len(self.classConfig['medicationsList'][medicationsFamily])), dtype=float)
			dose 					= np.zeros((t_max+1,len(self.classConfig['medicationsList'][medicationsFamily])), dtype='int')
			availableBiomarkerData 	= np.zeros((t_max+1,1), dtype='int')

			# Loop through the list of medications prescribed
			# Assume StartDay and EndDayFinal are in ascending order
			periodsTotal = np.zeros((t_max+1, 1), dtype='int')
			if (patientMed.Type.values is not np.nan):
				newperiod = 1
				for m in np.arange(len(patientMed.Type.values[0])):
					# Fix on 28-June-2019: consider scenario when StartDay<0, shift it to 0
					# Added +1 here so that if biomarker measured on StartDay, it'll be fitted to the previous period

					r0 = max(0, (int(patientMed['StartDay'].values[0][m]) + 1))
					# Added on 09.09.2019 to handle cases when prescription started after end of study
					if (r0 <= t_max):
						r1 = int(patientMed['EndDayFinal'].values[0][m] + 1) + 1
						c0 = np.isin(self.classConfig['medicationsList'][medicationsFamily], patientMed['Type'].values[0][m])
						dose[r0:r1, c0] = patientMed['Dose'].values[0][m]
						adherence[r0:r1, c0] = patientMed['Adherence'].values[0][m]
						periodsTotal[r0:r1, 0] = newperiod
						newperiod +=1

				# Added on 29.08.2019 : to impute those with missing data after dataAvailability day in cohortList_withDataAvailability.pkl
				# Check which is earlier, data availability or max(EndDayFinal). 
				# If last prescription ended before/at dataAvailability (i.e. max(EndDayFinal) <= dataAvailability), adherence at t > dataAvailability is equal to adherence at t=dataAvailability
				# If last prescription ended after dataAvailability (i.e. max(EndDayFinal) > dataAvailability), adherence at t > max(EndDayFinal) is equal to adherence at t=max(EndDayFinal)
				if patientAvail < t_max:
					lastDay = int(np.max(patientMed.EndDayFinal.values[0]) + 1)

					# If the last prescription ended before we lost track of the patient
					if (lastDay <= patientAvail):
						dose[(patientAvail+1):, :] = dose[patientAvail, :]
						adherence[(patientAvail+1):, :] = adherence[patientAvail, :]
						periodsTotal[(patientAvail+1):, 0] = newperiod
						newperiod +=1
					elif (lastDay > patientAvail):
						# If the last prescription ended after we lost track of the patient (but ended before end of study such that we need to impute)
						if (lastDay < (t_max)):
							dose[(lastDay+1):, :] = dose[lastDay, :]
							adherence[(lastDay+1):, :] = adherence[lastDay, :]
							periodsTotal[(lastDay+1):, 0] = newperiod
							newperiod +=1
			# # Identify periods when dose/drug changes
			# periodsTotal = np.cumsum(np.concatenate((np.array([0], dtype='int'), 
										# np.where(np.sum(np.absolute(np.diff(dose, axis=0)), axis=1)>0, 1, 0)))).reshape((t_max+1),1) + 1
			
			# Identify periods with biomarker data
			print(patientBiomarker)
			temp2 = list(patientBiomarker.filter(items=['_'.join(x) for x in [[b,a] for a in self.classConfig['biomarkersList'][biomarkersFamily] for b in ['Time']]]).apply(
				lambda x: tuple(reduce(np.union1d, x)), axis=1).values)
			if len(temp2) > 0:
				temp2 = temp2[0]

			# Just make sure values in temp are all integers
			temp = [int(i) for i in temp2]
			availableBiomarkerData[temp, 0] = 1
			availableBiomarkerData = periodsTotal * availableBiomarkerData
			# Replace adherence with -1 for periods with biomarker data
			[np.place(adherence, ((periodsTotal==x) * (dose>0)), -1) for x in availableBiomarkerData[availableBiomarkerData > 0]]
			
			# Create periods (0 when calculated adherence is used and {1, 2, ...} when adherence needs to be estimated)
			periods = np.copy(periodsTotal)
			periods[np.invert((adherence==-1).any(axis=1)), :] = 0
			periodMapping = dict(zip(np.unique(periods), np.arange(np.min(np.unique(periods)), len(np.unique(periods))+1)))
			periods = np.array([periodMapping[x] for w in periods for x in w])
			periods = periods.reshape(len(periods), 1)

			self.origMeds[medicationsFamily] = {'dose': dose, 'adherence': adherence, 'periods': periods, 'periodsTotal': periodsTotal}
			self.saveToNPZ('origMeds', '{}'.format(str(self.classConfig['savePath']['origMeds'])), self.classConfig['savePath']['intermediate'])
			print('processedmedications happened')
			return
			
		except Exception as e:
			# print 'Unable to process medication data :{}'.format(str(e))
			raise

		return

	def loadOrigMedications(self, medicationsFamily='statin', biomarkersFamily='lipid'):
		'''
		This function will load the processed (original, not-optimized) medication arrays of the patient and save them into self.origMeds

		Input arguments
		--------------
		medicationsFamily 	: (str) name of the medications family (corresponds to the key values in classConfig['medicationsList'])
								default: 'statin'
		biomarkersFamily	: (str) name of the biomarkers family (corresponds to the key values in classConfig['biomarkersList'])
								default: 'lipid'
		Output
		------
		After this function is run, dictionaries will be added to className.origMeds[medicationsFamily]
		There are 4 items: 'dose', 'adherence', 'periods', 'periodsTotal'
		For example, to access adherence: myPatient.origMeds['statin']['adherence']
		'''
		try:
			# print 'Loading origMeds'
			processedOrigFile = os.path.join(self.classConfig['savePath']['intermediate'], '{}'.format(str(self.classConfig['savePath']['origMeds'])), 
				'Patient{}_{}_{}.npz'.format(self.patientSN, 'origMeds', medicationsFamily))
			if not (os.path.isfile(processedOrigFile)):
				self.processMedications()
			else:
				arr = np.load(processedOrigFile)
				self.origMeds[medicationsFamily] = dict(arr)
			print('loadorigmed happened')
			return
		except Exception as e:
			# print 'Unable to extract the original medication data :{}'.format(str(e))
			raise

		return

	def saveOptimizedMedications(self, adherence, dose, Sxo, Cxo, x, totalrmse, alpha, medicationsFamily='statin', biomarkersFamily='lipid'):
		'''
		This function save the optimized ODE parameters to an .npz file
		Dose, Sxo, and Cxo are also saved to the .npz file to make it easier to load in the full model

		Input arguments
		--------------
		adherence : (np array (d,M)) 
		dose : (np array (d,M))
		Sxo : (np array (b,)) synthesis term without medication, arranged in the same order as x
		Cxo : (np array (b,)) initial concentration of biomarker x (arranged in the same order as x)
		x	: (np array (b,))

		Output
		------
		Numpy arrays of adherence, dose, Sxo, Cxo, and x will be saved to a .npz file
		'''
		try:
			self.optimizedMeds[medicationsFamily] = {'dose': dose, 'adherence': adherence, 'Sxo': Sxo, 'Cxo': Cxo, 'x': x, 'totalRmse': totalrmse, 'alpha': alpha}
			self.saveToNPZ('optimizedMeds', '{}'.format(str(self.classConfig['savePath']['optimizedMeds'])), self.classConfig['savePath']['final'])
			print('saveoptimizedmedications happened')

		except Exception as e:
			# print 'Unable to save the optimised medication data :{}'.format(str(e))
			raise

		return

	def loadOptimizedMedications(self, medicationsFamily='statin', biomarkersFamily='lipid'):
		'''
		This function will load the optimized adherence,dose,So,Co arrays of the patient and save them into self.optimizedMeds

		Input arguments
		--------------
		medicationsFamily 	: (str) name of the medications family (corresponds to the key values in classConfig['medicationsList'])
								default: 'statin'
		biomarkersFamily	: (str) name of the biomarkers family (corresponds to the key values in classConfig['biomarkersList'])
								default: 'lipid'


		Output
		------
		After this function is run, dictionaries will be added to className.optimizedMeds[medicationsFamily]
		There are 5 items: 'dose', 'adherence', 'Sxo', 'Cxo', 'x'
		For example, optimized adherence np array (d,M) can be accessed from myPatient.optimizedMeds['statin']['adherence']
		Sxo and Cxo are dictionaries, i.e. myPatient.optimizedMeds['statin']['Sxo']['LDL'] is the Sxo of LDL
		'''
		try:
			arr = np.load(os.path.join(self.classConfig['savePath']['final'], '{}'.format(str(self.classConfig['savePath']['optimizedMeds'])), 'Patient{}_optimizedMeds_{}.npz'.format(self.patientSN, medicationsFamily)))
			arr = dict(arr)
			arr['Sxo'] = dict(zip(arr['x'],arr['Sxo']))
			arr['Cxo'] = dict(zip(arr['x'],arr['Cxo']))
			self.optimizedMeds[medicationsFamily] = arr
			print('loadoptimizedmedications happened')
			return
		except Exception as e:
			# print 'Unable to extract the optimised medication data :{}'.format(str(e))
			raise

		return

	def biomarker(self, biomarkersList):
		'''
		This function will pull patient's biomarker data for all biomarkers specified in biomarkersList (in the same order)
		Path to biomarker table is specified in classConfig['biomarkersTable']

		Input arguments
		--------------
		biomarkersList : (list (b,)) list of biomarkers to be extracted. E.g. ['LDL', 'TC']

		Output
		--------------
		t : (np array (d, 1)) time (in day) when biomarker was measured (0 is index event, 1 is 1 day after index event, etc) 
		C : (np array (d, M)) C[x,y] is the biomarker reading of drug y on day t[x,0]. When no reading is available, C[x,y] = np.nan
		'''
		try:

			tbl = pd.read_pickle(str(self.classConfig['biomarkersTable']))
			print('tbl', tbl)
			print('patientUID biomarker', self.patientUID)
			currPatientBiomarker = tbl[tbl.NRIC_X== self.patientUID]
			print('tbl.NRIC_X', type(tbl.NRIC_X))
			print('tbl.NRIC_X type', tbl.NRIC_X.dtypes)
			print('currPatientBiomarker', currPatientBiomarker)
			finalTbl = pd.DataFrame(columns=['t', 'C', 'Type'])

			for b in biomarkersList:
				print(b)
				currBiomarkerArr = np.array(currPatientBiomarker[[x + b for x in ['Time_', 'C_']]].applymap(self.replaceNaN))
				print(currBiomarkerArr)
				currBiomarkerTbl = pd.DataFrame({'t': currBiomarkerArr[0,0], 'C': currBiomarkerArr[0,1]})
				if currBiomarkerTbl.empty:
					currBiomarkerTbl = pd.DataFrame({'t': [np.nan], 'C': [np.nan]})
				currBiomarkerTbl['Type'] = b
				finalTbl = finalTbl.append(currBiomarkerTbl)
				

			finalTbl = finalTbl.reset_index(drop=True)
			print(finalTbl)

			# Print those cases with multiple biomarkers (check the range of values make sense)
			if finalTbl.dropna(subset=['t']).shape[0] != 0:
				tmp = finalTbl.groupby(['Type', 't']).agg('count').reset_index()
				multiples = tmp[~tmp.C.isin([0,1])]
				if multiples.shape[0] != 0:
					multiples = np.array(multiples.values)
					for currType, currDay, currN in multiples:
						print('{} has multiple {} on day {}: {}'.format('Patient', currType, currDay,
							finalTbl[(finalTbl.Type == currType) & (finalTbl.t == currDay)].C.values))
				# Aggregate (mean) multiple measures on one day
				finalTbl = finalTbl.groupby(['Type', 't']).agg('mean').reset_index()

			finalTbl = finalTbl.pivot(index='t', columns='Type', values='C')
			finalTbl = finalTbl.loc[finalTbl.index.dropna()]
			finalTbl['t'] = finalTbl.index
			t = finalTbl['t'].values.reshape(finalTbl['t'].values.size, 1)
			C = finalTbl[biomarkersList].values
			print('biomarker happened')
			return t, C
		except Exception as e:
			# print 'Unable to extract the biomarker data :{}'.format(str(e))
			raise

		return

	def outcomes(self, outcomesList, dataFile=None):
		'''
		This function will pull patient's secondary outcome data for all outcomes specified in outcomesList 
		Path to outcome table is specified in classConfig['outcomeTable']

		Input arguments
		--------------
		outcomesList : (list (o,)) list of outcomes to be extracted. E.g. ['MI', 'Unstable angina']

		Output
		--------------
		t : (np array (d, 1)) time (in day) when any of the secondary outcome specified by outcomesList happened (union of all events)
				(0 is index event, 1 is 1 day after index event, etc) 
		outcome : (np array ones (d, 1)) 
		'''
		try:
			if dataFile is None:
				tbl = pd.read_pickle(str(self.classConfig['outcomeTable']))
			else:
				tbl = pd.read_pickle(dataFile)
				
			tbl = tbl[tbl.NRIC_X==self.patientUID]
			tbl = tbl.drop(['NRIC_X'], axis=1)
			if isinstance(tbl['secondary_event_time'].values[0], list):
				tbl = pd.DataFrame({'t': np.array(tbl['secondary_event_time'].values)[0], 'outcome': np.array(tbl['secondary_event'].values)[0]})
			else:
				tbl = pd.DataFrame({'t': np.array([tbl['secondary_event_time'].values[0]]), 'outcome': np.array([tbl['secondary_event'].values[0]])})

			finalTbl = tbl.pivot_table(index='t', columns='outcome', values='outcome', aggfunc=len)
			finalTbl['t'] = finalTbl.index
			t = finalTbl['t'].values.reshape(finalTbl['t'].values.size,1)

			# Filter the list of outcomes to only those that the patient has
			outcomesList = [x for x in outcomesList if x in finalTbl.columns]

			if len(outcomesList) != 0:
				outcome = finalTbl[outcomesList].values
				t = t[~np.all(np.isnan(outcome), axis=1),]
				outcome = outcome[~np.all(np.isnan(outcome), axis=1),]
				outcome[np.isnan(outcome)] = 0
				outcome[outcome>0] = 1
				# Combine all outcomes
				outcome = np.sum(outcome, axis=1).reshape(outcome.shape[0],1)
			else:
				outcome = np.nan
				t = np.nan
			
			print('outcomes happened')

			return t, outcome

		except Exception as e:
			# print 'Unable to extract the outcome data :{}'.format(str(e))
			raise

		return 

