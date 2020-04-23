import pandas as pd
import numpy as np
import pickle
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
import os 
import sys
sys.path.insert(0, '/Users/michelleng/LipidQSP/src/lib')
from databaseIO import pgIO
from psycopg2.sql import SQL, Identifier
from psycopg2 import sql
import decimal


#sys.path.insert(0, os.path.join(rootDir, 'modules'))
import lipidSimulationNew_16092019new as lipid

fileconfig = json.load(open('../config/lipidoptimizing.json'))
print(type(fileconfig))

def loadingdata(fileconfig):
    """
    Reads config data and loads data for lipid optimization
    
    Arguments:
        configfile {[type]} -- [description]
    """

    fileconfig = json.load(open('../config/lipidoptimizing.json'))

    patientquery = SQL("""select * from "testGround".patient_details_nomap;""")#.format(Identifier(fileconfig["rawpatientTable"])))
    columnpatientquery = SQL("""select column_name from information_schema.columns where table_schema = 'testGround' and table_name = 'patient_details_nomap'""")
    medsquery = SQL("""select * from "testGround".meds_data;""")#.format(Identifier(fileconfig["rawMedsTable"])))
    columnmedsquery = SQL("""select column_name from information_schema.columns where table_schema = 'testGround' and table_name = 'meds_data'""")
    cholquery = SQL("""select * from "testGround".results_chol;""")#.format(Identifier(fileconfig["rawCholTable"])))
    columncholquery = SQL("""select column_name from information_schema.columns where table_schema = 'testGround' and table_name = 'results_chol'""")
    ldlquery = SQL("""select * from "testGround".results_ldl;""")#.format(Identifier(fileconfig["rawLDLTable"])))
    columnldlquery = SQL("""select column_name from information_schema.columns where table_schema = 'testGround' and table_name = 'results_ldl'""")

    patientdetails = pgIO.getAllData(patientquery)
    patientcolumn = pgIO.getAllData(columnpatientquery)
    medsdetails = pgIO.getAllData(medsquery)
    medscolumn = pgIO.getAllData(columnmedsquery)
    choldetails = pgIO.getAllData(cholquery)
    cholcolumn = pgIO.getAllData(columncholquery)
    ldldetails = pgIO.getAllData(ldlquery)
    ldlcolumn = pgIO.getAllData(columnldlquery)
    
    Tpatientcolumn = [x[0] for x in patientcolumn]
    Tmedscolumn = [x[0] for x in medscolumn]
    Tcholcolumn = [x[0] for x in cholcolumn]
    Tldlcolumn = [x[0] for x in ldlcolumn]

    patientTable = pd.DataFrame(patientdetails, columns = Tpatientcolumn)
    medsTable = pd.DataFrame(medsdetails, columns = Tmedscolumn)
    cholTable = pd.DataFrame(choldetails, columns = Tcholcolumn)
    ldlTable = pd.DataFrame(ldldetails, columns = Tldlcolumn)

    patientTable.to_pickle(fileconfig["rawpatientpkl"])
    medsTable.to_pickle(fileconfig["rawMedspkl"])
    cholTable.to_pickle(fileconfig["rawCholpkl"])
    ldlTable.to_pickle(fileconfig["rawLDLpkl"])

    print('tables exported')

def cleaningdata(patientlist, fileconfig):
    
    fileconfig = json.load(open('../config/lipidoptimizing.json'))

    # reading patientlist
    #patientTable = pd.read_pickle(fileconfig["dataAvailTbl"])
    #patientlist = list(patientTable['NRIC_X'])
    medsTable = pd.read_pickle(fileconfig['rawMedspkl'])
    cholTable = pd.read_pickle(fileconfig['rawCholpkl'])
    ldlTable = pd.read_pickle(fileconfig['rawLDLpkl'])

    availability = pd.read_pickle(fileconfig['dataAvailTbl'])
    # add patientlist
    availabilitylist = []
    for x in patientlist: 
        if x not in availability['NRIC_X'].values:
            availabilitylist.append([x, 1200])
    if len(availabilitylist) > 0:
        Tavailability = pd.DataFrame(availabilitylist, columns = ['NRIC_X', 'dataAvailability'], index=None)
        finalAvailability = availability.append(Tavailability, ignore_index= True)
        finalAvailability.to_pickle(fileconfig["dataAvailTbl"])

    # reading statinlist
    print(fileconfig['medicationsList'])
    statinlist = fileconfig['medicationsList']['statin']

    for p in patientlist: 
        # changing p to str 
        p = str(p)
        print(p)

        # defining the data in concern for only p stated
        #focusedPatientTable = patientTable[patientTable['id'] == int(p)]
        focusMedsTable = medsTable[medsTable['id'] == p].sort_values(by = 'start_d', ascending = True).reset_index()
        focusCholTable = cholTable[cholTable['id'] == p].sort_values(by = 'duration', ascending = True).reset_index()
        focusLdlTable = ldlTable[ldlTable['id'] == p].sort_values(by = 'duration', ascending = True).reset_index()
        print('Tables focused')

        # changing units of focusCholTable and focusLDLTable values
        focusCholTable['values'] = focusCholTable['value']/38.67
        focusLdlTable['values'] = focusLdlTable['value']/38.67
        print('biomarkers are in correct units')

        # setting baseline stats
        TbaselineChol = focusCholTable['values'][0]
        Tbaselinestart = focusCholTable['duration'][0]
        TbaselineLdl = focusLdlTable['values'][0]
        print('baseline values are recorded')
            
        # refining the meds table to look at the statin data
        TbaselineMeds = focusMedsTable[focusMedsTable['mapped_generic'].isin(statinlist)].reset_index()
        TbaselineMedsstatus = bool(len(TbaselineMeds[TbaselineMeds['start_d'] < Tbaselinestart]['start_d'])==1)
        if TbaselineMedsstatus == True:
            TbaselineStatin = TbaselineMeds['mapped_generic'][0] 
            TbaselineStatinDose = TbaselineMeds['Dose'][0]
        else: 
            TbaselineStatin = np.nan
            TbaselineStatinDose = np.nan
        print('statin status singled out')
        
        # transforming the data
        TCholTable = focusCholTable[1:].groupby('id').agg(list).reset_index()
        TLdlTable = focusLdlTable[1:].groupby('id').agg(list).reset_index()
        print('biomarker tables are aggregated')
        #TPatientTable = focusPatientTable.groupby('id')

        # changing biomarkers columns
        biomarkercolumnname = ['id', 'duration','values']
        finalCholTable = TCholTable[biomarkercolumnname]
        finalLdlTable  = TLdlTable[biomarkercolumnname]
        print('biomarker table name are changed')
        
        # changing chol column names    
        finalCholTable.columns = ['NRIC_X', 'Time_Cholesterol', 'C_Cholesterol']
        finalLdlTable.columns = ['NRIC_X', 'Time_LDL', 'C_LDL']
        print('biomarker columns changed')

        # Making empty biomarkers 
        #hdlTriTable = pd.DataFrame({'NRIC_X': p, 'C_HDL': '' , 'Time_HDL': '' , 'C_Triglycerides': '' , 'Time_Triglycerides': ''}, index =[0])
        #ThdlTriTable = hdlTriTable.groupby('NRIC_X').agg(list).reset_index()
        #print('empty HDL Triglyceride table made')

        # Merging biomarker Tables
        biomarkerTable = pd.merge(finalCholTable, finalLdlTable, on = 'NRIC_X', how = 'outer')
        biomarkerTable['C_HDL'] = [[]]
        biomarkerTable['Time_HDL'] = [[]]
        biomarkerTable['C_Triglycerides'] = [[]]
        biomarkerTable['Time_Triglycerides'] = [[]]
        print('final biomarker table is merged and made')

        # Medications table 
        TbaselineMeds['start_d'] = [int(x) for x in TbaselineMeds['start_d']]
        TbaselineMeds['end_d'] = [int(x) for x in TbaselineMeds['end_d']]
        TfollowupMeds = TbaselineMeds.loc[TbaselineMeds['start_d'] >= Tbaselinestart]
        TfollowupMeds['TStartDay'] = TfollowupMeds['start_d'] - Tbaselinestart
        TfollowupMeds['TEndDayFinal'] = TfollowupMeds['end_d'] - Tbaselinestart
        print('medication is taken after baseline date')
  
        # Shift prescription start/end days 
        
        tempstart =  TfollowupMeds['TStartDay']
        tempend = TfollowupMeds['TEndDayFinal']

        for i in range(len(tempstart)-1):
            if i == (len(tempstart)-1):
                pass
            else:
                tempend[i] = (tempstart[i+1]-1)

        TfollowupMeds['StartDay'] = tempstart
        TfollowupMeds['EndDayFinal'] = tempend
        print('days shifted')

        #adherence calculation 

        tempadherence = []
        for i in range(len(TfollowupMeds['StartDay'])):
            adh1 = (TfollowupMeds['EndDayFinal'][i] - TfollowupMeds['StartDay'][i]+1)

            if i == (len(TfollowupMeds['StartDay']) - 1):
                adh2 = (TfollowupMeds['EndDayFinal'][i]+1) - TfollowupMeds['StartDay'][i]
                adhc = adh1/adh2
                tempadherence.append(adhc)
            else:
                adh2 = (TfollowupMeds['StartDay'][i+1] - TfollowupMeds['StartDay'][i])
                adhc = adh1/adh2
                tempadherence.append(adhc)
        
        print('adherence is calculated')

        # new column for adherence
        TfollowupMeds['adherence'] = tempadherence
        print('adherence added to medication table')

        # agg to list
        followupMeds = TfollowupMeds.groupby('id').agg(list).reset_index()
        print('final medication table is made')

        # change table names 
        medCol = { 
            'id': 'NRIC_X', 
            'StartDay': 'StartDay',
            'EndDayFinal': 'EndDayFinal',
            'mapped_generic': 'Type',
            'daily_dosage': 'Dose',
            'adherence': 'Adherence'
        }

        # renaming columns
        followupMeds.columns = followupMeds.columns.map(medCol)

        # final Meds Table (with limited columns) 
        finalMedCol = ['NRIC_X', 'StartDay', 'EndDayFinal', 'Type', 'Dose', 'Adherence']
        finalFollowUpMedsTable = followupMeds[finalMedCol]
        print('final medication table change is made')

        # Baseline Table
        baselineinfoTable= {'NRIC_X': p,
                        'Gender': 'Female',
                        'Age': 21,
                        'Race': 'Chinese',
                        'Cholesterol': TbaselineChol,
                        'LDL': TbaselineLdl,
                        'HDL': np.nan,
                        'Triglycerides': np.nan,
                        'Creatinine': np.nan,
                        'Statin_Prior': TbaselineMedsstatus,
                        'ACE_Prior': False,
                        'ARB_Prior': False,
                        'CCB_Prior': False,
                        'LoopDiuretic_Prior': False,
                        'Statin_Prior_Type': TbaselineStatin,
                        'Statin_Prior_Dose': TbaselineStatinDose
        }

        # Export to DataFrame
        finalbaselineinfoTable = pd.DataFrame(baselineinfoTable, index = [0])
        print('baselinetable made')
    
        # MedTableExport
        biomarkerTable.to_pickle(fileconfig["biomarkersTable"])
        finalFollowUpMedsTable.to_pickle(fileconfig['medicationsTable'])     
        finalbaselineinfoTable.to_pickle(fileconfig["baselineTable"])
        
        print('pickling happened.')
        
        # lipid.main(patientlist)


if __name__ == '__main__':
    try:
        patientlist = ['15701',]
        patientfileconfig = '../config/lipidoptimizing.json'
        #loadingdata(fileconfig = patientfileconfig)
        cleaningdata(patientlist, fileconfig = patientfileconfig)
        lipid.main(patientlist, classConfigFile = patientfileconfig)
    
    except Exception as e: 
        print("Can't call main function. Error - {}").format(str(e))
