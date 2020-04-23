import pandas as pd
import numpy as np
import scipy
import os, sys, time, json, math
import matplotlib.pyplot as plt
import seaborn as sns
from functools import reduce
from os.path import join
from datetime import datetime

def main():
	'''This function creates the numpy array for the statin efficacy for all lipids
	Output files are saved into Data/Final/Efficacy/
	(these files are required in order to run lipidSimulation.py)
	'''
	try:
		ldl_efficacy = np.array([[0.625, 0.486, 0.647, 0.402, 0.63, 0.485],
								[5.94, 29.35, 26.67, 10.57, 3.39, 6.98],
								[0.7, 0.7, 0.7, 0.7, 0.7, 0.7]])
		
		tc_efficacy = np.array([[0.455, 0.348, 0.471, 0.284, 0.458, 0.347],
								[5.94, 29.35, 26.67, 10.57, 3.39, 6.98],
								[0.7, 0.7, 0.7, 0.7, 0.7, 0.7]])
		
		trig_efficacy = np.array([[0.434, 0.290, 0.456, 0.204, 0.439, 0.289],
								[5.94, 29.35, 26.67, 10.57, 3.39, 6.98],
								[0.7, 0.7, 0.7, 0.7, 0.7, 0.7]])
		
		hdl_efficacy = np.array([[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
								[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
								[0.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
		
		np.save('Holmusk/AmgenRR/Data/Final/Efficacy/ldl_efficacy.npy', ldl_efficacy)
		np.save('Holmusk/AmgenRR/Data/Final/Efficacy/tc_efficacy.npy', tc_efficacy)
		np.save('Holmusk/AmgenRR/Data/Final/Efficacy/trig_efficacy.npy', trig_efficacy)
		np.save('Holmusk/AmgenRR/Data/Final/Efficacy/hdl_efficacy.npy', hdl_efficacy)
	except Exception as e:
		# print 'There was some problem with the main function: {}'.format(e)
		raise