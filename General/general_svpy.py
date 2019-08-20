class sv_model:
	def __init__(self,file_path,GUI=False):
		import sys,os
		try:
			import pandas
		except:
			sys.path.append('/usr/local/lib/python3.5/dist-packages')
			sys.path.append('/usr/lib/python3/dist-packages')
			sys.path.append('/home/zacharysexton/.local/lib/python3.5/site-packages')
			import pandas
		try:
			print('Gathering CSV Data...')
			self.data = pandas.read_csv(file_path,header=None)#importing csv type data using pandas module
			print('Creating project...')
			self.name = os.path.basename(file_path)
			self.data_manager = {'Paths'     :[],
								 'Contours'  :[],
								 'PolyData'  :[],
								 'Solids'    :[],
								 'VTK'       :[],
								 'Mesh'      :[],
								 'Simulation':[]}
		except:
			print('File_path is inaccessible...')
			return 
		self.GUI = GUI
		self.data = self.data.get_values()

	def __path__(self,sv_path,sv_path_name):
		from sv import Path,GUI,Repository
		p = Path.pyPath()
		p.NewObject(sv_path_name)
		for i in range(len(sv_path[:,0])):
			print(sv_path[i][:])
			temp = []
			for j in sv_path[i][:]:
				temp.append(float(j))
			p.AddPoint(temp)
		p.CreatePath()
		self.data_manager['Paths'].append(sv_path_name)
		if self.GUI == True:
			GUI.ImportPathFromRepos(sv_path_name,'Paths')

	def path(self):
		import numpy as np 
		ind = int(np.nonzero(self.data[0,:]=='Path')[0]) #will be a limitation later
		print(self.data[:,ind])
		path_lengths = self.__path_lengths__(self.data[:,ind])[1:]
		print(path_lengths)
		for i in range(len(path_lengths)-1):
			self.__path__(self.data[path_lengths[i]:path_lengths[i+1],1:4],self.data[path_lengths[i],ind])

	def __contour__(self,path_object,slice_index):
		from sv import Contour,GUI
		import numpy as np 
		Contour.SetContourKernel('Circle')
		c = Contour.pyContour()
		c.NewObject('path_object'+str(slice_index),path_object,slice_index)
		
	def __geometry__():
		pass 


	def __solid__():
		pass


	def mesh():
		pass 


	def pre():
		pass


	def sim():
		pass


	def post():
		pass

	def __path_lengths__(self,path_vector):
		temp = []
		for i in range(len(path_vector)):
			if path_vector[i].isspace()==False:
				temp.append(i)
			elif i == len(path_vector)-1:
				temp.append(i+1)
			else:
				pass 
		return temp