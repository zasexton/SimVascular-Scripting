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
			self.clear()
			print('Gathering CSV Data...')
			self.data = pandas.read_csv(file_path,header=None)#importing csv type data using pandas module
			print('Creating project...')
			self.name = os.path.basename(file_path)
			self.data_manager = {'Paths'      :[],
								 'Contours'   :[],
								 'PolyData'   :[],
								 'Solids'     :[],
								 'VTK'        :[],
								 'Mesh'       :[],
								 'Simulation' :[],
								 'Path_Points':{}}
		except:
			print('File_path is inaccessible...')
			return 
		self.GUI = GUI
		self.data = self.data.get_values()

	def __path__(self,sv_path,sv_path_name): #PASSING
		from sv import Path,GUI,Repository
		p = Path.pyPath()
		p.NewObject(sv_path_name)
		self.data_manager['Path_Points'][sv_path_name] = []
		for i in range(len(sv_path[:,0])):
			print(sv_path[i][:])
			temp = []
			for j in sv_path[i][:]:
				temp.append(float(j))
			self.data_manager['Path_Points'][sv_path_name].append(temp)
			p.AddPoint(temp)
		p.CreatePath()
		self.data_manager['Paths'].append(sv_path_name)
		if self.GUI == True:
			GUI.ImportPathFromRepos(sv_path_name,'Paths')

	def path(self): #PASSING
		import numpy as np 
		self.path_radii = {}
		ind = int(np.nonzero(self.data[0,:]=='Path')[0]) #will be a limitation later
		print(self.data[:,ind])
		path_lengths = self.__path_lengths__(self.data[:,ind])[1:]
		print(path_lengths)
		for i in range(len(path_lengths)-1):
			self.__path__(self.data[path_lengths[i]:path_lengths[i+1],1:4],self.data[path_lengths[i],ind])
			self.path_radii[self.data[path_lengths[i],ind]] = self.data[path_lengths[i]:path_lengths[i+1],4] #the radius index is not always garunteed to be in column 4

	def __contour_subfunction__(self,path_object,slice_index,radius):  #PASSING
		from sv import Contour,GUI
		import numpy as np 
		Contour.SetContourKernel('Circle')
		c = Contour.pyContour()
		c.NewObject('C_'+path_object+'_'+str(slice_index),path_object,slice_index)
		c.SetCtrlPtsByRadius(self.data_manager['Path_Points'][path_object][int(slice_index/25)],radius) #change later
		c.Create()
		self.data_manager['Contours'].append('C_'+path_object+'_'+str(slice_index))
		c.GetPolyData('C_'+path_object+'_'+str(slice_index)+'p')
		self.data_manager['PolyData'].append('C_'+path_object+'_'+str(slice_index)+'p')
		return 'C_'+path_object+'_'+str(slice_index)

	def __contour_path__(self,path_object,slices=None):
		from sv import GUI,Path
		path_contour_list = []
		if slices==None:
			r = self.path_radii[path_object]
			slices = len(r)
			p = Path.pyPath()
			p.GetObject(path_object)
			slice_index = []
			object_path = p.GetPathPosPts()
			for path_point in self.data_manager['Path_Points'][path_object]:
				slice_index.append(object_path.index(path_point))
		else:
			#will require interpolation
			pass 
		print(slice_index)
		for i in range(slices):
			path_contour_list.append(self.__contour_subfunction__(path_object,slice_index[i],int(r[i])))
		if self.GUI == True:
			GUI.ImportContoursFromRepos('Contours_'+path_object,path_contour_list,path_object,'Segmentations')
		return 


	def contour(self):  #PASSING 
		for i in self.data_manager['Paths']:
			self.__contour_path__(i)		
		pass


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

	def __linear_interp__(first_value,second_value,first_slice,second_slice):
		pass

	def clear(self):
		from sv import Repository
		if len(Repository.List()) == 0:
			print('Repository Empty')
		else:
			for i in Repository.List():
				Repository.Delete(i)
			print('Repository Cleared')
			for key in self.data_manager.keys():
				self.data_manager[key] = []
			print('Data Manager Cleared')
		return