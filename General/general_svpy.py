class sv_model:
	def __init__(self,file_path,GUI=False):
		import sys,os
		import numpy as np 
		try:
			self.clear()
			print('Gathering CSV Data...')
			self.data = np.genfromtxt(file_path, delimiter=',',dtype=str)
			print('Creating project...')
			self.name = os.path.basename(file_path)
			self.data_manager = {'Paths'      :[],
								 'Contours'   :[],
								 'PolyData'   :[],
								 'Solids'     :[],
								 'VTK'        :[],
								 'Mesh'       :[],
								 'Simulation' :[],
								 'Path_Points':{},
								 'Path_Contours':{},
								 'Path_PolyData':{},
								 'Path_SampleData':{},
								 'Path_AlignedData':{},
								 'Path_Lofts':{}}
		except:
			print('File_path is inaccessible...')
			return 
		self.GUI = GUI
		self.path()
		self.contour()
		self.loft()
		self.solid()

	def __path__(self,sv_path,sv_path_name): #PASSING
		from sv import Path,GUI,Repository
		p = Path.pyPath()
		p.NewObject(sv_path_name)
		self.data_manager['Path_Points'][sv_path_name] = []
		for i in range(len(sv_path[:,0])):
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
		path_lengths = self.__path_lengths__(self.data[:,ind])[1:]
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
		self.data_manager['Path_Contours'][path_object].append('C_'+path_object+'_'+str(slice_index))
		c.GetPolyData('C_'+path_object+'_'+str(slice_index)+'p')
		self.data_manager['PolyData'].append('C_'+path_object+'_'+str(slice_index)+'p')
		self.data_manager['Path_PolyData'][path_object].append('C_'+path_object+'_'+str(slice_index)+'p')
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
			path_contour_list.append(self.__contour_subfunction__(path_object,slice_index[i],float(r[i])))
		if self.GUI == True:
			GUI.ImportContoursFromRepos('Contours_'+path_object,path_contour_list,path_object,'Segmentations')
		return 


	def contour(self):  #PASSING 
		for path_object in self.data_manager['Paths']:
			self.data_manager['Path_Contours'][path_object] = []
			self.data_manager['Path_PolyData'][path_object] = []
			self.__contour_path__(path_object)		
		pass


	def __geometry__(self,path_object,spline=True,NumSegs=60):
		from sv import Geom,GUI,Solid
		import math
		for PolyData in self.data_manager['Path_PolyData'][path_object]:
			Geom.SampleLoop(PolyData,NumSegs,PolyData+'s')
			self.data_manager['Path_SampleData'][path_object].append(PolyData+'s')

		for index in range(len(self.data_manager['Path_PolyData'][path_object])-1):
			if index == 0:
				Geom.AlignProfile(self.data_manager['Path_SampleData'][path_object][index],self.data_manager['Path_SampleData'][path_object][index+1],path_object+'alignment'+str(index),0)
				self.data_manager['Path_AlignedData'][path_object].append(self.data_manager['Path_SampleData'][path_object][index])
				self.data_manager['Path_AlignedData'][path_object].append(path_object+'alignment'+str(index))
			else:
				Geom.AlignProfile(self.data_manager['Path_AlignedData'][path_object][index],self.data_manager['Path_SampleData'][path_object][index+1],path_object+'alignment'+str(index),0)
				self.data_manager['Path_AlignedData'][path_object].append(path_object+'alignment'+str(index))
		print(self.data_manager['Path_AlignedData'][path_object])
		if spline == True:
			Geom.LoftSolid(self.data_manager['Path_AlignedData'][path_object],path_object+'_loft',60,120,100,20,0,1)
		else:
			pass #will have nurbs lofting later 
		if self.GUI == True:
			pass
		#	GUI.ImportPolyDataFromRepos(path_object+'_loft','Models') #something is wrong with the lofting 
		else:
			pass
		return 

	def loft(self):	
		for path_object in self.data_manager['Paths']:
			self.data_manager['Path_SampleData'][path_object] = []
			self.data_manager['Path_Lofts'][path_object] = []
			self.data_manager['Path_AlignedData'][path_object] = []
			self.__geometry__(path_object)
		return 

	def __solid_subprocess__(self,path_object):
		from sv import Solid,Repository,VMTKUtils
		Solid.SetKernel('PolyData')
		solid = Solid.pySolidModel()
		solid.NewObject(path_object+'_solid')
		VMTKUtils.Cap_with_ids(path_object+'_loft',path_object+'_capped',0,0)
		solid.SetVtkPolyData(path_object+'_capped')
		self.data_manager['Solids'].append(path_object+'_capped')
		solid.GetBoundaryFaces(90)
		faceids = solid.GetFaceIds()

	def __Union__(self):
		from sv import Geom 
		Geom.All_union(self.data_manager['Solids'],len(self.data_manager['Solids']),'Model',0.00001)
		pass

	def __subtraction__(self):
		from sv import Solid
		temp = Solid.pySolidModel()
		temp.Subtract('temp',self.data_manager['Solids'][0],self.data_manager['Solids'][1])
		# temp.GetPolyData('temp_polydata')
		return 

	def solid(self):
		for path_object in self.data_manager['Paths']:
			self.__solid_subprocess__(path_object)
		pass
		#need to mirror solid generation in GUI
		#solids created from current lofting and capping aproach
		# are not robust and fail during meshing or smoothing
		# contours are fine. 
	def smooth():
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
			if (path_vector[i].isspace()==False and path_vector[i] != ''):
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