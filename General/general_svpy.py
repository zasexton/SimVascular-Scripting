class sv_model:
	def __init__(self,file_path,GUI=False):
		import sys,os,pandas
		try:
			print('Gathering CSV Data...')
			data = pandas.read_csv(file_path)#importing csv type data using pandas module
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
			print('file_path is inaccessible...')

	def path(self,sv_path,sv_path_name):
		from sv import Path,GUI,Repository
		p = Path.pyPath()
		p.NewObject(sv_path_name)
		for i in range(len(sv_path[:][0])):
			p.AddPoint(sv_path[i][:])
		p.CreatePath()
		self.data_manager{'Paths'}.append(sv_path_name)
		if self.GUI = True:
			GUI.ImportPathFromRepos(sv_path_name,'Paths')
	def contour(self,path_object,slice_index):
		from sv import Contour,GUI
		Contour.SetContourKernel('Circle')
		c = Contour.pyContour()
		c.NewObject('path_object'+str(slice_index),path_object,slice_index)
		
	def geometry():
		pass 
	def solid():
		pass
	def mesh():
		pass 
	def pre():
		pass 
	def sim():
		pass
	def post():
		pass
