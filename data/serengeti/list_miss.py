import os
cont={}
miss_files=[]
for mode in os.listdir('img'):
	cont[mode]=0
	for animal in os.listdir(os.path.join('img',mode)):
		for  file in os.listdir(os.path.join('img',mode,animal)):
			if not(os.path.isfile(os.path.join('descriptions',mode,animal,file[:-4]+'.json'))):
				print(os.path.join('descriptions',mode,animal,file[:-4]+'.json'))
				miss_files.append(os.path.join('img',mode,animal,file))
				cont[mode]=cont[mode]+1
				#os.remove(os.path.join('img',mode,animal,file))
with open('deleted_files.txt','w') as tfile:
	tfile.write('\n'.join(miss_files))
for i in cont.keys():
	print(i,' ', cont[i])
