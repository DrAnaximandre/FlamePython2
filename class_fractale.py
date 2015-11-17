from utils import *
import numpy as np
from PIL import Image,ImageFilter
import sys
import time

class Function:
	def __init__(self,ws,params,additives):
		self.ws=[ws]
		self.params=params
		self.additives=[additives]

	def call(self,toto):
		x_loc=np.dot(toto,self.params[0:3])
		y_loc=np.dot(toto,self.params[3:6])
		tostr=''
		for i in range(len(self.additives)):
			tostr= tostr+'self.ws['+str(i)+']*'+self.additives[i]+'(x_loc,y_loc)'
		return(eval(tostr))


class Variation:

	def __init__(self):
		self.Nfunctions=0
		self.functions=[]
		self.vproba=[0]
		self.cols=[]
		self.lockVariation=False
		self.rotation=[]
		self.final=False

	def addRandomFunction(self):
		lis=['linear','swirl','spherical','expinj','bubble','pdj']
		if not self.lockVariation:
			self.Nfunctions+=1
			col=np.random.randint(0,255,3)
			self.cols.append(col)
			proba=np.random.rand(1)
			self.vproba.append(proba)
			params=np.random.uniform(-1,1,(6,1))
			r=np.random.randint(0,6,1)
			additives=lis[r]
			ws=np.random.uniform(-1,1,1)
			self.functions.append(Function(ws,params,additives))
		else:
			print("This varaition is locked, I can't do anything")


	def addFunction(self,ws,params,additives,proba,col):
		if not self.lockVariation:
			self.Nfunctions+=1
			self.cols.append(col)
			self.vproba.append(proba)
			self.functions.append(Function(ws,params,additives))
		else:
			print("This varaition is locked, I can't do anything")

	def addFinal(self,ws,params,additives):
		if not self.lockVariation:
			self.final=Function(ws,params,additives)
		else:
			print("This varaition is locked, I can't do anything")

	def addRotation(self,angle):
		if not self.lockVariation:
			if angle not in [180,120,90]:
				print("Angle not implemented yet")
			else:
				self.rotation.append(angle)
		else:
			print("This varaition is locked, I can't do anything")


	def fixProba(self):
		self.vproba=[p/np.sum(self.vproba) for p in self.vproba]
		self.vproba=np.cumsum(self.vproba)
		self.lockVariation=True

	def runAllfunctions(self,totoF,totoC):
		Nloc=totoF.shape[0]
		r=np.random.uniform(size=Nloc)
		resF=np.zeros(shape=(Nloc,2))
		resC=np.zeros(shape=(Nloc,3))
		for i in range(len(self.vproba)-1):
			masku1=r>self.vproba[i]
			masku2=r<self.vproba[i+1]
			sel=np.where((masku1) & (masku2))[0]
			resF[sel,:]=self.functions[i].call(totoF[sel,:])
			colorbrew=np.ones(shape=(len(sel),3))*self.cols[i]
			resC[sel,:]=(totoC[sel,:]+colorbrew)/2

		if self.final:
			onesfinal=np.ones(resF.shape[0])
			totoF=np.column_stack((onesfinal,resF))
			resF=self.final.call(totoF)
		else:
			pass
		return(resF,resC)

	def runAllrotations(self,resF):
		Nloc=resF.shape[0]
		r=np.random.uniform(size=Nloc)
		for i in range(len(self.rotation)):
			if self.rotation[i]==120:
				a120=np.pi*2/3
				rot120=np.matrix([[np.cos(a120),np.sin(a120)],[-np.sin(a120),np.cos(a120)]])
				sel1=np.where(r<.33)[0]
				sel2=np.where(r>.66)[0]
				resF[sel1,:]=np.dot(resF[sel1,:],rot120)
				resF[sel2,:]=np.dot(np.dot(resF[sel2,:],rot120),rot120)

			elif self.rotation[i]==180:
				a180=np.pi
				rot180=np.matrix([[np.cos(a180),np.sin(a180)],[-np.sin(a180),np.cos(a180)]])
				sel1=np.where(r<.5)[0]
				resF[sel1,:]=np.dot(resF[sel1,:],rot180)
				
			elif self.rotation[i]==90:
				a90=np.pi/2
				rot90=np.matrix([[np.cos(a90),np.sin(a90)],[-np.sin(a90),np.cos(a90)]])
				sel=np.where(r<.25)[0]
				resF[sel,:]=np.dot(resF[sel,:],rot90)
				sel=np.where((r<.5) & (r>.25))[0]
				resF[sel,:]=np.dot(np.dot(resF[sel,:],rot90),rot90)
				sel=np.where((r<.75) & (r>.5))[0]
				resF[sel,:]=np.dot(np.dot(np.dot(resF[sel,:],rot90),rot90),rot90)

			# # else:
			# 	aloc=self.rotation[i]*np.pi/180
			# 	rotloc=np.matrix([[np.cos(aloc),np.sin(aloc)],[-np.sin(aloc),np.cos(aloc)]])
			# 	sel1=np.where(r<.5)[0]
			# 	resF[sel1,:]=np.dot(resF[sel1,:],rotloc)
				

		return(resF)

class Fractale:

	def __init__(self,burn,niter,zoom=1):
		self.zoom=zoom
		self.variations=[]
		self.Ns=[]
		self.burn=burn
		self.niter=niter
		self.lockBuild=False

	def addVariation(self,var,N):
		self.variations.append(var)
		self.Ns.append(N)

	def build(self):
		'''
			it is not advised to add variations after a build

		'''
		if not self.lockBuild:
			totalSize=np.sum(self.Ns)*self.niter
			self.F=np.random.uniform(-1,1,size=(totalSize,2))
			self.C=np.ones(shape=(totalSize,3))*255
			[v.fixProba() for v in self.variations]
			self.lockBuild=True

		else:
			print("You have already built this Fractale")

	def run1iter(self,whichiter,burn):
		sumNS=np.sum(self.Ns)
		a=sumNS*whichiter
		b=sumNS*(whichiter+1)
		c=sumNS*(whichiter+2)
		rangeIdsIN=np.arange(a,b)
		if burn:
			rangeIdsOUT=np.arange(a,b)
		else:
			rangeIdsOUT=np.arange(b,c)

		## safety check
		if len(rangeIdsIN)!=sumNS:
			print("the number of indices provided is different from the number of points in one image")
			sys.exit()

		ones=np.ones(len(rangeIdsIN))
		totoF=np.column_stack((ones,self.F[rangeIdsIN,:]))
		totoC=self.C[rangeIdsIN,:]

		for i in range(len(self.variations)):
			if i==0:
				idstoto=np.arange(self.Ns[0])
			else:
				idstoto=np.arange(sum(self.Ns[:i]),sum(self.Ns[:i])+self.Ns[i])

			resloc,coloc=self.variations[i].runAllfunctions(totoF[idstoto,:],totoC[idstoto,:])
			self.C[rangeIdsOUT[idstoto],:]=coloc
			self.F[rangeIdsOUT[idstoto],:]=self.variations[i].runAllrotations(resloc)


	def runAll(self):
		sumNS=np.sum(self.Ns)
		for i in np.arange(self.burn):
			self.run1iter(0,True)
		for i in np.arange(self.niter-1):
			self.run1iter(i,False)

		self.F=self.F*self.zoom


	def toImagerec(self,ids=None,sizeImage=800):
		imgtemp = Image.new( 'RGB', (sizeImage,sizeImage), "black")
		bitmap = np.array(imgtemp)
		intensity=np.zeros((sizeImage,sizeImage,3))        
		
		F_loc=(sizeImage*(self.F+1)/2).astype("i2")
		C_loc=self.C


		goods=np.where((F_loc[:,0]<sizeImage) & (F_loc[:,0]>0) & (F_loc[:,1]<sizeImage) & ( F_loc[:,1]>0))[0]
		print("         number of points in the image: "+str(len(goods)))

		def affectrec(F_loc_loc,C_loc_loc,rangei, rangej,depth):
			if F_loc_loc.shape[0]<=2:
				pass
			else:
				if len(rangei)<=2 and len(rangej)<=2:
					N_loc=F_loc_loc.shape[0]
					intensity[rangei[0],rangej[0]]=N_loc
					nbiz=(np.arange(N_loc)+1)/(N_loc+1)
					nbizbiz=np.rollaxis(np.tile(nbiz,(3,1)),1)
					a=np.mean(C_loc_loc*nbizbiz,axis=0)
					bitmap[rangei[0],rangej[0]]=a
				else:
					nri0=rangei[:int(len(rangei)/2+1)]
					nri1=rangei[int(len(rangei)/2):]
					nrj0=rangej[:int(len(rangej)/2+1)]
					nrj1=rangej[int(len(rangej)/2):]
					for i in [nri0,nri1]:
						for j in [nrj0,nrj1]:
							mask0=((F_loc_loc[:,0] >= i[0]) & (F_loc_loc[:,0] <= i[-1]))
							mask1=((F_loc_loc[:,1] >= j[0]) & (F_loc_loc[:,1] <= j[-1]))
							mask=np.where((mask0) & (mask1))[0]
							affectrec(F_loc_loc[mask,:],C_loc_loc[mask,:],i,j,depth+1)

		affectrec(F_loc[goods,:],C_loc[goods,:],np.arange(sizeImage+1),np.arange(sizeImage+1),0)				

		nmax=np.amax(intensity)
		print(nmax)
		intensity=np.sqrt(np.log(intensity+1)/np.log(nmax+1))
		bitmap=np.uint8(bitmap*intensity)
		out=Image.fromarray(bitmap)
		print("starting Kernel smoothing")
		# supsammpK=ImageFilter.Kernel((5,5),[1,1,1,1,1,1,4,8,4,1,3,8,15,8,3,1,4,8,4,1,1,1,3,1,1])
		supsammpK=ImageFilter.Kernel((3,3),[1,3,1,3,4,3,1,3,1])
		out=out.filter(supsammpK)
		return(out)


	def toScore(self,sizeImage=1200,coef=1,p=6):
		F_loc=(sizeImage*(self.F+1)/2).astype("i2")

		goods=np.where((F_loc[:,0]<sizeImage) & (F_loc[:,0]>0) & (F_loc[:,1]<sizeImage) & ( F_loc[:,1]>0))[0]
		print("         number of points in the image: "+str(len(goods)))

		scoreOut=[]
		scoreIn=[]
		rangei=np.linspace(0,sizeImage,num=p)
		rangej=rangei
		for i in range(p-1):
			for j in range(p-1):
				locs=np.where((F_loc[goods,0]<rangei[i+1]) &
				 	(F_loc[goods,0]>rangei[i]) &
				  	(F_loc[goods,1]<rangej[j+1]) &
				  	( F_loc[goods,1]>rangej[j]))[0]
				scoreOut.append(len(locs))
				rangeiloc=np.linspace(rangei[i],rangei[i+1],3)
				rangejloc=np.linspace(rangej[j],rangej[j+1],3)
				for k in range(2):
					for l in range(2):
						loc_loc=np.where((F_loc[goods,0]<rangeiloc[k+1]) &
									 	(F_loc[goods,0]>rangeiloc[k]) &
									  	(F_loc[goods,1]<rangejloc[l+1]) &
									  	( F_loc[goods,1]>rangejloc[l]))[0]
						scoreIn.append(len(loc_loc))

		print(np.std(scoreOut))
		print(np.std(scoreIn))
		return(-np.std(scoreOut)+np.std(scoreIn))


	def toImage(self,sizeImage=1200,coef=1):
		imgtemp = Image.new( 'RGB', (sizeImage,sizeImage), "black")
		bitmap = np.array(imgtemp)
		intensity=np.zeros((sizeImage,sizeImage,3))        
		
		F_loc=(sizeImage*(self.F+1)/2).astype("i2")
		C_loc=self.C


		goods=np.where((F_loc[:,0]<sizeImage) & (F_loc[:,0]>0) & (F_loc[:,1]<sizeImage) & ( F_loc[:,1]>0))[0]
		print("         number of points in the image: "+str(len(goods)))

		for i in goods:
			a=(C_loc[i,:]*coef+bitmap[F_loc[i,0],F_loc[i,1]])/(coef+1)
			bitmap[F_loc[i,0],F_loc[i,1]]=a
			intensity[F_loc[i,0],F_loc[i,1],:]+=1

		nmax=np.amax(intensity)
		print(nmax)
		intensity=np.power(np.log(intensity+1)/np.log(nmax+1),.2)
		bitmap=np.uint8(bitmap*intensity)
		out=Image.fromarray(bitmap)
		print("starting Kernel smoothing")
		supsammpK=ImageFilter.Kernel((3,3),[1,3,1,3,5,3,1,3,1])
		out=out.filter(supsammpK)
		return(out)


if __name__=='__main__':
	N=50000
	F1=Fractale(burn=10,niter=20,zoom=1)
	v1=Variation()
	v1.addFunction(.9,[0,1,0,0,0,1],'linear',.2,[255,0,0])
	v1.addFunction(.5,[1,1,0,0,0,1],'linear',.2,[0,255,0])
	v1.addFunction(.5,[0,1,0,1,0,1],'linear',.2,[0,0,255])
	v1.addFunction(.4,[-.3,1,1,-.4,.2,0],"spherical",.1,[255,255,255])
	v1.addRotation(120)


	F1.addVariation(v1,N)
	F1.build()
	F1.runAll()
	print("goto image")
	out=F1.toImage()
	out.save("figure.png")