from sklearn import cluster,datasets
import numpy as np
import os
from skimage.feature import hog
import matplotlib.pyplot as plt
import glob
from skimage import io
from skimage.transform import resize

def loading_data(path,extention="*.png"):
    imageTitleList=[]
    imgMatrix=[]
    i=0
    allImage = glob.glob(path + extention)
    print(allImage)
    HogFeatures=[]
    for image in allImage:
        img=io.imread(image)
        imgName = os.path.basename(image)
        imgTitle = os.path.splitext(imgName)[0]
        imageTitleList.append(imgTitle)
        imgMatrix.append(img)
        # print(imgTitle)
        img=resize(img,(200,200))
        fd, hog_image = hog(img, orientations=8, pixels_per_cell=(16, 16),
                            cells_per_block=(1, 1), visualise=True)

        HogFeatures.append(fd)
    x_train=np.array(HogFeatures)
    imgMatrix=np.array(imgMatrix)
    imageTitleList=np.array(imageTitleList)
    return x_train,imgMatrix,imageTitleList




SourcePath=""
X,imgMatrix,titleList=loading_data(SourcePath)
clf=cluster.KMeans(n_clusters=5)
clf.fit(X)
labels=clf.labels_
centroid=clf.cluster_centers_
#print(centroid)
labels=clf.labels_
#print(labels)
color=["g.","r.","y.","c.","k.","o."]
path=""
for i in range(len(imgMatrix)):
    if labels[i]== 0:
        io.imsave(path+"0/"+str(titleList[i]+".png"),imgMatrix[i])
    elif labels[i] == 1:
        io.imsave(path+"1/"+str(titleList[i]+".png"),imgMatrix[i])
    elif labels[i] == 2:
        io.imsave(path+"2/"+str(titleList[i]+".png"),imgMatrix[i])
    elif labels[i] == 3:
        io.imsave(path+"3/"+str(titleList[i]+".png"),imgMatrix[i])
    elif labels[i] == 4:
        io.imsave(path+"4/"+str(titleList[i]+".png"),imgMatrix[i])

for i in range(len(X)):
    plt.plot(X[i][0],X[i][1],color[labels[i]],markersize=30)
    plt.scatter(centroid[:,0],centroid[:,1],marker="x",s=150,linewidths=5)
plt.show()
