from pyhdf.SD import *
import numpy as np
import cv2 
import Image, ImageDraw
import sys

sys.setrecursionlimit(50000)

class Reader:
    def getarray (self):
        return self.image

class HDFReader(Reader):
    def __init__(self, filename):
        hdfFile = SD(filename)
        bo = re.compile('.*?=\s+([-]*\d*\.\d+).*',re.DOTALL)
        coord_string = hdfFile.__getattr__('ArchiveMetadata.0')[hdfFile.__getattr__('ArchiveMetadata.0').find('NORTHBOUNDINGCOORDINATE')]
        self.west = float(bo.match(coord_string.split('WESTBOUNDINGCOORDINATE')[1]).group(1))
        self.east = float(bo.match(coord_string.split('EASTBOUNDINGCOORDINATE')[1]).group(1))
        self.south = float(bo.match(coord_string.split('SOUTHBOUNDINGCOORDINATE')[1]).group(1))
        self.north = float(bo.match(coord_string.split('NORTHBOUNDINGCOORDINATE')[1]).group(1))
        tmp = np.array (hdfFile.select(1)[:], dtype = np.int16)
        emp = hdfFile.getfillvalue()
        tmp[tmp == emp] = 0
        min = tmp.min()
        rng = tmp.max() - min
        self.image = np.array(float(tmp-min)/rng*255, dtype=np.uint8)
        hdfFile.end()

class ImageReader(Reader): #png, jpeg, bmp
    def __init__(self, filename):
        im = cv2.imread(filename);
        if len(im.shape) == 1: 
            self.image = im
        else:
            self.image = im[:,:,0]

class Writer:
    def save(self, filename):
        self.image.save(filename)


class ArrayWriter(Writer):
    def __init__(self, arry):
        arry[np.where(arry >0)] = 255
        self.image = Image.fromarray(arry)


class GraphicsFilters:
    @classmethod
    def opening(self, data, area, bias, newdata):
       # (dx,dy) = area
        lx = len(data[0])
        ly = len(data)

        fu = np.vstack([data[1:,:],np.zeros((1,lx))])
        fd = np.vstack([np.zeros((1,lx)),data[:-1,:]])
        fl = np.hstack([data[:,1:],np.zeros((ly,1))])
        fr = np.hstack([np.zeros((ly,1)),data[:,:-1]])
        flu = np.vstack([fl[1:,:],np.zeros((1,lx))])
        frd = np.vstack([np.zeros((1,lx)),fr[:-1,:]])
        fdl = np.hstack([fd[:,1:],np.zeros((ly,1))])
        fur = np.hstack([np.zeros((ly,1)),fu[:,:-1]])

        tmask = fu + fd + fl + flu + frd + fdl + fur
        newdata[np.where(tmask>bias)] = 1

        

        #for x in range(dx,len(data[0])-dx):
        #    for y in range(dy,len(data)-dy):
        #        a = data[y-dy:y+dy+1,x-dx:x+dx+1]
        #        if len(np.where(a==1)[0]) >= bias:
        #            newdata[y,x] = 1

    @classmethod
    def closing(self, data, area, bias, newdata):
        lx = len(data[0])
        ly = len(data)

        fu = np.vstack([data[1:,:],np.zeros((1,lx))])
        fd = np.vstack([np.zeros((1,lx)),data[:-1,:]])
        fl = np.hstack([data[:,1:],np.zeros((ly,1))])
        fr = np.hstack([np.zeros((ly,1)),data[:,:-1]])
        flu = np.vstack([fl[1:,:],np.zeros((1,lx))])
        frd = np.vstack([np.zeros((1,lx)),fr[:-1,:]])
        fdl = np.hstack([fd[:,1:],np.zeros((ly,1))])
        fur = np.hstack([np.zeros((ly,1)),fu[:,:-1]])

        tmask = fu + fd + fl + flu + frd + fdl + fur
        newdata[np.where(tmask<bias)] = 0

    @classmethod
    def deletecluster(self, data, mask, bias, num, celllist):
        if len(celllist) > bias:
            return bias
        if len(celllist)==0:
           return num
        (x,y) = celllist.pop()
        if 0<=x-1<len(data[0]) and 0<=y<len(data) and data[y,x-1] == 1 and mask[y,x-1] == 0 :
           celllist.append((x-1,y))
           mask[y,x-1] = 1

        if 0<=x+1<len(data[0]) and 0<=y<len(data) and data[y,x+1] == 1 and mask[y,x+1] == 0 :
           celllist.append((x+1,y))
           mask[y,x+1] = 1

        if 0<=x<len(data[0]) and 0<=y-1<len(data) and data[y-1,x] == 1 and mask[y-1,x] == 0 :
           celllist.append((x,y-1))
           mask[y-1,x] = 1

        if 0<=x<len(data[0]) and 0<=y+1<len(data) and data[y+1,x] == 1 and mask[y+1,x] == 0 :
           celllist.append((x,y+1))
           mask[y+1,x] = 1

        tnum = self.deletecluster(data,mask,bias,num+1,celllist)
        if tnum < bias :
           data[y,x] = 0
        return tnum

    @classmethod
    def deletesmallcluster(self, data, bias):
        mask = np.zeros(data.shape)
        for x in range(0,len(data[0])):
            print "start", x
            for y in range(0,len(data)):
                if data[y,x] == 1 and mask[y,x] == 0 :
                    mask[y,x] = 1
                    self.deletecluster(data,mask,bias,0,[(x,y)])


class ImageThining:
    def __init__(self):
        self.pat_w = [
                np.array([[0.,0.,0.],[0.,1.,1.],[0.,1.,0.]]),
                np.array([[0.,0.,0.],[0.,1.,0.],[1.,1.,0.]]),
                np.array([[0.,0.,0.],[1.,1.,0.],[0.,1.,0.]]),
                np.array([[1.,0.,0.],[1.,1.,0.],[0.,0.,0.]]),
                np.array([[0.,1.,0.],[1.,1.,0.],[0.,0.,0.]]),
                np.array([[0.,1.,1.],[0.,1.,0.],[0.,0.,0.]]),
                np.array([[0.,1.,0.],[0.,1.,1.],[0.,0.,0.]]),
                np.array([[0.,0.,0.],[0.,1.,1.],[0.,0.,1.]]),
               ] 
        self.pat_b = [
                np.array([[1.,1.,0.],[1.,0.,0.],[0.,0.,0.]]),
                np.array([[1.,1.,1.],[0.,0.,0.],[0.,0.,0.]]),
                np.array([[0.,1.,1.],[0.,0.,1.],[0.,0.,0.]]),
                np.array([[0.,0.,1.],[0.,0.,1.],[0.,0.,1.]]),
                np.array([[0.,0.,0.],[0.,0.,1.],[0.,1.,1.]]),
                np.array([[0.,0.,0.],[0.,0.,0.],[1.,1.,1.]]),
                np.array([[0.,0.,0.],[1.,0.,0.],[1.,1.,0.]]),
                np.array([[1.,0.,0.],[1.,0.,0.],[1.,0.,0.]]),
               ]

    def thining(self, data):
        pass


class RivFinder:
    def __init__(self,img):
        self.img = img
        self.aimg = img

    def filterimage(self):
        im = self.img.copy()
        nim = self.img.copy()

        for i in range(0, 48):
            GraphicsFilters.closing(im, (1,1), 3, nim)
            im = nim
            print i ,"opening1"

        for i in range(0, 48):
            GraphicsFilters.closing(im, (1,1), 4, nim)
            im = nim
            print i , "opening2"

        for i in range(0, 12):
            GraphicsFilters.opening(im, (1,1), 6, nim)
            im = nim
            print i , "closing"

       #a GraphicsFilters.deletesmallcluster(im,1000)
        return im




if __name__ == "__main__":
    reader = ImageReader("data/a.png")
    img = reader.getarray()
    img[np.where(img < 70 )] = 0
    img[np.where(img >= 70 )] = 1
    rf = RivFinder(img)
    ans = rf.filterimage()
    writer = ArrayWriter(ans)
    writer.save("data/b.png")


