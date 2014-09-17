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
    def __sum8(self,data):
        fu = np.vstack([data[1:,:],data[ -1,:]])
        fd = np.vstack([data[0 ,:],data[:-1,:]])
        fl = np.hstack([data[:,1:],data[:,-1:]])
        fr = np.hstack([data[:,:1],data[:,:-1]])
        flu = np.vstack([fl[1:,:],fl[ -1,:]])
        frd = np.vstack([fr[0 ,:],fr[:-1,:]])
        fdl = np.hstack([fd[:,1:],fd[:,-1:]])
        fur = np.hstack([fu[:,:1],fu[:,:-1]])
        tmask = fu + fd + fl + fr + flu + frd + fdl + fur
        return tmask

    @classmethod
    def opening(self, data, bias):
        tmask = self.__sum8(data)
        newdata = np.array(data)
        newdata[np.where(tmask>bias)] = 1
        return newdata

    @classmethod
    def closing(self, data, bias):
        tmask = self.__sum8(data)
        newdata = np.array(data)
        newdata[np.where(tmask<=bias)] = 0
        return newdata

    @classmethod
    # -1 : now searching
    # 0 : no signal
    # 1 : signal
    # 2 : a part of big cluster
    def __deletecluster(self, flags, bias, num, celllist):

        #End the search
        if len(celllist)==0:
            if num > 100 : print num
            return num

        (x,y) = celllist.pop()

        #Cut the search
        if num >= bias or flags[y,x]==2:
            return bias

        flags[y,x]=-1

        if flags[y,x-1] > 0:
            celllist.append((x-1,y))
        if flags[y-1,x] > 0:
            celllist.append((x,y-1))
        if flags[y,x+1] > 0:
            celllist.append((x+1,y))
        if flags[y+1,x] > 0:
            celllist.append((x,y+1))

        tnum = self.__deletecluster(flags,bias,num+1,celllist)

        if tnum < bias:
            flags[y,x] = 0
        else:
            flags[y,x] = 2

        return tnum

    @classmethod
    def deletecluster(self, mask, bias):
        lx = len(mask[0])
        ly = len(mask)
        flags = np.zeros((ly+2,lx+2),dtype=np.int8)
        flags[1:-1,1:-1] = np.array(mask)
        for y in range(1,ly+1):
            print "start", y
            for x in range(1,lx+1):
                if flags[y,x] == 1:
                    tnum = self.__deletecluster(flags, bias, 0, [(x,y)])
                    if tnum < bias:
                        flags[y,x] = 0
                    else:
                        flags[y,x] = 2
        flags[np.where(flags==2)] = 1
        return np.array(flags[1:-1,1:-1], dtype = np.uint8)

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
            src = np.array(data, dtype=np.float32)
            thresh, src_b = cv.threshold(src, 0.5, 1.0, cv.THRESH_BINARY_INV)
            thresh, src_f = cv.threshold(src, 0.5, 1.0, cv.THRESH_BINARY)
            thresh, src_w = cv.threshold(src, 0.5, 1.0, cv.THRESH_BINARY)
            th = 1.
            while th > 0:
                th = 0.
                th = 0.
                src_w = cv.filter2D(src_w, cv.CV_32F, self.pat_w[i])
                src_b = cv.filter2D(src_b, cv.CV_32F, self.pat_b[i])
                thresh, src_w = cv.threshold(src_w, 2.99, 1, cv.THRESH_BINARY)
                thresh, src_b = cv.threshold(src_b, 2.99, 1, cv.THRESH_BINARY)
                src_w = np.array(np.logical_and(src_w,src_b), dtype=np.float32)
                th += np.sum(src_w)
                src_f = np.array(np.logical_xor(src_f, src_w), dtype=np.float32)
                src_w = src_f.copy()
                thresh, src_b = cv.threshold(src_f, 0.5, 1.0, cv.THRESH_BINARY_INV)
            return src_w


class RivFinder:
    def __init__(self,img):
        self.img = img

    def filterimage(self):
        im = self.img.copy()

        print "closing"
        for i in range(0, 1):
            im = GraphicsFilters.closing(im, 1)

        print "opening2"
        for i in range(0, 6):
            im = GraphicsFilters.opening(im, 4)

        print "closing"
        for i in range(0, 3):
            im = GraphicsFilters.closing(im, 5)

        im = GraphicsFilters.deletecluster(im,1000)

        imgT = ImageThining()
        im = imgT.thining(im)

        return im




if __name__ == "__main__":
    reader = ImageReader("data/a.png")
    img = reader.getarray()
    img[np.where(img < 70 )] = 0
    img[np.where(img >= 70 )] = 1
    img = 1 - img
    ArrayWriter(img).save("data/c.png")
    rf = RivFinder(img)
    ans = rf.filterimage()
    ArrayWriter(ans).save("data/b.png")


