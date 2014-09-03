from pyhdf.SD import *
import numpy as np
import cv2
import Image, ImageDraw

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
        self.image = Image.fromarray(arry)


class GraphicsFilters:
    @classmethod
    def opening(self, data, area, bias, newdata):
        (dx,dy) = area
        for x in range(dx,len(data[0])-dx):
            for y in range(dy,len(data)-dy):
                a = data[y-dy:y+dy+1,x-dx:x+dx+1]
                if len(np.where(a==1)[0]) >= bias:
                    newdata[y,x] = 1

    @classmethod
    def closing(self, data, area, bias, newdata):
        (dx,dy) = area
        for x in range(dx,len(data[0])-dx):
            for y in range(dy,len(data)-dx):
                a = data[y-dy:y+dy+1,x-dx:x+dx+1]
                if len(np.where(a==1)[0]) <= bias:
                    newdata[y,x] = 0

    @classmethod
    def deletecluster(self, data, mask, bias, num, celllist):
       if len(celllist)==0:
           return num
       (x,y) = celllist.pop()
       if 0<=x<len(data[0]) and 0<=y<len(data) and data[y,x]==1 and mask[y,x]==0:
           celllist.extend([(x-1,y),(x+1,y),(x,y-1),(x,y+1)])
           mask[y,x] = 1
           tnum = self.deletecluster(data,mask,bias,num+1,celllist)
           if tnum < bias :
               data[y,x] = 0
           return tnum
       else:
           return self.deletecluster(data,mask,bias,num,celllist)

    @classmethod
    def deletesmallcluster(self, data, bias):
        mask = np.zeros(data.shape)
        for x in range(0,len(data[0])):
            for y in range(0,len(data)):
                    self.deletecluster(data,mask,bias,0,[(x,y)])


class RivFinder:
    def __init__(self,img):
        self.img = img
        self.aimg = img

    def filterimage(self):
        im = self.img.copy()
        nim = self.img.copy()
        
        for i in range(0, 12):
            GraphicsFilters.opening(im, (1,1), 3, nim)
            im = nim

        for i in range(0, 12):
            GraphicsFilters.opening(im, (1,1), 4, nim)
            im = nim

        for i in range(0, 12):
            GraphicsFilters.closing(im, (1,1), 6, nim)
            im = nim

        GraphicsFilters.deletesmallcluster(im,1000)
        return im


if __name__ == "__main__":
    reader = ImageReader("a.png")
    img = reader.getarray()
    rf = RivFinder(img)
    ans = rf.filterimage()
    writer = ArrayWriter(ans)
    writer.save("b.png")


