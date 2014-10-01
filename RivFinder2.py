from pyhdf.SD import *
import numpy as np
import cv2 as cv
import Image, ImageDraw
import sys
import gdal, ogr, os, osr
import thinning
import Raster2VectorLine

sys.setrecursionlimit(50000)

class Reader:
    def getarray(self):
        return self.image.copy()

    def get2tonearray(self, func):
        ans = np.zeros(self.image.shape,dtype=np.uint8)
        ans[np.where(func(self.image))] = 1
        return ans

    def get2tonearrayx(self):
        ans = cv.adaptiveThreshold(self.image,255,cv.ADAPTIVE_THRESH_GAUSSIAN_C,cv.THRESH_BINARY_INV,11,10)
        return ans

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
        im = cv.imread(filename);
        if len(im.shape) == 1:
            self.image = im
        else:
            self.image = im[:,:,0]

class Writer:
    def save(self, filename):
        self.image.save(filename)

class ArrayWriter(Writer):
    def __init__(self, arry):
        new = arry.copy()
        new[np.where(new > 0)] = 255
        self.image = Image.fromarray(new)

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
            for x in range(1,lx+1):
                if flags[y,x] == 1:
                    tnum = self.__deletecluster(flags, bias, 0, [(x,y)])
                    if tnum < bias:
                        flags[y,x] = 0
                    else:
                        flags[y,x] = 2
        flags[np.where(flags==2)] = 1
        return np.array(flags[1:-1,1:-1], dtype = np.uint8)

class ImageRasterizer:
    @classmethod
    def array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,array):
        cols = array.shape[1]
        rows = array.shape[0]
        originX = rasterOrigin[0]
        originY = rasterOrigin[1]

        driver = gdal.GetDriverByName('GTiff')
        outRaster = driver.Create(newRasterfn, cols, rows, 1, gdal.GDT_Byte)
        outRaster.SetGeoTransform((originX, pixelWidth, 0, originY, 0, pixelHeight))
        outband = outRaster.GetRasterBand(1)
        outband.WriteArray(array)
        outRasterSRS = osr.SpatialReference()
        outRasterSRS.ImportFromEPSG(4326)
        outRaster.SetProjection(outRasterSRS.ExportToWkt())
        outband.FlushCache()

    @classmethod
    def main(newRasterfn,array):
        rasterOrigin = (-123.25745,45.43013)
        pixelWidth = 2
        pixelHeight = 2
        reversed_arr = array[::-1] # reverse array so the tif looks like the array
        array2raster(newRasterfn,rasterOrigin,pixelWidth,pixelHeight,reversed_arr) # convert array to raster


class RivFinder:
    def __init__(self,img):
        self.img = img

    def filteredimage(self):
        im = self.img.copy()

        print "opening"
        for i in range(0, 12):
            im = GraphicsFilters.opening(im, 3)
            ArrayWriter(im.copy()).save("data/temp/t1_"+str(i)+".png")

        im = GraphicsFilters.deletecluster(im,100)
        ArrayWriter(im).save("data/temp/t2.png")

        print "opening"
        for i in range(0, 12):
            im = GraphicsFilters.opening(im, 3)
            ArrayWriter(im.copy()).save("data/temp/t3_"+str(i)+".png")

        print "closing"
        for i in range(0, 1):
            im = GraphicsFilters.closing(im, 5)
            ArrayWriter(im).save("data/temp/t4_"+str(i)+".png")

        im = GraphicsFilters.deletecluster(im,2000)
        ArrayWriter(im).save("data/temp/t5.png")

        print "thinning"
        im = thinning.thinning(im)

        return im



if __name__ == "__main__":
    reader = ImageReader("data/1.png")
    img = reader.get2tonearray(lambda x: x < 70)
    #img = reader.get2tonearrayx()
    ArrayWriter(img).save("data/a.png")
    rf = RivFinder(img)
    ans = rf.filteredimage()
    ArrayWriter(ans).save("data/b.png")
    Raster2VectorLine.main("data/b.png","data/c.shp",0)

    #imgT = ImageThining()
    #img = imgT.thining(ans)
    #ArrayWriter(img).save("data/d.png")

