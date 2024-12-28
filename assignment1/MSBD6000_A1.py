#############################################################################
## The code owned by
## Name:Lam Chun Ting Jeff
## SID: 12222973
## ITSC: ctjlam
## Last Modified: 18/10/2021 
## The code is for MSBD6000J Spatial and multimedia database in HKUST
############################################################################

from tqdm import tqdm
import pandas as pd
import random
import pickle
import os
from enum import Enum

FILE_NAME = "AllPOI Simplified.csv"
MAX_BUCKET_SIZE = 256


# Defined with a point (x (float),y (float))
class Point:
    def __init__(self, x=0,y=0):
        self.x = x
        self.y = y

    def __repr__(self):
        return "Point (X: "+str(self.x)+", Y: "+str(self.y)+")"

    def __str__(self):
        return "Point (X: "+str(self.x)+", Y: "+str(self.y)+")"

# Defined with a rectangle (lowerLeftPt (point), upperRightPt (point))
class Rectangle:
    def __init__(self, lowerLeftPt, upperRightPt):
        assert(isinstance(lowerLeftPt, Point))
        assert(isinstance(upperRightPt, Point))
        assert((lowerLeftPt.x<upperRightPt.x)and(lowerLeftPt.y<upperRightPt.y))
        self.lowerLeftPt = lowerLeftPt
        self.upperRightPt = upperRightPt
    
    def isPtBounded(self,pt):
        assert(isinstance(pt, Point))
        if (pt.x >= self.lowerLeftPt.x and pt.x<= self.upperRightPt.x) and (pt.y >= self.lowerLeftPt.y and pt.y <= self.upperRightPt.y):
            return True
        else:
            return False

    def __str__(self):
        return "Rectangle (Low: "+str(self.lowerLeftPt)+" High: "+str(self.upperRightPt)+")"

# Define a minimum bounding rectangle, given a dataframe with column ['X'] and ['Y']
class MBR(Rectangle):
    def __init__(self, df):
        x = df['X']
        y = df['Y']
        Rectangle.__init__(self,Point(x.min(), y.min()),Point(x.max(), y.max()))

    def isWindowInContraint(self, window):
        assert(isinstance(window, Rectangle))
        return self.isPtBounded(window.lowerLeftPt) and self.isPtBounded(window.upperRightPt)

    def __str__(self):
        return "MBR\n(LowPoint: "+str(self.lowerLeftPt)+")\n(MaxPoint: "+str(self.upperRightPt)+")"

# Define set of rectangle relationship
class RectangleRelationship(Enum):
    WARPPED = 1         # Rectanlge B in A
    CONTAIN = 2         # Some in A and B parts intercept
    NO_RELATION = 3     # Do not overlap 


# Define a grid, with (r,b) pairs in the specification
# For (r, b) pairs, where r is a rectangle and b is a pointer to a bucket that holds all points in D inside r.
class Grid:
    def __init__(self, boundingRectangle):
        assert(isinstance(boundingRectangle, Rectangle))
        self.boundingRectangle = boundingRectangle
        self.bucket = []

    ## Given a window (Rectangle), it will find the relationship between bounding rectangle
    def getRelationship(self, window):
        assert(isinstance(window, Rectangle))

        if (window.lowerLeftPt.x < self.boundingRectangle.upperRightPt.x and window.upperRightPt.x > self.boundingRectangle.lowerLeftPt.x and window.lowerLeftPt.y > self.boundingRectangle.upperRightPt.y and window.upperRightPt.y < self.boundingRectangle.lowerLeftPt.y):
            return RectangleRelationship.NO_RELATION
        elif (window.lowerLeftPt.x <= self.boundingRectangle.lowerLeftPt.x and window.upperRightPt.x >= self.boundingRectangle.upperRightPt.x and window.upperRightPt.y >= self.boundingRectangle.upperRightPt.y and window.lowerLeftPt.y <= self.boundingRectangle.lowerLeftPt.y):
            return RectangleRelationship.WARPPED
        else:
            return RectangleRelationship.CONTAIN

    ## Given a window (Rectangle), generate an interested window within the bounding rectangle
    ## Then count the number in the bucket within the computed window
    def getNumberOfPointsInsideWindow(self, window):
        assert(isinstance(window, Rectangle))
        relationship = self.getRelationship(window)
        assert(relationship == RectangleRelationship.CONTAIN)

        x1 = max(window.lowerLeftPt.x, self.boundingRectangle.lowerLeftPt.x)
        x2 = min(window.upperRightPt.x, self.boundingRectangle.upperRightPt.x)
        y1 = max(window.lowerLeftPt.y, self.boundingRectangle.lowerLeftPt.y)
        y2 = min(window.upperRightPt.y, self.boundingRectangle.upperRightPt.y)
        lowerLeftPt = Point(min(x1,x2),min(y1,y2))
        upperRightPt = Point(max(x1,x2),max(y1,y2))

        ## Computed window inside the grid
        windowInGrid = Rectangle(lowerLeftPt,upperRightPt)

        count = 0
        for pt in self.bucket:
            if (windowInGrid.isPtBounded(pt) == True):
                count = count+1

        return count

    def __repr__(self):
        return "Grid\n(LowPoint: "+str(self.boundingRectangle.lowerLeftPt)+")\n(MaxPoint: "+str(self.boundingRectangle.upperRightPt)+")\n"

    def __str__(self):
        return "Grid\n(LowPoint: "+str(self.boundingRectangle.lowerLeftPt)+")\n(MaxPoint: "+str(self.boundingRectangle.upperRightPt)+")\n"


## 2D Indexing Database, Parent class defining necessary function and instance
class TwoDimensionalIndexDB:
    def __init__(self, mbr):
        self.mbr = mbr
        self.queryProfile = {"totalNumberOfPoints": 0,"numberOfIndexCells":0, "numberOfPointsSearch":0 }
    
    def _resetQuery(self):
        self.queryProfile = {"totalNumberOfPoints": 0,"numberOfIndexCells":0, "numberOfPointsSearch":0 }

    ## Insert a pt into a bucket, if excceed max bucket size, decompose based on indexing method
    def insert(self):
        raise NotImplementedError("Please use either Excell or Quad Tree class")

    ## Perform a window query
    def windowQuery(self):
        raise NotImplementedError("Please use either Excell or Quad Tree class")

    ## Print the profile for indexing method
    def printProfile(self):
        raise NotImplementedError("Please use either Excell or Quad Tree class")

    ## Print the information for latest window query
    def printQueryInfo(self):
        print("Query Profile: "+str(self.queryProfile))


#Q2
## Excell method, inheritance from 2D Indexing Database
class EXCELL(TwoDimensionalIndexDB):
    def __init__(self, mbr, maxBucketSize):
        assert(isinstance(mbr, MBR))
        super().__init__(mbr)
        self.maxBucketSize = maxBucketSize
        self.__numberOfDividor = 1
        self.grids = [Grid(mbr)]

    ## Specific for EXCELL, given a point, found the grid index
    def __getPtIndex(self, pt):
        assert(isinstance(pt,Point))
        index_x = int((pt.x - self.mbr.lowerLeftPt.x)/(self.mbr.upperRightPt.x - self.mbr.lowerLeftPt.x)*self.__numberOfDividor)
        index_y = int((pt.y - self.mbr.lowerLeftPt.y)/(self.mbr.upperRightPt.y - self.mbr.lowerLeftPt.y)*self.__numberOfDividor)

        if (index_x == self.__numberOfDividor):
            index_x = index_x - 1
        if (index_y == self.__numberOfDividor):
            index_y = index_y - 1

        return index_x*self.__numberOfDividor + index_y

    ## Insert method
    def insert(self, pt):
        assert(isinstance(pt, Point))
        assert(self.mbr.isPtBounded(pt) == True)
        isFull = False

        ## Find the grids index
        index = self.__getPtIndex(pt)

        ## Check if the pt is belongs to that grid
        if (self.grids[index].boundingRectangle.isPtBounded(pt) == True):

            ## Insert into it
            self.grids[index].bucket.append(pt)

            ## Check if bucket exceed maxBucket size
            if(len(self.grids[index].bucket)>self.maxBucketSize):
                isFull = True
        else:
            raise RuntimeError("Cannot insert into any grid")
        
        ## Do deomposition if necessary
        if (isFull == True):

            ## Decompose
            self.__numberOfDividor = self.__numberOfDividor + 1

            ## Re divide the MBR into the nxn, where n = number of dividor
            new_grids = []
            for i in range(self.__numberOfDividor):
                for j in range(self.__numberOfDividor):
                    lower_x = self.mbr.lowerLeftPt.x + (self.mbr.upperRightPt.x - self.mbr.lowerLeftPt.x)*(i)/self.__numberOfDividor
                    lower_y = self.mbr.lowerLeftPt.y + (self.mbr.upperRightPt.y - self.mbr.lowerLeftPt.y)*(j)/self.__numberOfDividor
                    upper_x = self.mbr.lowerLeftPt.x + (self.mbr.upperRightPt.x - self.mbr.lowerLeftPt.x)*(i+1)/self.__numberOfDividor
                    upper_y = self.mbr.lowerLeftPt.y + (self.mbr.upperRightPt.y - self.mbr.lowerLeftPt.y)*(j+1)/self.__numberOfDividor
                    new_grids.append(Grid(Rectangle(Point(lower_x,lower_y),Point(upper_x, upper_y))))

            ## Put old bucket data into new bucket
            old_grids = self.grids
            self.grids = new_grids

            for grid in old_grids:
                for pt in grid.bucket:
                    self.insert(pt)

    ## Bucket iterator, only for EXCELL method
    def __gridBucketIterator(self):
        for grid in self.grids:
            yield len(grid.bucket)

    ## Getting the bucket distribution
    def getBucketDistribution(self):
        result = {"0": 0, "1-25":0, "26-239":0, "240-255":0, "256":0}
        for size in self.__gridBucketIterator():
            if (size == 0):
                result["0"] = result["0"]+1
            elif (size >= 1 and size<=25):
                result["1-25"] = result["1-25"]+1
            elif (size >= 26 and size<=239):
                result["26-239"] = result["26-239"]+1
            elif (size >= 240 and size<=255):
                result["240-255"] = result["240-255"]+1
            elif (size == 256):
                result["256"] = result["256"]+1
            else:
                raise ValueError("Something wrong")
        return result

    ## Perform window query
    def windowQuery(self, window):
        assert(isinstance(window,Rectangle))
        assert(self.mbr.isWindowInContraint(window))

        ## Reset window query information
        self._resetQuery()

        ## Get the range of the index that have to perform search
        lowest_index = self.__getPtIndex(window.lowerLeftPt)
        highest_index = self.__getPtIndex(window.upperRightPt)

        for i in range(lowest_index, highest_index+1):
            
            ## Travse grid index
            self.queryProfile["numberOfIndexCells"] = self.queryProfile["numberOfIndexCells"]+1

            ## Get the relationship between the grid and the window query
            relationship = self.grids[i].getRelationship(window)

            ## No relation, just skip this grid
            if (relationship == RectangleRelationship.NO_RELATION):
                continue
            
            ## The window bounded with the grid, all bucket will counted
            elif (relationship == RectangleRelationship.WARPPED):
                self.queryProfile["totalNumberOfPoints"] = self.queryProfile["totalNumberOfPoints"] + len(self.grids[i].bucket)

            ## The window paritialy inside the grid, check how many points inside the grid
            else:
                self.queryProfile["totalNumberOfPoints"] = self.queryProfile["totalNumberOfPoints"] + self.grids[i].getNumberOfPointsInsideWindow(window)
                self.queryProfile["numberOfPointsSearch"] = self.queryProfile["numberOfPointsSearch"] + len(self.grids[i].bucket)

        return self.queryProfile["totalNumberOfPoints"]

    def getNumberOfPoint(self):
        result = 0
        for size in self.__gridBucketIterator():
            result = result + size
        return result
    
    def printProfile(self):
        print("Minimum Bounding Rectangle: "+str(self.mbr))
        print("Number of dividor: "+str(self.__numberOfDividor))

    def printQueryInfo(self):
        print("EXCELL Query Profile: "+str(self.queryProfile))
        

#Q3
## Quad Tree node defination, which use z value as name
class Node:
    def __init__(self, name, boundingRectangle):
        assert(isinstance(boundingRectangle, Rectangle))
        self.name = name

        self.grid = Grid(boundingRectangle)

        self.isLeaf = True
        self.children = None

    def __str__(self):
        return "Node (isLeaf: "+str(self.isLeaf)+") Bucket Info: "+str(self.bucket)

## Quad Tree method, inheritance from 2D Indexing Database
class QuadTree(TwoDimensionalIndexDB):
    def __init__(self, mbr, maxBucketSize):
        assert(isinstance(mbr, MBR))
        super().__init__(mbr)
        self.root = Node("root", mbr)
        self.maxBucketSize = maxBucketSize
        self.__numberOfDecompaction = 0
    
    ## Get the pt index for the grid
    def __getPtIndex(self, pt, grid):
        assert(isinstance(pt, Point))
        assert(isinstance(grid, Grid))
        index_x = int((pt.x - grid.boundingRectangle.lowerLeftPt.x)/(grid.boundingRectangle.upperRightPt.x - grid.boundingRectangle.lowerLeftPt.x)*2)
        index_y = int((pt.y - grid.boundingRectangle.lowerLeftPt.y)/(grid.boundingRectangle.upperRightPt.y - grid.boundingRectangle.lowerLeftPt.y)*2)

        if (index_x == 2):
            index_x = index_x - 1
        if (index_y == 2):
            index_y = index_y - 1

        return index_x*2 + index_y

    ## Perform insert, in recursive way from root
    def insert(self, pt):
        assert(isinstance(pt, Point))
        assert(self.root.grid.boundingRectangle.isPtBounded(pt) == True)
        self.__insertRecursive(self.root, pt)

    ## Insert recursion
    def __insertRecursive(self, node, pt):
        assert(isinstance(pt, Point))
        assert(isinstance(node, Node))
        assert(node.grid.boundingRectangle.isPtBounded(pt) == True)

        ## Check if it is leaf
        if (node.isLeaf):

            ## Directly append
            node.grid.bucket.append(pt)

            ## Check if bucket required decompose
            if (len(node.grid.bucket)>self.maxBucketSize):
                
                ## No longer a leaf, and do decomposition
                node.isLeaf = False

                ## Perform decomposition
                node.children = []
                for i in range(2):
                    for j in range(2):
                        lower_x = node.grid.boundingRectangle.lowerLeftPt.x + (node.grid.boundingRectangle.upperRightPt.x - node.grid.boundingRectangle.lowerLeftPt.x)*(i)/2
                        lower_y = node.grid.boundingRectangle.lowerLeftPt.y + (node.grid.boundingRectangle.upperRightPt.y - node.grid.boundingRectangle.lowerLeftPt.y)*(j)/2
                        upper_x = node.grid.boundingRectangle.lowerLeftPt.x + (node.grid.boundingRectangle.upperRightPt.x - node.grid.boundingRectangle.lowerLeftPt.x)*(i+1)/2
                        upper_y = node.grid.boundingRectangle.lowerLeftPt.y + (node.grid.boundingRectangle.upperRightPt.y - node.grid.boundingRectangle.lowerLeftPt.y)*(j+1)/2
                        node.children.append(Node(node.name+str(i*2+j), Rectangle(Point(lower_x,lower_y),Point(upper_x, upper_y))))

                ## Check the maximum depth of quad tree
                m = len(node.name)-4
                if (m>self.__numberOfDecompaction):
                    self.__numberOfDecompaction = m
                
                ## For each point in original bucket, insert again into the new decomposed node
                for decompose_pt in node.grid.bucket:
                    self.__insertRecursive(node, decompose_pt)
                
                ## Remove the bucket as the data all transfered
                del node.grid.bucket
        
        ## If it is not a leaf
        else:

            ## Perform travsal
            ## Find the index of the children grid
            index = self.__getPtIndex(pt,node.grid)

            ## Do recursive if that pt is inside that node
            if (node.children[index].grid.boundingRectangle.isPtBounded(pt)):
                self.__insertRecursive(node.children[index],pt)
            else:
                raise RuntimeError("Something wrong in index calculation")

    ## Recursive method to Generate the bucket length in each leaf node
    def __depthFirstSearchForBucketSize(self, node):
        if (node.isLeaf == True):
            yield len(node.grid.bucket)
        else:
            for i in node.children:
                yield from self.__depthFirstSearchForBucketSize(i)

    ## Recursive method to find the node fullfill window
    def __depthFirstSearchForQuery(self, node, window):

        ## If it is already leaf node
        if (node.isLeaf == True):

            self.queryProfile["numberOfIndexCells"] = self.queryProfile["numberOfIndexCells"]+1

            ## Find the relationship of this node to window
            relationship = node.grid.getRelationship(window)

            ## If the window warp with the grid
            if (relationship == RectangleRelationship.WARPPED):

                ## All element with this node is be considered
                self.queryProfile["totalNumberOfPoints"] = self.queryProfile["totalNumberOfPoints"] + len(node.grid.bucket)

            ## If the grid contain some parts of the window
            elif (relationship == RectangleRelationship.CONTAIN):

                ## Find the points inside within the window
                self.queryProfile["totalNumberOfPoints"] = self.queryProfile["totalNumberOfPoints"] + node.grid.getNumberOfPointsInsideWindow(window)

                ## All points in this bucket will be searched
                self.queryProfile["numberOfPointsSearch"] = self.queryProfile["numberOfPointsSearch"] + len(node.grid.bucket)
        
        ## It is not leaf node
        else:

            ## For each children
            for i in node.children:

                self.queryProfile["numberOfIndexCells"] = self.queryProfile["numberOfIndexCells"]+1

                ## Do travesal if there is relation between the window and the bounding rectangle
                if (i.grid.getRelationship(window) != RectangleRelationship.NO_RELATION):
                    self.__depthFirstSearchForQuery(i,window)
    
    ## Getting the bucket distribution
    def getBucketDistribution(self):
        result = {"0": 0, "1-25":0, "26-239":0, "240-255":0, "256":0}
        for size in self.__depthFirstSearchForBucketSize(self.root):
            if (size == 0):
                result["0"] = result["0"]+1
            elif (size >= 1 and size<=25):
                result["1-25"] = result["1-25"]+1
            elif (size >= 26 and size<=239):
                result["26-239"] = result["26-239"]+1
            elif (size >= 240 and size<=255):
                result["240-255"] = result["240-255"]+1
            elif (size == 256):
                result["256"] = result["256"]+1
            else:
                raise ValueError("Something wrong")
        return result
    
    def getTreeHeight(self, node=None):
        if node is None:
            node = self.root
        
        if (node.isLeaf == True):
            return 1
        else:
            depth = []
            for child in node.children:
                depth.append(self.getTreeHeight(child))
            return max(depth)+1

    ## Perform window query
    def windowQuery(self, window):
        assert(isinstance(window,Rectangle))
        assert(self.mbr.isWindowInContraint(window))

        ## Reset the information of the query progress
        self._resetQuery()

        ## Do recursive search
        self.__depthFirstSearchForQuery(self.root,window)

        return self.queryProfile["totalNumberOfPoints"]

    ## Get total number of points in this db            
    def getNumberOfPoint(self):
        result = 0
        for size in self.__depthFirstSearchForBucketSize(self.root):
            result = result + size
        return result
    
    def printProfile(self):
        print("Minimum Bounding Rectangle: "+str(self.mbr))
        print("Number of decompaction: "+str(self.getTreeHeight()))

    def printQueryInfo(self):
        print("Quad Tree Query Profile: "+str(self.queryProfile))

############################################################################################################

## Randomly generate a window within the MBR
def randomWindowGenerate(mbr, n=10):
    assert(isinstance(n,int))
    assert(isinstance(mbr,MBR))
    for i in range(n):
        x1 = random.uniform(mbr.lowerLeftPt.x, mbr.upperRightPt.x)
        x2 = random.uniform(mbr.lowerLeftPt.x, mbr.upperRightPt.x)
        y1 = random.uniform(mbr.lowerLeftPt.y, mbr.upperRightPt.y)
        y2 = random.uniform(mbr.lowerLeftPt.y, mbr.upperRightPt.y)

        yield Rectangle(Point(min(x1,x2), min(y1,y2)),Point(max(x1,x2), max(y1,y2)))


## Get EXCELL method, find if there is file exist, then just return; or else, it will create a excell db with the dataset
def getEXCELL(filename, storage_path="./excell.obj"):
    dataframe = pd.read_csv(filename, header=None, names=['X','Y'])
    minimumBoundingRectangle = MBR(dataframe)

    excell = None

    need_create = True
    
    try:
        if (os.path.exists(storage_path)):
            file_handler = open(storage_path,'rb')
            excell = pickle.load(file_handler)

            print("=========== EXCELL Found Info ==============")
            excell.printProfile()
            print("Bucket Distribution: "+str(excell.getBucketDistribution()))
            print("Number of Points: "+str(excell.getNumberOfPoint()))
            print("============================================")
            
            ans = input("Would you like to recreate excell? (y/n) ")
            if (ans.lower() != 'y'):
                need_create = False
    except Exception as e:
        print(e)
        print("Trying to resconstruct EXCELL")
        need_create = True


    if (need_create == True):
        excell = EXCELL(minimumBoundingRectangle, MAX_BUCKET_SIZE)

        for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Inserting into EXCELL"):
            excell.insert(Point(row['X'], row['Y']))
        
        print("Saving excell into "+str(storage_path)+"... ")
        file_handler = open(storage_path,'wb')
        pickle.dump(excell, file_handler)

        print("=========== EXCELL Created Info ==============")
        excell.printProfile()
        print("Bucket Distribution: "+str(excell.getBucketDistribution()))
        print("Number of Points: "+str(excell.getNumberOfPoint()))
        print("=============================================")

    return excell


## Get QUAD Tree method, find if there is file exist, then just return; or else, it will create a quad tree db with the dataset
def getQuadTree(filename, storage_path="./quadTree.obj"):
    dataframe = pd.read_csv(filename, header=None, names=['X','Y'])
    minimumBoundingRectangle = MBR(dataframe)

    quadTree = None

    need_create = True
    try:
        if (os.path.exists(storage_path)):
            file_handler = open(storage_path,'rb')
            quadTree = pickle.load(file_handler)

            print("========== Quad Tree Found Info ===========")
            quadTree.printProfile()
            print("Bucket Distribution: "+str(quadTree.getBucketDistribution()))
            print("Number of Points: "+str(quadTree.getNumberOfPoint()))
            print("============================================")
            
            ans = input("Would you like to recreate Quad Tree? (y/n) ")
            if (ans.lower() != 'y'):
                need_create = False
    except Exception as e:
        print(e)
        print("Trying to resconstruct quad tree")
        need_create = True

    if (need_create == True):
        quadTree = QuadTree(minimumBoundingRectangle,MAX_BUCKET_SIZE)

        for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0], desc="Inserting into quad tree"):
            quadTree.insert(Point(row['X'], row['Y']))
        
        print("Saving quad tree into "+str(storage_path)+"... ")
        file_handler = open(storage_path,'wb')
        pickle.dump(quadTree, file_handler)

        print("========== Quad Tree Created Info ===========")
        quadTree.printProfile()
        print("Bucket Distribution: "+str(quadTree.getBucketDistribution()))
        print("Number of Points: "+str(quadTree.getNumberOfPoint()))
        print("============================================")

    return quadTree


if __name__ == "__main__":
    
    try:
        ## Excell implementation
        db_1 = getEXCELL(FILE_NAME)

        ## Quad Tree implementation
        db_2 = getQuadTree(FILE_NAME)

        print("Both EXCELL and Quad Tree ready")
        input("press enter for windows query part...")

        windows = randomWindowGenerate(db_2.mbr)
        for window in windows:
            print("===================================")
            print("Query Window: "+str(window))
            db_1.windowQuery(window)
            db_1.printQueryInfo()
            db_2.windowQuery(window)
            db_2.printQueryInfo()
            print("===================================")
    except KeyboardInterrupt:
        print("Program exit")

