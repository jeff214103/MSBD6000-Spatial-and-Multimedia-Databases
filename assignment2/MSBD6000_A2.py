import pandas as pd
import math
from tqdm import tqdm
import pickle
import os
import sys
import random


FILE_NAME = "AllPOI Simplified.csv"
SUB_TREE = [2,5]
BUCKET_SIZE = [128,156,256]


def ptDistance(pt1, pt2):
    assert(isinstance(pt1, Point))
    assert(isinstance(pt2, Point))
    return math.sqrt((pt1.x - pt2.x)* (pt1.x - pt2.x) + (pt1.y - pt2.y)*  (pt1.y - pt2.y))

###################################################################################
#Retrieved from https://github.com/LBDM2707/Python_R-Tree/blob/master/Region_tree.py
class Rect:
    def __init__(self, x1, y1, x2, y2):
        assert(x1<=x2)
        assert(y1<=y2)
        self.x1 = x1
        self.y1 = y1
        self.x2 = x2
        self.y2 = y2

    def perimeter(self):
        return 2 * (abs(self.x2 - self.x1) + abs(self.y2 - self.y1))

    def is_overlap(self, rect):
        assert(isinstance(rect,Rect))
        if self.y1 > rect.y2 or self.y2 < rect.y1 or self.x1 > rect.x2 or self.x2 < rect.y1:
            return False
        return True

    def contain_rect(self, rect):
        assert(isinstance(rect,Rect))
        return self.x1 < rect.x1 and self.y1 < rect.y1 and self.x2 > rect.x2 and self.y2 > rect.y2

    def has_point(self, point):
        assert(isinstance(point,Point))
        return self.x1 <= point.x <= self.x2 and self.y1 <= point.y <= self.y2

    def __str__(self):
        return "Rect: ({}, {}), ({}, {})".format(self.x1, self.y1, self.x2, self.y2)


class Point:
    def __init__(self, id, x, y):
        self.id = id
        self.x = x
        self.y = y

    def __str__(self):
        return "Point #{}: ({}, {})".format(self.id, self.x, self.y)


class Node(object):
    def __init__(self, max_bucket_size, max_sub_tree):
        self.max_bucket_size = max_bucket_size
        self.max_sub_tree = max_sub_tree
        self.id = 0
        # for internal nodes
        self.child_nodes = []
        # for leaf nodespyth
        self.data_points = []
        self.parent_node = None
        self.MBR = Rect(-1, -1, -1, -1)

    def add_point(self, point):
        assert(isinstance(point,Point))
        # update in the right position to keep the list ordered
        self.add_points([point])
        pass

    def add_points(self, points):
        self.data_points += points
        # update MBR
        self.update_MBR()
        pass

    def perimeter_increase_with_point(self, point):
        assert(isinstance(point,Point))
        x1 = point.x if point.x < self.MBR.x1 else self.MBR.x1
        y1 = point.y if point.y < self.MBR.y1 else self.MBR.y1
        x2 = point.x if point.x > self.MBR.x2 else self.MBR.x2
        y2 = point.y if point.y > self.MBR.y2 else self.MBR.y2
        return Rect(x1, y1, x2, y2).perimeter() - self.perimeter()

    def perimeter(self):
        # only calculate the half perimeter here
        return self.MBR.perimeter()

    def is_underflow(self):
        return (self.is_leaf() and len(self.data_points) <= self.max_bucket_size) or \
               (not self.is_leaf() and len(self.child_nodes) <= self.max_sub_tree)

    def is_overflow(self):
        return (self.is_leaf() and len(self.data_points) > self.max_bucket_size) or \
               (not self.is_leaf() and len(self.child_nodes) > self.max_sub_tree)

    def is_root(self):
        return self.parent_node is None

    def is_leaf(self):
        return len(self.child_nodes) == 0

    def add_child_node(self, node):
        assert(isinstance(node,Node))
        self.add_child_nodes([node])
        pass

    def add_child_nodes(self, nodes):
        for node in nodes:
            node.parent_node = self
            self.child_nodes.append(node)
        self.update_MBR()
        pass

    def update_MBR(self):
        if self.is_leaf():
            self.MBR.x1 = min([point.x for point in self.data_points])
            self.MBR.x2 = max([point.x for point in self.data_points])
            self.MBR.y1 = min([point.y for point in self.data_points])
            self.MBR.y2 = max([point.y for point in self.data_points])
        else:
            self.MBR.x1 = min([child.MBR.x1 for child in self.child_nodes])
            self.MBR.x2 = max([child.MBR.x2 for child in self.child_nodes])
            self.MBR.y1 = min([child.MBR.y1 for child in self.child_nodes])
            self.MBR.y2 = max([child.MBR.y2 for child in self.child_nodes])
        if self.parent_node and not self.parent_node.MBR.contain_rect(self.MBR):
            self.parent_node.update_MBR()
        pass

    # Get perimeter of an MBR formed by a list of data points
    @staticmethod
    def get_points_MBR_perimeter(points):
        x1 = min([point.x for point in points])
        x2 = max([point.x for point in points])
        y1 = min([point.y for point in points])
        y2 = max([point.y for point in points])
        return Rect(x1, y1, x2, y2).perimeter()

    @staticmethod
    def get_nodes_MBR_perimeter(nodes):
        x1 = min([node.MBR.x1 for node in nodes])
        x2 = max([node.MBR.x2 for node in nodes])
        y1 = min([node.MBR.y1 for node in nodes])
        y2 = max([node.MBR.y2 for node in nodes])
        return Rect(x1, y1, x2, y2).perimeter()


class RTree:
    def __init__(self, max_bucket_size, max_sub_tree):
        self.max_bucket_size = max_bucket_size
        self.max_sub_tree = max_sub_tree
        self.root = Node(self.max_bucket_size, max_sub_tree)

    def insert_point(self, point, cur_node=None):
        assert(isinstance(point,Point))
        # init U as node
        # print("{} is leaf: {}".format(self.root, self.root.is_leaf()))
        if cur_node is None:
            cur_node = self.root

            # print("{} is leaf: {}".format(cur_node, cur_node.is_leaf()))
        # Insertion logic start
        if cur_node.is_leaf():
            cur_node.add_point(point)
            # handle overflow
            if cur_node.is_overflow():
                self.__handle_overflow(cur_node)
        else:
            chosen_child = self.__choose_best_child(cur_node, point)
            self.insert_point(point, cur_node=chosen_child)

    # Find a suitable one to expand:
    @staticmethod
    def __choose_best_child(node, point):
        assert(isinstance(point,Point))
        best_child = None
        best_perimeter = 0
        # Scan the child nodes
        for item in node.child_nodes:
            if node.child_nodes.index(item) == 0 or best_perimeter > item.perimeter_increase_with_point(point):
                best_child = item
                best_perimeter = item.perimeter_increase_with_point(point)
        return best_child

    # WIP
    def __handle_overflow(self, node):
        assert(isinstance(node,Node))
        node, new_node = self.__split_leaf_node(node) if node.is_leaf() else self.__split_internal_node(node)

        if self.root is node:
            self.root = Node(self.max_bucket_size, self.max_sub_tree)
            self.root.add_child_nodes([node, new_node])
        else:
            node.parent_node.add_child_node(new_node)
            if node.parent_node.is_overflow():
                self.__handle_overflow(node.parent_node)

    # WIP
    def __split_leaf_node(self, node):
        assert(isinstance(node,Node))
        m = len(node.data_points)
        best_perimeter = -1
        best_set_1 = []
        best_set_2 = []
        # Run x axis
        all_point_sorted_by_x = sorted(node.data_points, key=lambda point: point.x)
        for i in range(int(0.4 * m), int(m * 0.6) + 1):
            list_point_1 = all_point_sorted_by_x[:i]
            list_point_2 = all_point_sorted_by_x[i:]
            temp_sum_perimeter = Node.get_points_MBR_perimeter(list_point_1) \
                                 + Node.get_points_MBR_perimeter(list_point_2)
            if best_perimeter == -1 or best_perimeter > temp_sum_perimeter:
                best_perimeter = temp_sum_perimeter
                best_set_1 = list_point_1
                best_set_2 = list_point_2
        # Run y axis
        all_point_sorted_by_y = sorted(node.data_points, key=lambda point: point.y)
        for i in range(int(0.4 * m), int(m * 0.6) + 1):
            list_point_1 = all_point_sorted_by_y[:i]
            list_point_2 = all_point_sorted_by_y[i:]
            temp_sum_perimeter = Node.get_points_MBR_perimeter(list_point_1) \
                                 + Node.get_points_MBR_perimeter(list_point_2)
            if best_perimeter == -1 or best_perimeter > temp_sum_perimeter:
                best_perimeter = temp_sum_perimeter
                best_set_1 = list_point_1
                best_set_2 = list_point_2
        node.data_points = best_set_1
        node.update_MBR()
        new_node = Node(self.max_bucket_size, self.max_sub_tree)
        new_node.add_points(best_set_2)
        return node, new_node

    # WIP
    def __split_internal_node(self, node):
        assert(isinstance(node,Node))
        m = len(node.child_nodes)
        best_perimeter = -1
        best_set_1 = []
        best_set_2 = []
        # Run x axis
        all_node_sorted_by_x = sorted(node.child_nodes, key=lambda child: child.MBR.x1)
        for i in range(int(0.4 * m), int(m * 0.6) + 1):
            list_node_1 = all_node_sorted_by_x[:i]
            list_node_2 = all_node_sorted_by_x[i:]
            temp_sum_perimeter = Node.get_nodes_MBR_perimeter(list_node_1) \
                                 + Node.get_nodes_MBR_perimeter(list_node_2)
            if best_perimeter == -1 or best_perimeter > temp_sum_perimeter:
                best_perimeter = temp_sum_perimeter
                best_set_1 = list_node_1
                best_set_2 = list_node_2
                # Run y axis
        all_node_sorted_by_y = sorted(node.child_nodes, key=lambda child: child.MBR.y1)
        for i in range(int(0.4 * m), int(m * 0.6) + 1):
            list_node_1 = all_node_sorted_by_y[:i]
            list_node_2 = all_node_sorted_by_y[i:]
            temp_sum_perimeter = Node.get_nodes_MBR_perimeter(list_node_1) \
                                 + Node.get_nodes_MBR_perimeter(list_node_2)
            if best_perimeter == -1 or best_perimeter > temp_sum_perimeter:
                best_perimeter = temp_sum_perimeter
                best_set_1 = list_node_1
                best_set_2 = list_node_2
        node.child_nodes = best_set_1
        node.update_MBR()
        new_node = Node(self.max_bucket_size, self.max_sub_tree)
        new_node.add_child_nodes(best_set_2)
        return node, new_node

    # Take in a Rect and return number of data point that is covered by the R tree.
    def region_query(self, rect, node=None):
        assert(isinstance(rect,Rect))
        # initiate with root
        if node is None:
            node = self.root

        if node.is_leaf():
            count = 0
            for point in node.data_points:
                if rect.has_point(point):
                    count += 1
            return count
        else:
            total = 0
            for child in node.child_nodes:
                # print("{} and {} is overlapped {}".format(rect, child.MBR, rect.is_overlap(child.MBR)))
                if rect.is_overlap(child.MBR):
                    total += self.region_query(rect, child)
            return total

    ############# Self develop function Start Here #############
    
    def __minimum_distance(self, pt, rect):
        assert(isinstance(pt,Point))
        assert(isinstance(rect,Rect))
        dx = max(rect.x1 - pt.x, 0, pt.x - rect.x2)
        dy = max(rect.y1 - pt.y, 0, pt.y - rect.y2)
        return math.sqrt(dx*dx+dy*dy)
    
    #Following function refer to http://www.cs.umd.edu/~nick/papers/nnpaper.pdf
    def __minmax_distance(self, pt, rect):
        assert(isinstance(pt,Point))
        assert(isinstance(rect,Rect))

        term_1_x = 0
        if (pt.x <= (rect.x1+rect.x2)/2):
            term_1_x = rect.x1
        else:
            term_1_x = rect.x2
        
        term_1_y = 0
        if (pt.y <= (rect.y1+rect.y2)/2):
            term_1_y = rect.y1
        else:
            term_1_y = rect.y2
        
        term_2_x = 0
        if (pt.x >= (rect.x1+rect.x2)/2):
            term_2_x = rect.x1
        else:
            term_2_x = rect.x2

        term_2_y = 0
        if (pt.y >= (rect.y1+rect.y2)/2):
            term_2_y = rect.y1
        else:
            term_2_y = rect.y2
        
        return min(math.sqrt((pt.x-term_1_x)*(pt.x-term_1_x)+(pt.y-term_2_y)*(pt.y-term_2_y)), (math.sqrt((pt.x-term_2_x)*(pt.x-term_2_x)+(pt.y-term_1_y)*(pt.y-term_1_y))))

    def __reset_nn_profile(self):
        self.nn_profile = {"numberOfNodeVisit":0, "numberOfPointCal":0, "prunedMBR": 0}
    
    def print_nn_profile(self):
        if (self.nn_profile is None):
            print("No profile found")
        else:
            print("Number of node visited: {}".format(self.nn_profile['numberOfNodeVisit']))
            print("Number of pt to q calculated (Except pt to rect): {}".format(self.nn_profile['numberOfPointCal']))
            print("Number of pruned MBR: {}".format(self.nn_profile['prunedMBR']))

    #Following function refer to http://dept.cs.williams.edu/~heeringa/publications/heeringa-tamc.pdf
    def neighest_neighbour_search(self, search_pt, nearest_neighbour=None, node=None):
        assert(isinstance(search_pt,Point))
        if node is None:
            self.__reset_nn_profile()
            node = self.root
            nearest_neighbour = {'pt':None, 'dist':sys.maxsize}
        
        if node.is_leaf():
            self.nn_profile['numberOfNodeVisit'] += 1
            for m in node.data_points:
                self.nn_profile['numberOfPointCal'] += 1
                dist = ptDistance(search_pt,m)
                if (dist < nearest_neighbour['dist']):
                    nearest_neighbour['pt'] = m
                    nearest_neighbour['dist'] = dist
        else:
            self.nn_profile['numberOfNodeVisit'] += 1
            ABL_list = []
            for child in node.child_nodes:
                ABL_list.append((child, self.__minimum_distance(search_pt,child.MBR)))
                
            #Sort by min dist
            ABL_list = sorted(ABL_list, key=lambda x:x[1])

            #H2 Pruning
            for node, min_dist in ABL_list:
                minmax_dist = self.__minmax_distance(search_pt, node.MBR)
                if (minmax_dist < nearest_neighbour['dist']):
                    nearest_neighbour['dist'] = minmax_dist

            for node, min_dist in ABL_list:
            #H3 Pruning
                if (min_dist < nearest_neighbour['dist']):
                    nearest_neighbour = self.neighest_neighbour_search(search_pt,nearest_neighbour,node)
                else:
                    self.nn_profile['prunedMBR'] += 1
        return nearest_neighbour 
                


    def __get_tree_height(self, node=None):
        if node is None:
            node = self.root
        
        if node.is_leaf():
            return 1
        else:
            depth = []
            for child in node.child_nodes:
                depth.append(self.__get_tree_height(child))
            return max(depth)+1

    def __get_leaf_count(self, node=None):
        if node is None:
            node = self.root

        if node.is_leaf():
            return {'leaf':1, 'non_leaf':0}
        else:
            total = {'leaf':0, 'non_leaf':1}
            for child in node.child_nodes:
                data = self.__get_leaf_count(child)
                total['leaf'] = total['leaf'] + data['leaf']
                total['non_leaf'] = total['non_leaf'] + data['non_leaf'] 
            return total
    
    def __dfs_bucket_utilization(self, node=None):
        if node is None:
            node = self.root
        if node.is_leaf():
            yield len(node.data_points)
        else:
            for child in node.child_nodes:
                yield from self.__dfs_bucket_utilization(child)

    def get_bucket_utilization(self):
        result = {"<20%":0, "20-80%":0, ">80%":0}
        for size in self.__dfs_bucket_utilization(self.root):
            percentage = int((size/self.max_bucket_size)*100)
            if (percentage < 20):
                result["<20%"] += 1
            elif (percentage > 80):
                result[">80%"] += 1
            else:
                result["20-80%"] += 1
        return result
    
    def get_overlap_mbr(self, node=None):
        if node is None:
            node = self.root
        if node.is_leaf():
            return 0
        else:
            total_over_lap = 0
            for child in node.child_nodes:
                total_over_lap += self.get_overlap_mbr(child)
            for i in range(len(node.child_nodes)):
                for j in range(i+1,len(node.child_nodes)):
                    if (node.child_nodes[i].MBR.is_overlap(node.child_nodes[j].MBR)):
                        total_over_lap += 1
            return total_over_lap

    def print_profile(self):
        print("Setting - Max Bucket Size: {} \t Max Sub Tree: {}".format(self.max_bucket_size, self.max_sub_tree))
        print("Tree Depth: {}".format(self.__get_tree_height()))
        print("Leaf Count: {}".format(self.__get_leaf_count()))
        print("Bucket Utilization: {}".format(self.get_bucket_utilization()))
        print("Number of non-leaf overlapping: {}".format(self.get_overlap_mbr()))
########################################################################################


def getRTree(max_bucket_size, max_sub_tree, filename):
    R_tree = None
    need_create = True
    storage_path = "r_tree_s_{}_b_{}.obj".format(max_sub_tree, max_bucket_size)
    try:
        if (os.path.exists(storage_path)):
            file_handler = open(storage_path,'rb')
            R_tree = pickle.load(file_handler)

            print("=========== R-Tree Found Info ==============")
            R_tree.print_profile()
            print("============================================")
            
            ans = input("Would you like to recreate R_tree? (y/n) ")
            if (ans.lower() != 'y'):
                need_create = False
    except Exception as e:
        print(e)
        print("Trying to resconstruct RTree")
        need_create = True


    if (need_create == True):
        dataframe = pd.read_csv(filename, header=None, names=['X','Y'])
        R_tree = RTree(max_bucket_size, max_sub_tree)
        for index, row in tqdm(dataframe.iterrows(), total=dataframe.shape[0]):
            R_tree.insert_point(Point(index,row['X'],row['Y']))
        
        print("Saving R_tree into "+str(storage_path)+"... ")
        file_handler = open(storage_path,'wb')
        pickle.dump(R_tree, file_handler)

        print("=========== R-Tree Created Info ==============")
        R_tree.print_profile()
        print("=============================================")

    return R_tree


if __name__ == "__main__":
    for bucket_size in BUCKET_SIZE:
        for sub_tree in SUB_TREE:
            print("Getting R Tree (Bucket Size(b): {} Sub Tree(d): {})....".format(bucket_size,sub_tree))
            R_Tree = getRTree(bucket_size,sub_tree, FILE_NAME)

            df = pd.read_csv(FILE_NAME, header=None, names=['X','Y'])
            x = df['X']
            y = df['Y']
            min_x = x.min()
            max_x = x.max()
            min_y = y.min()
            max_y = y.max()
            for i in range(10):
                x = random.uniform(min_x-1, max_x+1)
                y = random.uniform(min_y-1, max_y+1)
                query_pt = Point("Query pt", x,y)
                nn = R_Tree.neighest_neighbour_search(query_pt)

                print("========== Query {} ==========".format(i+1))
                print("Query Pt: {}".format(query_pt))
                print("NN Point: {}".format(nn['pt']))
                print("Pt Distance: {}".format(nn['dist']))
                R_Tree.print_nn_profile()
                print("==============================")



