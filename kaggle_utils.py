import numpy as np
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import math

def box_nms(rects, overlap_thr):

    assert(len(rects) != 0), 'There are no boxes in the list '

    boxes = np.array(rects, dtype=np.float32)

    passed = []

    # extract the top left and bottom right co-ordinates of all boxes
    x_tl = boxes[:, 0]
    y_tl = boxes[:, 1]
    x_br = boxes[:, 2]
    y_br = boxes[:, 3]

    # sort the indexes boxes in terms of their bottom right y coordinate
    sorted_y_br_idx = np.argsort(y_br)

    # area of all boxes
    all_area = (x_br-x_tl+1) * (y_br-y_tl+1)

    # find the overlapping boxes with the box that is last in the sorted list.
    # throw away those whose overlap area exceeds the threshold
    print('Original no of boxes = {:d}'.format(len(rects)))
    while(len(sorted_y_br_idx) > 0):
        # take the last box and retain it
        last_box_idx = sorted_y_br_idx[-1]
        passed.append(last_box_idx)

        # find the overlap of all boxes with the last box
        dx1 = np.maximum(x_tl[last_box_idx], x_tl[sorted_y_br_idx[:-1]])  
        # exclue the last index as that is the rectange against whom we are computing overlap
        dy1 = np.maximum(y_tl[last_box_idx], y_tl[sorted_y_br_idx[:-1]])
        dx2 = np.minimum(x_br[last_box_idx], x_br[sorted_y_br_idx[:-1]])
        dy2 = np.minimum(y_br[last_box_idx], y_br[sorted_y_br_idx[:-1]])

        # overlap segment of the edges. Make sure all are +ve
        dx = np.maximum(dx2 - dx1 + 1, 0)
        dy = np.maximum(dy2 - dy1 + 1, 0)
        # overlap area
        ov_area = dx * dy
        # overlap percentage
        ov_percent = ov_area / all_area[last_box_idx]
        # now delete all boxes which have overlap percent more than the threshold
        redundant_idx = np.where(ov_percent > overlap_thr)[0]
        # we alreaded retained the last index. hence we can delete that too
        redundant_idx = np.concatenate(([len(sorted_y_br_idx)-1], redundant_idx))
        sorted_y_br_idx = np.delete(sorted_y_br_idx, redundant_idx)
        print('Remaining boxes = {:d}'.format(len(sorted_y_br_idx)))

    boxes = boxes[passed].astype(np.int)
    return boxes.tolist()

def plot_mean_centroids(cent_list, obj_info):
    plt.scatter(*zip(*cent_list), marker='o', color='r')
    plt.axis([0, 640, 0, 480])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title(obj_info)
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    plt.show()

def plot_catwise_centroids(obj_dict, obj_type, cat_range=(0,10)):
    classes = ('c0', 'c1', 'c2', 'c3','c4','c5','c6','c7','c8','c9')
    rect_list = [[] for i in range(10)]
    rect_cent = [[] for i in range(10)]

    for img, objs in obj_dict.iteritems():
        cat_idx = classes.index(objs['cls'])
        rects = objs[obj_type]
        if(rects):
            if(isinstance(rects[0], list)):
                for rec in rects:
                    rect_list[cat_idx].append(rec[:4])
            else:
                rect_list[cat_idx].append(rects[:4])

    for c, wl in enumerate(rect_list):
        for w in wl:
            cent = [((w[2]-w[0])/2 + w[0]), ((w[3]-w[1])/2 + w[1])]
            rect_cent[c].append(cent[:])

    #colors = cm.rainbow(np.linspace(0, 1, 10))
    colors = ('#080808', '#DAF7A6', '#EAF505', '#05F510', '#05F1F5', '#0D05F5', '#F505EA', '#633974', '#95A5A6', '#F50526')
    plt.axis([0, 640, 0, 480])
    plt.xlabel('X')
    plt.ylabel('Y')
    plt.title('{:s} centroids'.format(obj_type))
    ax = plt.gca()
    ax.set_ylim(ax.get_ylim()[::-1])
    ax.xaxis.tick_top()
    ax.yaxis.tick_left()
    for c in range(*cat_range):
        if (len(rect_cent[c])):
            plt.scatter(*zip(*rect_cent[c]), marker='o', color=colors[c], label=classes[c])
    plt.legend(loc=2)
    plt.show()


def plot_objpair_dist_histogram(obj_dict, obj0, obj1, sep=True, cat_range=(0,10)):
    classes = ('c0', 'c1', 'c2', 'c3','c4','c5','c6','c7','c8','c9')
    dist_dict = {}
    for c in classes:
        dist_dict[c] = []

    for img, objs in obj_dict.iteritems():
        if(len(objs[obj0]) == 0 or len(objs[obj1]) == 0):
            continue
        # if more than one object of the type is present, take the first one
        if(isinstance(objs[obj0][0], list)):
            obj0_box = objs[obj0][0][:4]
        else:
            obj0_box = objs[obj0][:4]
        if(isinstance(objs[obj1][0], list)):
            obj1_box = objs[obj1][0][:4]
        else:
            obj1_box = objs[obj1][:4]

        # compute head and steering centroid.
        obj0_c = ((obj0_box[2] - obj0_box[0])/2.0 + obj0_box[0], (obj0_box[3] - obj0_box[1])/2.0 + obj0_box[1])
        obj1_c = ((obj1_box[2] - obj1_box[0])/2.0 + obj1_box[0], (obj1_box[3] - obj1_box[1])/2.0 + obj1_box[1])
        # find the distance
        dist = math.sqrt((obj1_c[0]-obj0_c[0])**2 + (obj1_c[1]-obj0_c[1])**2)
        dist_dict[objs['cls']].append(dist)

    colors = ('#080808', '#DAF7A6', '#EAF505', '#05F510', '#05F1F5', '#0D05F5', '#F505EA', '#633974', '#95A5A6', '#F50526')
    if(not sep):
        for c in range(*cat_range):
            plt.hist(dist_dict[classes[c]], 100, color=colors[c], label=classes[c])
        plt.legend(loc=2)
        plt.grid(True)
    else:
        fig, ax = plt.subplots(2, 5, sharex=True, sharey=True)
        for c in range(*cat_range):
            ax[c/5][c%5].hist(dist_dict[classes[c]], 50, color=colors[c], label=classes[c])
            ax[c/5][c%5].set_title(classes[c])
            ax[c/5][c%5].grid(True)
        
    plt.xlabel('Distance btw {:s} and {:s}'.format(obj0, obj1))
    plt.ylabel('Frequency of occurance')
    plt.show()
        
if __name__=='__main__':
    print('Nothing to execute')
