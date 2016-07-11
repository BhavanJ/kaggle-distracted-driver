import numpy as np
from kaggle_utils import plot_mean_centroids, plot_catwise_centroids, plot_objpair_dist_histogram
from collections import OrderedDict
import math

_OBJS = ('head', 'wrist', 'steering', 'radio', 'phone','cup',
    'left_hand_steering', 'right_hand_steering', 'both_hands_steering', 'right_hand_phone', 'left_hand_phone', 'drinking_near_steering')

CLASSES = ('c0', 'c1', 'c2', 'c3','c4','c5','c6','c7','c8','c9')

_FEATS = OrderedDict({
    'f0' : ('Presense of steering with both hands', 'both_hands_steering'),
    'f1' : ('Presense of steering with left hand', 'left_hand_steering'),
    'f2' : ('Presense of steering with right hand','right_hand_steering'),
    'f3' : ('Presense of right hand with phone','right_hand_phone'),
    'f4' : ('Presense of left hand with phone','left_hand_phone'),
    'f5' : ('Presense of phone', 'phone'),
    'f6' : 'Left wrist present',
    'f7' : 'Right wrist preset',
    'f8' : 'Bottle/can present',
    'f9' : 'Steering present',
    'f10' : 'Distance btw head and steering centroid',
    'f11' : 'Distance btw  wrist and steering',
    'f12' : 'Distance btw head and phone',
    'f13' : 'Distance btw left wrist and head',
    'f14' : 'Distance btw right wrist and head',
    'f15' : 'Centroid/Qudrant of right wrist(operating radio class)',
    'f16' : 'Wrist present inside radio?',
    'f17' : 'Wrist present inside head object?',
    'f18' : ('Cup present?', 'cup'),
    'f19' : ('Presense of drinking near steering?', 'drinking_near_steering')})

def _overlap_area(gt_rect, det_rect):
    """Computes overlap area percentage between ground truth box and detected box
    gt_rect  : [xmin, ymain, xmax, ymax]
    det_rect : [xmin, ymain, xmax, ymax]
    """
    dx = min(gt_rect[2], det_rect[2]) - max(gt_rect[0] , det_rect[0])
    dy = min(gt_rect[3], det_rect[3]) - max(gt_rect[1] , det_rect[1])
    if dx > 0 and dy > 0:
        overlap_area = dx*dy
    else:
        overlap_area = 0.0

    gt_area = (gt_rect[2] - gt_rect[0]) * (gt_rect[3] - gt_rect[1])
  
    return float(overlap_area) / gt_area

def remove_multiple_detections(obj_list, max_cnt, overlap_thr):
    
    remove_list = []
    final_list = []
    for o in range(len(obj_list)-1):
        for ro in range(o, len(obj_list)-1):
            overlap = _overlap_area(obj_list[ro][:4], obj_list[ro+1][:4])
            if(overlap > overlap_thr):
                r = ro if obj_list[ro][4] < obj_list[ro+1][4] else ro+1
                if(r not in remove_list):
                    remove_list.append(r)

    for o in range(len(obj_list)):
        if(o not in remove_list):
            final_list.append(obj_list[o])

    filt_list = []
    if(len(final_list) > max_cnt):
        scores = [o[4] for o in final_list]
        sort_scores = sorted(scores, reverse=True)
        keep_idx = [scores.index(s) for s in sort_scores[:max_cnt]]
        for i in keep_idx:
            filt_list.append(final_list[i])
    else:
        filt_list = final_list

    return filt_list
            
    
def _get_object_list(obj_dict):
    
    sample_entry = obj_dict[obj_dict.keys()[0]]
    obj_list = sample_entry.keys()
    obj_list.remove('cls')
    return obj_list

def _get_mean_centroid(obj_dict, obj_type):
    """ returns mean centroid of given object type for all 10 classes. Also returns the dictionary containing centroid of head for all
    images, replacing the missing values by the mean
    """
    def __get_cat_mean(obj_dict, obj_type, cat):
        mean_centroid = [0., 0.]
        mean_box = [0.0 for i in range(5)]
        cnt = 0
        for img, objs in obj_dict.iteritems():
            if(objs['cls'] == cat ):
                if(len(objs[obj_type]) == 1):
                    mean_centroid[0] += ((objs[obj_type][0][2] - objs[obj_type][0][0])/2.0 + objs[obj_type][0][0])
                    mean_centroid[1] += ((objs[obj_type][0][3] - objs[obj_type][0][1])/2.0 + objs[obj_type][0][1])
                    mean_box = [mean_box[i]+objs[obj_type][0][i] for i in range(5)]
                    cnt += 1
                elif(len(objs[obj_type]) > 1):
                    # take the box with max score
                    scores = [o[4] for o in objs[obj_type]]
                    top_box = objs[obj_type][scores.index(max(scores))]
                    mean_centroid[0] += ((top_box[2] - top_box[0])/2.0 + top_box[0])
                    mean_centroid[1] += ((top_box[3] - top_box[1])/2.0 + top_box[1])
                    mean_box = [mean_box[i]+top_box[i] for i in range(5)]
                    cnt += 1
                else:
                    pass
        mean_box = [mean_box[i]/cnt for i in range(5)]
        return [mean_centroid[0]/cnt, mean_centroid[1]/cnt], mean_box

    mean_cent = []
    mean_box = []
    for cls in CLASSES:
        mean_c, mean_b = __get_cat_mean(obj_dict, obj_type, cls)
        mean_cent.append(mean_c)
        mean_box.append(mean_b)

    return mean_cent, mean_box

def  _recover_mandatory_objs(combined_objs, head_mean_box, steering_mean_box, train=True):
    recovered_obj = combined_objs
    # recover head and steering
    for img, objs in recovered_obj.iteritems():
        if(train):
            cls_idx = CLASSES.index(objs['cls'])
        else:
            # test samples do not have any class info. Hence we use mean of c0 category.
            # TODO: instead of c0 category, replace by mean across all categories.
            cls_idx = 0

        if(len(objs['head']) == 0):
            recovered_obj[img]['head'] = [head_mean_box[cls_idx]]
        if(len(objs['steering']) == 0):
            recovered_obj[img]['steering'] = [steering_mean_box[cls_idx]]

    return recovered_obj

def _behind_head(obj, head_c, obj_cls, train=True):
    obj_c_x = -1.0
    behind = False
    if(train):
        hc_x = head_c[CLASSES.index(obj_cls)][0]
    else:
        # test samples do not have any class info. Hence we use mean of c0 category.
        # TODO: instead of c0 category, replace by mean across all categories.
        hc_x = head_c[0][0]

    obj_c_x = (obj[2] - obj[0])/2.0 + obj[0]
    if(obj_c_x < hc_x):
        behind = True

    return behind

def _filter_detections(obj_dict, head_mean_c, steering_mean, train=True):

    filtered_dict = obj_dict
    def __filter_head(obj_dict):
        hf_dict = obj_dict
        for img, objs in hf_dict.iteritems():
            # if more than one head is detected, consider one with highest score
            if(len(objs['head']) > 1):
                scores = [h[4]  for h in objs['head']]
                hf_dict[img]['head'] = objs['head'][scores.index(max(scores))]
            elif(len(objs['head']) == 1):
                # just convert list of list to single list
                hf_dict[img]['head'] = objs['head'][0]
            else:
                # there are no head detected for this image. need to replace by mean of centroid
                pass

        return hf_dict

    def __filter_wrists(obj_dict, head_c, train=True):
        wf_dict = obj_dict
        # TODO: first remove all wrist detections which are behind the head and whose score is less than threshold
        
        for img, objs in wf_dict.iteritems():
            wrists = objs['wrist']
            to_remove = []
            remain_wrists = []
            for i, wr in enumerate(wrists):
                if((_behind_head(wr, head_c, objs['cls'], train)) or (wr[4] < 0.95)):
                    to_remove.append(i)
            for i, box in enumerate(wrists):
                if(i not in to_remove):
                    remain_wrists.append(wrists[i])
            # 
            # kind of NMS on the objects
            if(len(remain_wrists) >= 2):
                wf_dict[img]['wrist'] = remove_multiple_detections(remain_wrists, 2, 0.7)
            else:
                wf_dict[img]['wrist'] = remain_wrists

        return wf_dict

    def __filter_steering(obj_dict):
        sf_dict = obj_dict
        for img, objs in sf_dict.iteritems():
            if(len(objs['steering']) > 1):
                scores = [s[4]  for s in objs['steering']]
                sf_dict[img]['steering'] = objs['steering'][scores.index(max(scores))]
            elif(len(objs['steering']) == 1):
                # just convert list of list to single list
                sf_dict[img]['steering'] = objs['steering'][0]
            else:
                # there are no head detected for this image. need to replace by mean of centroid
                pass

        return sf_dict
   
    def __filter_hands_steering(obj_dict):
        """'left_hand_steering', 'right_hand_steering', 'both_hands_steering' all 3 objects, ideally cannot
        occur together. If they happen to occur, then we will take the one with max score. 
        """
        hsf_dict = obj_dict
        to_filter = ('left_hand_steering', 'right_hand_steering', 'both_hands_steering')
        for img, objs in hsf_dict.iteritems():
            # first make sure that all 3 objects have single detections. Take one with max score in case of multiple detections
            for obj_cls in to_filter:
                if(len(objs[obj_cls]) > 1):
                    scores = [b[4] for b in objs[obj_cls]]
                    hsf_dict[img][obj_cls] = objs[obj_cls][scores.index(max(scores))]
                elif(len(objs[obj_cls]) == 1):
                    hsf_dict[img][obj_cls] = objs[obj_cls][0]
                else:
                    pass
            
        for img, objs in hsf_dict.iteritems():
            scores = []
            for obj_cls in to_filter:
                if(objs[obj_cls]): # if the list is not empty.
                    scores.append(objs[obj_cls][4])
            if(len(scores) > 1):
                keep_obj = to_filter[scores.index(max(scores))]
                for obj_cls in to_filter:
                    if obj_cls != keep_obj:
                        hsf_dict[img][obj_cls] = []
        return hsf_dict

    def __filter_phone_with_hands(obj_dict):
        """ right hand with phone and left hand with phone is unusual. Remove double detections.
        """
        pf_dict = obj_dict
        to_filter = ('right_hand_phone', 'left_hand_phone')
        for img, objs in pf_dict.iteritems():
            # first make sure that all  objects have single detections. Take one with max score in case of multiple detections
            for obj_cls in to_filter:
                if(len(objs[obj_cls]) > 1):
                    scores = [b[4] for b in objs[obj_cls]]
                    pf_dict[img][obj_cls] = objs[obj_cls][scores.index(max(scores))]
                elif(len(objs[obj_cls]) == 1):
                    pf_dict[img][obj_cls] = objs[obj_cls][0]
                else:
                    pass
            
        for img, objs in pf_dict.iteritems():
            scores = []
            for obj_cls in to_filter:
                if(objs[obj_cls]): # if the list is not empty.
                    scores.append(objs[obj_cls][4])
            if(len(scores) > 1):
                keep_obj = to_filter[scores.index(max(scores))]
                for obj_cls in to_filter:
                    if obj_cls != keep_obj:
                        pf_dict[img][obj_cls] = []
        return pf_dict

    def __filter_phone_or_cup(obj_dict, head_c, pc, thr=0.95, train=True):
        pf_dict = obj_dict
        top_score = 0
        for img, objs in pf_dict.iteritems():
            top_score = 0.0
            pc_x = -1.0
            if(train):
                hc_x = head_c[CLASSES.index(objs['cls'])][0]
            else:
                # test samples do not have any class info. Hence we use mean of c0 category.
                # TODO: instead of c0 category, replace by mean across all categories.
                hc_x = head_c[0][0]
            #print objs['phone']
            if(len(objs[pc]) > 1):
                scores = [h[4]  for h in objs[pc]]
                pf_dict[img][pc] = objs[pc][scores.index(max(scores))]
                top_score = max(scores)
                # x co-ordinate of the box centroid
                pc_x = (pf_dict[img][pc][2] - pf_dict[img][pc][0])/2.0 + pf_dict[img][pc][0]
            elif(len(objs[pc]) == 1):
                top_score = objs[pc][0][4]
                pf_dict[img][pc] = objs[pc][0]
                pc_x = (pf_dict[img][pc][2] - pf_dict[img][pc][0])/2.0 + pf_dict[img][pc][0]
            else:
                pass
            # threshold for phone detection
            # discard all phone detections which are behind the head
            if(top_score < thr or pc_x < hc_x):
                pf_dict[img][pc] = []
                
            

        return pf_dict

    #for img, objs in filtered_dict.iteritems():
    #    if(len(objs['head']) > 1):
    #        print('Image contains multiple head detections {:s} , {:d}'.format(img, len(objs['head'])))

    print('Filtering head...')
    filtered_dict = __filter_head(filtered_dict)
    print('Filtering steering...')
    filtered_dict = __filter_steering(filtered_dict)
    print('Filtering hands on steering objects...')
    filtered_dict = __filter_hands_steering(filtered_dict)
    print('Filtering multiple detections of phone with hands...')
    #filtered_dict =  __filter_phone_with_hands(filtered_dict)
    print('Filtering multiple detections of phone...')
    filtered_dict =  __filter_phone_or_cup(filtered_dict, head_mean_c, 'phone', thr=0.95, train=train)
    print('Filtering multiple detections of cup...')
    #filtered_dict =  __filter_phone_or_cup(filtered_dict, head_mean_c, 'cup', thr=0.95, train=train)
    print('Filtering multiple detections of radio...')
    # just reuse same method for radio also
    filtered_dict =  __filter_phone_or_cup(filtered_dict, head_mean_c, 'radio', thr=0.95, train=train)
    print('Filtering wrists...')
    filtered_dict = __filter_wrists(filtered_dict, head_mean_c, train=train)
    print('Filtering of objects finished...')
    # TODO: all wrists and phones whose centroids are behind should be removed

    return filtered_dict

def create_boolean_features(obj_dict, feat, feat_dict):
    obj_type = _FEATS[feat][1]
    for img, objs in obj_dict.iteritems():
        # if the list is not null
        if(objs[obj_type]):
            feat_dict[img][feat] = 1.0
        else:
            feat_dict[img][feat] = 0.0

    return feat_dict

def distance_btw_head_steering(obj_dict, feat, feat_dict):
    dist_list = []
    for img, objs in obj_dict.iteritems():
        # compute head and steering centroid.
        head_c = ((objs['head'][2] - objs['head'][0])/2.0 + objs['head'][0], 
            (objs['head'][3] - objs['head'][1])/2.0 + objs['head'][1])
        steer_c = ((objs['steering'][2] - objs['steering'][0])/2.0 + objs['steering'][0], 
            (objs['steering'][3] - objs['steering'][1])/2.0 + objs['steering'][1])
        # find the distance
        dist = math.sqrt((steer_c[0]-head_c[0])**2 + (steer_c[1]-head_c[1])**2)
        feat_dict[img][feat] = dist
        dist_list.append(dist)

    # normalize the feature
    max_dist = max(dist_list)
    min_dist = min(dist_list)
    for img, feats in feat_dict.items():
        feat_dict[img][feat] = (feat_dict[img][feat]-min_dist) / max_dist
    return feat_dict

def objs_vicinity_feature(obj_dict, feat, feat_dict, pri_obj, sec_obj):
    for img, objs in obj_dict.iteritems():
        feat_val = 0.0
        if((len(objs[sec_obj]) != 0) and (len(objs[pri_obj]) != 0)):
            po = objs[pri_obj]
            if(isinstance(objs[sec_obj][0], list)):
                ov = []
                for so in objs[sec_obj]:
                    overlap = _overlap_area(po[:4], so[:4])
                    ov.append(overlap)
                feat_val = max(ov)
            else:
                feat_val = _overlap_area(po[:4], objs[sec_obj][:4])

        feat_dict[img][feat] = feat_val

    return feat_dict

def wrist_radio_vicinity(obj_dict, feat, feat_dict):
    for img, objs in obj_dict.iteritems():
        feat_val = 0.0
        if(len(objs['wrist']) != 0 and len(objs['radio']) != 0):
            radio = objs['radio']
            ov = []
            for wrist in objs['wrist']:
                overlap = _overlap_area(radio[:4], wrist[:4])
                #if(overlap > 0.5):
                #    feat_val = 1.0
                
        feat_dict[img][feat] = feat_val

    return feat_dict

def compute_features(obj_dict_list, img_cls_dict, train=True, **kwargs):
    """ Given the list of dictionaries, with each dictionary containing some objects detected for each train/val/test image,
    this method computes the features required for the decision tree by combinining all object detections and interpretations.

    img_cls_dict = {'img_name': class_code, 'img_name':class_code, ....}

    ---------
    return : X, y : features and class code(0 to 9)
    """
    no_dict = len(obj_dict_list)
    print('You provided {:d} separate object dictionaries.'.format(no_dict))
    assert(no_dict > 0), 'Provide atleast one object set dictionary'

    
    no_imgs = len(obj_dict_list[0])
    assert(len(img_cls_dict) == no_imgs ), 'Images in object dict != images in class dict'

    for d in range(1, no_dict):
        assert(len(obj_dict_list[d]) == no_imgs), 'Some object dictionaries have less/more no of images compared to others.'

            
    # combine all object dictionaries
    combined_objs = {}
    for name, cls in img_cls_dict.iteritems():
        combined_objs[name] = {'cls': cls}
        for d in obj_dict_list:
            combined_objs[name].update(d[name])
    print('Combined all object dictionaries')
    obj_names = _get_object_list(combined_objs)
    print('Objects found in the pickle files are'),
    print(obj_names)

    # compute mean centroid of must present objects and substitute
    # the category mean for missing mandatory objects
    if(train):
        # mean centroid of head for all categories
        head_mean_c, head_mean_box = _get_mean_centroid(combined_objs, 'head')

        # mean centroid of steering for all categories
        steering_mean_c, steering_mean_box = _get_mean_centroid(combined_objs, 'steering')

        # recover mandatory objects(head and steering) using the mean centroid
        combined_objs = _recover_mandatory_objs(combined_objs, head_mean_box, steering_mean_box)
        
        # object filtering
        filtered_objs = _filter_detections(combined_objs, head_mean_c, steering_mean_c)
    else:
        # If the feature computation is on test set, replace
        # the missing items by the centroid computed on training set.
        # object filtering
        head_mean_c = kwargs['head_mean_c']
        steering_mean_c = kwargs['steering_mean_c']
        head_mean_box = kwargs['head_mean_box']
        steering_mean_box = kwargs['steering_mean_box']
        # recover mandatory objects(head and steering) using the mean centroid
        combined_objs = _recover_mandatory_objs(combined_objs, head_mean_box, steering_mean_box, train=False)
        filtered_objs = _filter_detections(combined_objs, head_mean_c, steering_mean_c, train=False)
        


    #plot_mean_centroids(head_mean_c, 'Centroid of head')
    # initialize feature dictionary
    feat_dict = {}
    for img_name, objs in filtered_objs.iteritems():
        feat_dict[img_name] = {'cls': objs['cls']}

    # compute boolean features
    feat_dict = create_boolean_features(filtered_objs, 'f0', feat_dict)
    feat_dict = create_boolean_features(filtered_objs, 'f1', feat_dict)
    feat_dict = create_boolean_features(filtered_objs, 'f2', feat_dict)
    feat_dict = create_boolean_features(filtered_objs, 'f3', feat_dict)
    #feat_dict = create_boolean_features(filtered_objs, 'f4', feat_dict)
    feat_dict = create_boolean_features(filtered_objs, 'f5', feat_dict)
    #feat_dict = create_boolean_features(filtered_objs, 'f18', feat_dict)
    #feat_dict = create_boolean_features(filtered_objs, 'f19', feat_dict)
    # distance btw head and steering
    #feat_dict = distance_btw_head_steering(filtered_objs, 'f10', feat_dict)
    objs_vicinity_feature(filtered_objs, 'f17', feat_dict, 'head', 'wrist')
    objs_vicinity_feature(filtered_objs, 'f16', feat_dict, 'radio', 'wrist')
    #objs_vicinity_feature(filtered_objs, 'f11', feat_dict, 'steering', 'wrist')
    objs_vicinity_feature(filtered_objs, 'f12', feat_dict, 'head', 'phone')

    #print filtered_objs
    #plot_catwise_centroids(filtered_objs, 'head')
    #plot_catwise_centroids(filtered_objs, 'steering')
    #plot_catwise_centroids(filtered_objs, 'phone')
    #plot_catwise_centroids(filtered_objs, 'left_hand_steering')
    #plot_catwise_centroids(filtered_objs, 'right_hand_phone')
    #plot_catwise_centroids(filtered_objs, 'wrist', cat_range=(8,9))
    #plot_catwise_centroids(filtered_objs, 'radio', cat_range=(0,10))
    #plot_objpair_dist_histogram(filtered_objs, 'head', 'phone', cat_range=(0,10))
    if(train):
        mean_model = [head_mean_c, steering_mean_c, head_mean_box, steering_mean_box]
        return feat_dict, mean_model
    else:
        return feat_dict

    

if __name__=='__main__':
    print _FEATS['f0']
