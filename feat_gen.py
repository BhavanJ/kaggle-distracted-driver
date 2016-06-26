import numpy as np
_OBJS = ('head', 'wrist', 'steering', 'radio', 'phone',
    'left_hand_steering', 'right_hand_steering', 'both_hands_steering', 'right_hand_phone')

_CLASSES = ('c0', 'c1', 'c2', 'c3','c4','c5','c6','c7','c8','c9')

_FEATS = ('Presense of steering with both hands', 'Presense of steering with left hand',
    'Presense of steering with right hand', 'Presense of right hand with phone',
    'Presense of left hand with phone', 'Presense of phone' 'Left wrist present',
    'Right wrist preset', 'Bottle/can present', 'Steering present', 'Head present'
    'Distance btw left wrist and steering', 'Distance btw right wrist and steering',
    'Distance btw left wrist and head', 'Distance btw right wrist and head',
    'Centroid/Qudrant of left wrist(operating radio class)')

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
    """ returns mean centroid of head for all 10 classes. Also returns the dictionary containing centroid of head for all
    images, replacing the missing values by the mean
    """
    def __get_cat_head_mean(obj_dict, obj_type, cat):
        mean_centroid = [0., 0.]
        cnt = 0
        for img, objs in obj_dict.iteritems():
            if(objs['cls'] == cat and len(objs[obj_type]) != 0):
                mean_centroid[0] += (objs[obj_type][2] - objs[obj_type][0])/2.0
                mean_centroid[1] += (objs[obj_type][3] - objs[obj_type][1])/2.0
                cnt += 1
        return [mean_centroid[0]/cnt, mean_centroid[1]/cnt]

    mean_cent = []
    for cls in _CLASSES:
        mean_cent.append(__get_cat_head_mean(obj_dict, obj_type, cls))

    return mean_cent

def _get_mean_steering_centroid(obj_dict):
    """ returns mean value of centroid of steering for all 10 classes. Also returns the dictionary containing centroid of steering for all
    images, replacing the missing values by the mean
    """
def _filter_detections(obj_dict):

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

    def __filter_wrists(obj_dict):
        wf_dict = obj_dict
        for img, objs in wf_dict.iteritems():
            wrists = objs['wrist']
            # kind of NMS on the objects
            if(len(wrists) >= 2):
                wf_dict[img]['wrist'] = remove_multiple_detections(wrists, 2, 0.7)
            else:
                pass

        return hf_dict

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
        # TODO
        return pf_dict

    def __filter_phone(obj_dict):
        pf_dict = obj_dict
        for img, objs in pf_dict.iteritems():
            if(len(objs['phone']) > 1):
                scores = [h[4]  for h in objs['phone']]
                pf_dict[img]['phone'] = objs['phone'][scores.index(max(scores))]
            elif(len(objs['phone']) == 1):
                pf_dict[img]['phone'] = objs['phone'][0]
            else:
                pass

        return pf_dict

    for img, objs in filtered_dict.iteritems():
        if(len(objs['head']) > 1):
            print('Image contains multiple head detections {:s} , {:d}'.format(img, len(objs['head'])))

    print('Filtering head...')
    filtered_dict = __filter_head(filtered_dict)
    print('Filtering steering...')
    filtered_dict = __filter_steering(filtered_dict)
    print('Filtering hands on steering objects...')
    filtered_dict = __filter_hands_steering(filtered_dict)
    print('Filtering multiple detections of phone with hands...')
    filtered_dict =  __filter_phone_with_hands(filtered_dict)
    print('Filtering multiple detections of phone...')
    filtered_dict =  __filter_phone(filtered_dict)
    print('Filtering of objects finished...')

    return filtered_dict

def compute_features(obj_dict_list, img_cls_dict):
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
    filtered_objs = _filter_detections(combined_objs)

    # mean centroid of head for all categories
    head_mean_c = _get_mean_centroid(filtered_objs, 'head')
    print(head_mean_c)

    # mean centroid of steering for all categories
    steering_mean_c = _get_mean_centroid(filtered_objs, 'steering')
    print(steering_mean_c)


