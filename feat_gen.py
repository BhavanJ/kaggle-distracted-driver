import numpy as np
_OBJS = ('head', 'wrist', 'steering', 'radio', 'phone',
    'left_hand_steering', 'right_hand_steering', 'both_hands_steering', 'right_hand_phone')

_FEATS = ('Presense of steering with both hands', 'Presense of steering with left hand',
    'Presense of steering with right hand', 'Presense of right hand with phone',
    'Presense of left hand with phone', 'Presense of phone' 'Left wrist present',
    'Right wrist preset', 'Bottle/can present', 'Steering present', 'Head present'
    'Distance btw left wrist and steering', 'Distance btw right wrist and steering',
    'Distance btw left wrist and head', 'Distance btw right wrist and head',
    'Centroid/Qudrant of left wrist(operating radio class)')

def _get_mean_head_centroid(obj_dict):
    """ returns mean centroid of head for all 10 classes. Also returns the dictionary containing centroid of head for all
    images, replacing the missing values by the mean
    """

def _get_mean_steering_centroid(obj_dict):
    """ returns mean value of centroid of steering for all 10 classes. Also returns the dictionary containing centroid of steering for all
    images, replacing the missing values by the mean
    """


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

    
    no_imgs = len(obj_dict_list[0].keys())
    assert(len(img_cls_dict.keys()) == no_imgs ), 'Images in object dict != images in class dict'

    for d in range(1, no_dict):
        assert(len(obj_dict_list[d].keys()) == no_imgs), 'Some object dictionaries have less/more no of images compared to others.'

            
    # combine all object dictionaries
    combined_objs = {}
    for name, cls in img_cls_dict.iteritems():
        combined_objs[name] = {'cls': cls}
        for d in obj_dict_list:
            combined_objs[name].update(d[name])
    print('Combined all object dictionaries')


