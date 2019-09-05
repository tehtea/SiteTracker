'''
Script to test traffic light localization and detection
'''

import numpy as np
import tensorflow as tf
from PIL import Image
import os
from matplotlib import pyplot as plt
import matplotlib
import time
from glob import glob
import logging

import helpers

NON_MAXIMUM_SUPPRESSION_THRESHOLD = 0.2

cwd = os.path.dirname(os.path.realpath(__file__))

# Uncomment the following two lines if need to use the visualization_tunitls
#os.chdir(cwd+'/models')
import visualization_utils

class PersonDetector(object):
    def __init__(self):

        self.car_boxes = []
        
        os.chdir(cwd)
        
        #Tensorflow localization/detection model
        # Single-shot-dectection with mobile net architecture trained on COCO
        # dataset
        detect_model_name = 'ssd_mobilenet_v1_coco_2017_11_17'
        
        PATH_TO_CKPT = detect_model_name + '/frozen_inference_graph.pb'
        
        # setup tensorflow graph
        self.detection_graph = tf.Graph()
        
        # configuration for possible GPU use
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        # load frozen tensorflow detection model and initialize 
        # the tensorflow graph
        with self.detection_graph.as_default():
            od_graph_def = tf.GraphDef()
            with tf.gfile.GFile(PATH_TO_CKPT, 'rb') as fid:
               serialized_graph = fid.read()
               od_graph_def.ParseFromString(serialized_graph)
               tf.import_graph_def(od_graph_def, name='')
               
            self.sess = tf.Session(graph=self.detection_graph, config=config)
            self.image_tensor = self.detection_graph.get_tensor_by_name('image_tensor:0')
              # Each box represents a part of the image where a particular object was detected.
            self.boxes = self.detection_graph.get_tensor_by_name('detection_boxes:0')
              # Each score represent how level of confidence for each of the objects.
              # Score is shown on the result image, together with the class label.
            self.scores =self.detection_graph.get_tensor_by_name('detection_scores:0')
            self.classes = self.detection_graph.get_tensor_by_name('detection_classes:0')
            self.num_detections =self.detection_graph.get_tensor_by_name('num_detections:0')
    
    # Helper function to convert image into numpy array    
    def load_image_into_numpy_array(self, image):
         (im_width, im_height) = image.size
         return np.array(image.getdata()).reshape(
            (im_height, im_width, 3)).astype(np.uint8)       
    # Helper function to convert normalized box coordinates to pixels
    def box_normal_to_pixel(self, box, dim):
    
        height, width = dim[0], dim[1]
        box_pixel = [int(box[0]*height), int(box[1]*width), int(box[2]*height), int(box[3]*width)]
        return np.array(box_pixel)       
        
    def get_localization(self, image, visual=False):  
        """Determines the locations of the traffic light in the image

        Args:
            image: camera image

        Returns:
            list of bounding boxes: coordinates [y_up, x_left, y_down, x_right]

        """
        category_index={1: {'id': 1, 'name': u'person'},
                        2: {'id': 2, 'name': u'bicycle'},
                        3: {'id': 3, 'name': u'car'},
                        4: {'id': 4, 'name': u'motorcycle'},
                        5: {'id': 5, 'name': u'airplane'},
                        6: {'id': 6, 'name': u'bus'},
                        7: {'id': 7, 'name': u'train'},
                        8: {'id': 8, 'name': u'truck'},
                        9: {'id': 9, 'name': u'boat'},
                        10: {'id': 10, 'name': u'traffic light'},
                        11: {'id': 11, 'name': u'fire hydrant'},
                        13: {'id': 13, 'name': u'stop sign'},
                        14: {'id': 14, 'name': u'parking meter'}}  
        
        with self.detection_graph.as_default():
              image_expanded = np.expand_dims(image, axis=0)
              (boxes, scores, classes, num_detections) = self.sess.run(
                  [self.boxes, self.scores, self.classes, self.num_detections],
                  feed_dict={self.image_tensor: image_expanded})
          
              if visual == True:
                  visualization_utils.visualize_boxes_and_labels_on_image_array(
                      image,
                      np.squeeze(boxes),
                      np.squeeze(classes).astype(np.int32),
                      np.squeeze(scores),
                      category_index,
                      use_normalized_coordinates=True,min_score_thresh=.4,
                      line_thickness=3)
    
                  plt.figure(figsize=(9,6))
                  plt.imshow(image)
                  plt.show()  
              
              boxes=np.squeeze(boxes)
              classes =np.squeeze(classes)
              scores = np.squeeze(scores)
    
              cls = classes.tolist()
              
              # The ID for person is 1
              idx_vec = [i for i, v in enumerate(cls) if ((v==1) and (scores[i]>0.5))]
              filtered_scores = [i for i, v in enumerate(scores.tolist()) if i in idx_vec]
              if len(idx_vec) ==0:
                  print('no detection!')
              else:
                  tmp_car_boxes=[]
                  for idx in idx_vec:
                      dim = image.shape[0:2]
                      box = self.box_normal_to_pixel(boxes[idx], dim)
                      box_h = box[2] - box[0] # that means box[2] is y1, box[0] is y2
                      box_w = box[3] - box[1] # that means box[3] is x2, box[1] is x1
                      ratio = box_h/(box_w + 0.01)
                      
                      #if ((ratio < 0.8) and (box_h>20) and (box_w>20)):
                      tmp_car_boxes.append(box)
                      logging.info('{} , confidence: {} ratio: {}'.format(box, scores[idx], ratio))
                      '''   
                      else:
                          print('wrong ratio or wrong size, ', box, ', confidence: ', scores[idx], 'ratio:', ratio)
                      '''    
                  # non-maximum suppression - cluster overlapping boxes first, then in each cluster, pick the box with highest confidence
                  # to cluster overlapping boxes, loop through the list of boxes and create cluster if the pair is not found together yet. 
                  box_clusters = []
                  indices_to_remove = []
                  for i in range(len(tmp_car_boxes)):
                      for j in range(len(tmp_car_boxes)):
                          if (i == j):
                              continue
                          else:
                              tmp_box_one = tmp_car_boxes[i]
                              tmp_box_two = tmp_car_boxes[j]
                            #   tmp_box_one = get_iou_adapter(tmp_car_boxes[i])
                            #   tmp_box_two = get_iou_adapter(tmp_car_boxes[j])
                              iou = get_iou(tmp_box_one, tmp_box_two)
                            #   print(iou)
                              if (iou > NON_MAXIMUM_SUPPRESSION_THRESHOLD):
                                  tmp_box_one_confidence = filtered_scores[i]
                                  tmp_box_two_confidence = filtered_scores[j]
                                  if (tmp_box_one_confidence > tmp_box_two_confidence):
                                      indices_to_remove.append(j)
                                  else:
                                      indices_to_remove.append(i)
                  indices_to_remove = set(indices_to_remove) # remove duplicates
                  filtered_boxes = [tmp_car_boxes[i] for i in range(len(tmp_car_boxes)) if i not in indices_to_remove]
                                #   both_found_in_box_clusters = False
                                #   one_found_in_box_cluster = False
                                #   for cluster in box_clusters:
                                #       if tmp_box_one in cluster and tmp_box_two in cluster:
                                #           both_found_in_box_clusters = True
                                #           break
                                #       elif tmp_box_one in cluster or tmp_box_two in cluster:
                                #           one_found_in_box_cluster = True
                                #   if (not both_found_in_box_clusters):
                                #       box_clusters.append(tmp_box_one, tmp_box_two)

                            #       for  
                  if (len(filtered_boxes) != len(tmp_car_boxes)):
                      logging.info('Non-maximum suppression removed {} detections'.format(len(tmp_car_boxes) - len(filtered_boxes)))
                  
                  self.car_boxes = filtered_boxes
             
        return self.car_boxes

# converts a box to the pattern that get_iou expects
def get_iou_adapter(box):
    return [box[1], box[2], box[3], box[0]]
    
def get_iou(a, b, epsilon=1e-5):
    """ Given two boxes `a` and `b` defined as a list of four numbers:
            [x1,y1,x2,y2]
        where:
            x1,y1 represent the upper left corner
            x2,y2 represent the lower right corner
        It returns the Intersect of Union score for these two boxes.

    Args:
        a:          (list of 4 numbers) [x1,y1,x2,y2]
        b:          (list of 4 numbers) [x1,y1,x2,y2]
        epsilon:    (float) Small value to prevent division by zero

    Returns:
        (float) The Intersect of Union score.
    """
    # COORDINATES OF THE INTERSECTION BOX
    x1 = max(a[0], b[0])
    y1 = max(a[1], b[1])
    x2 = min(a[2], b[2])
    y2 = min(a[3], b[3])

    # AREA OF OVERLAP - Area where the boxes intersect
    width = (x2 - x1)
    height = (y2 - y1)
    # handle case where there is NO overlap
    if (width<0) or (height <0):
        return 0.0
    area_overlap = width * height

    # COMBINED AREA
    area_a = (a[2] - a[0]) * (a[3] - a[1])
    area_b = (b[2] - b[0]) * (b[3] - b[1])
    area_combined = area_a + area_b - area_overlap

    # RATIO OF AREA OF OVERLAP OVER COMBINED AREA
    iou = area_overlap / (area_combined+epsilon)
    return iou


if __name__ == '__main__':
        matplotlib.use("TkAgg")
        det = PersonDetector()
        os.chdir(cwd)
        TEST_IMAGE_PATHS= glob(os.path.join('test_images/', '*.jpg'))
        # print(len(TEST_IMAGE_PATHS))
        for i, image_path in enumerate(TEST_IMAGE_PATHS):
            print('')
            print('*************************************************')
            
            img_full = Image.open(image_path)
            img_full_np = det.load_image_into_numpy_array(img_full)
            start = time.time()
            b = det.get_localization(img_full_np, visual=False)
            for i in range(len(b)):
                img1 = helpers.draw_box_label(i, img_full_np, b[i], box_color=(255, 0, 0))
                plt.imshow(img1)
            plt.show()

            end = time.time()
            print('Localization time: ', end-start)
#            
            
