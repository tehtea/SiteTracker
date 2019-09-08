#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""@author: ambakick
"""

import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import glob
#from moviepy.editor import VideoFileClip
from collections import deque
from sklearn.utils.linear_assignment_ import linear_assignment
# from scipy.optimize import linear_sum_assignment

import helpers
import detector
import tracker
import cv2
import logging

# Global variables to be used by functions of VideoFileClip
frame_count = 0 # frame counter

max_age = 15  # no.of consecutive unmatched detection before 
             # a track is deleted

min_hits =1  # no. of consecutive matches needed to establish a track

tracker_list =[] # list for trackers, shared across multiple frames

# list for track ID
track_id_list= deque([str(i) for i in range(1, 21)])

# debug = True
debug = False

VIDEO_TEST = True # test with a video file

for handler in logging.root.handlers[:]:
    logging.root.removeHandler(handler)
logging.basicConfig(filename='C:\\Users\\tehtea\\Person-Detection-and-Tracking\\demo.log', level=logging.INFO)


def assign_color(id):
    colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (0, 255, 255), (255, 255, 255)]
    return colors[id % len(colors)]

def assign_detections_to_trackers(trackers, detections, iou_thrd = 0.3):
    '''
    From current list of trackers and new detections, output matched detections,
    unmatchted trackers, unmatched detections.
    '''    
    
    IOU_mat= np.zeros((len(trackers),len(detections)),dtype=np.float32)
    # at the end of this loop, each tracker box should be paired with all detection boxes, with the corresponding IOU score stored at that position.
    for t,trk in enumerate(trackers):
        #trk = convert_to_cv2bbox(trk) 
        for d,det in enumerate(detections):
         #   det = convert_to_cv2bbox(det)
            IOU_mat[t,d] = helpers.box_iou2(trk,det) # get IOU score of tracker box and detection box.
    
    # Matching the tracker boxes to each detection now becomes an Assignment Problem. To solve this,
    # use the Hungarian algorithm (also known as Munkres algorithm)
    
    # The IOU_mat creates a very useful real-valued matrix C. The cost function used was the IOU.
    # return matches as a pair. Tracker on left side, Detection on right side.   
    IOU_mat = 1 / (IOU_mat + 1) # offset by 1 to prevent division by zero
    matched_idx = linear_assignment(IOU_mat)

    unmatched_trackers, unmatched_detections = [], []
    for t,trk in enumerate(trackers):
        if(t not in matched_idx[:,0]):
            unmatched_trackers.append(t)

    for d, det in enumerate(detections):
        if(d not in matched_idx[:,1]):
            unmatched_detections.append(d)

    matches = []
   
    # For creating trackers we consider any detection with an 
    # overlap less than iou_thrd to signifiy the existence of 
    # an untracked object
    
    for m in matched_idx:
        if(IOU_mat[m[0],m[1]]<iou_thrd):
            unmatched_trackers.append(m[0])
            unmatched_detections.append(m[1])
        else:
            matches.append(m.reshape(1,2))
    
    if(len(matches)==0):
        matches = np.empty((0,2),dtype=int)
    else:
        matches = np.concatenate(matches,axis=0) # need make sure it is a numpy array

    return matches, np.array(unmatched_detections), np.array(unmatched_trackers)       

def pipeline(img):
    '''
    Pipeline function for detection and tracking
    '''
    global frame_count
    global max_age
    global min_hits
    global track_id_list
    global debug
    global tracker_list

    frame_count+=1
    
    z_box = det.get_localization(img) # get the location of each detected person in current image.
    if debug:
       print('Frame:', frame_count)
       
    x_box =[] # stores a list of boxes by each tracker
    if debug: 
        for i in range(len(z_box)):
           img1= helpers.draw_box_label(i, img, z_box[i], box_color=assign_color(i))
           plt.imshow(img1)
        plt.show()

    if len(tracker_list) > 0:
        for trk in tracker_list:
            x_box.append(trk.box)
    
    matched, unmatched_dets, unmatched_trks \
    = assign_detections_to_trackers(x_box, z_box, iou_thrd = 0.3) # 'matched' is an array of IDs that pairs a matched object in array with a tracker in the array
        
    if debug:
        print('Detection: ', z_box)
        print('x_box: ', x_box)
        print('matched:', matched)
        print('unmatched_det:', unmatched_dets)
        print('unmatched_trks:', unmatched_trks)
    
         
    # Deal with matched detections     
    if matched.size>0:
        for trk_idx, det_idx in matched:
            z = z_box[det_idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk= tracker_list[trk_idx]
            tmp_trk.kalman_filter(z)
            xx = tmp_trk.x_state.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            x_box[trk_idx] = xx
            tmp_trk.box =xx
            tmp_trk.hits += 1
    
    # Deal with unmatched detections      
    if len(unmatched_dets)>0:
        for idx in unmatched_dets:
            z = z_box[idx]
            z = np.expand_dims(z, axis=0).T
            tmp_trk = tracker.Tracker() # Create a new tracker
            x = np.array([[z[0], 0, z[1], 0, z[2], 0, z[3], 0]]).T
            tmp_trk.x_state = x
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box = xx
            tmp_trk.id = track_id_list.popleft() # assign an ID for the tracker
            # print(tmp_trk.id)
            tracker_list.append(tmp_trk)
            x_box.append(xx)
    
    # Deal with unmatched tracks       
    if len(unmatched_trks)>0:
        for trk_idx in unmatched_trks:
            tmp_trk = tracker_list[trk_idx]
            tmp_trk.no_losses += 1
            tmp_trk.predict_only()
            xx = tmp_trk.x_state
            xx = xx.T[0].tolist()
            xx =[xx[0], xx[2], xx[4], xx[6]]
            tmp_trk.box =xx
            x_box[trk_idx] = xx
                   
       
    # The list of tracks to be annotated  
    good_tracker_list =[]
    for trk in tracker_list:
        if ((trk.hits >= min_hits) and (trk.no_losses <=max_age)):
             good_tracker_list.append(trk)
             x_cv2 = trk.box
             if debug:
                 print('updated box: ', x_cv2)
                 print()
             img= helpers.draw_box_label(trk.id,img, x_cv2) # Draw the bounding boxes on the 
                                             # images
    # Book keeping
    deleted_tracks = filter(lambda x: x.no_losses >max_age, tracker_list)  
    
    for trk in deleted_tracks:
        track_id_list.append(trk.id)
    
    tracker_list = [x for x in tracker_list if x.no_losses<=max_age]
    
    if debug:
       print('Ending tracker_list: ',len(tracker_list))
       print('Ending good tracker_list: ',len(good_tracker_list))
    
    cv2.imshow("frame",img)
    return img
    
if __name__ == "__main__":    
    
    det = detector.PersonDetector()
    
    if debug: # test on a sequence of images
        images = [plt.imread(file) for file in glob.glob('./test_images/*.jpg')]
        matplotlib.use("TkAgg")
        for i in range(len(images))[0:7]:
            image = images[i]
            image_box = pipeline(image)   
            plt.figure()
            plt.imshow(image_box)
        plt.show()
    else: # test on a video file.
        try:
            cap = cv2.VideoCapture(0)
        except: # usually an issue with running from RPI
            cap = cv2.VideoCapture(-1)
        if (VIDEO_TEST):
            cap = cv2.VideoCapture(r'./test_data/construction_footage.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output.avi',fourcc, 8.0, (640,480))

        while(True):
            logging.info('startingFrame')
            ret, img = cap.read()
            if (image.size < 0 or image.size < 0):
                break
            #print(img)
            
            np.asarray(img)
            new_img = pipeline(img)
            out.write(new_img)
            logging.info('endingFrame')
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
            
        cap.release()
        cv2.destroyAllWindows()
        # print(round(end-start, 2), 'Seconds to finish')
