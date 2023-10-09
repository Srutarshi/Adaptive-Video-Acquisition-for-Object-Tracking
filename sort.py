from __future__ import print_function

import os.path
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from skimage import io
from sklearn.utils.linear_assignment_ import linear_assignment
import glob
import time
import argparse
from filterpy.kalman import KalmanFilter

# not useful
def iou(bb_test,bb_gt):
  """
  Computes IOU between two bboxes in the form [x1,y1,x2,y2]
  """
  xx1 = np.maximum(bb_test[0], bb_gt[0])
  yy1 = np.maximum(bb_test[1], bb_gt[1])
  xx2 = np.minimum(bb_test[2], bb_gt[2])
  yy2 = np.minimum(bb_test[3], bb_gt[3])
  w = np.maximum(0., xx2 - xx1)
  h = np.maximum(0., yy2 - yy1)
  wh = w * h
  o = wh / ((bb_test[2]-bb_test[0])*(bb_test[3]-bb_test[1])
    + (bb_gt[2]-bb_gt[0])*(bb_gt[3]-bb_gt[1]) - wh)
  return(o)

# not useful
def convert_bbox_to_z(bbox):
  """
  Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form
    [x,y,s,r] where x,y is the centre of the box and s is the scale/area and r is
    the aspect ratio
  """
  #print (bbox)
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h    #scale is just area
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))

# not useful
def convert_x_to_bbox(x,score=None):
  """
  Takes a bounding box in the centre form [x,y,s,r] and returns it in the form
    [x1,y1,x2,y2] where x1,y1 is the top left and x2,y2 is the bottom right
  """
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))


class KalmanBoxTracker(object):
  """
  This class represents the internel state of individual tracked objects observed as bbox.
  """
  count = 0
  def __init__(self,bbox):
    """
    Initialises a tracker using initial bounding box.
    """
    #define constant velocity model
    dim_x = 7
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])

    self.kf.R[2:,2:] *= 10. #10
    self.kf.P[4:,4:] *= 1000. #1000 #give high uncertainty to the unobservable initial velocities
    self.kf.P *= 10. #10
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01

    x = np.zeros((dim_x,1))
    self.kf.x[:4] = convert_bbox_to_z(bbox) #binned bbs
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
    self.class_hit_streak = 0

  def update(self,bbox):
    """
    Updates the state vector with observed bbox.
    """
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.class_hit_streak += 1
    #print ("\n in KLT update") #same as new detections in __main__
    #print (bbox)      # this is the z fed into the KLT.update()
    ab = convert_bbox_to_z(bbox)
    self.kf.update(ab)

  def predict(self):
    """
    Advances the state vector and returns the predicted bounding box estimate.
    """
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    #print (self.kf.x.shape)
    #print ("\n")
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
      self.class_hit_streak = 0

    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    #print (self.history[-1])
    #print ("\n")
    return self.history[-1]

  def get_state(self):
    """
    Returns the current bounding box estimate.
    """
    return convert_x_to_bbox(self.kf.x)

def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.1):
  """
  Assigns detections to tracked object (both represented as bounding boxes)

  Returns 3 lists of matches, unmatched_detections and unmatched_trackers
  """
  # trackers are passed here from __main__
  # print (trackers)
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)

  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det,trk)
  matched_indices = linear_assignment(-iou_matrix)

  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)

  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)

  #filter out matched with low IOU
  matches = []
  for m in matched_indices:
    if(iou_matrix[m[0],m[1]]<iou_threshold):
      unmatched_detections.append(m[0])
      unmatched_trackers.append(m[1])
    else:
      matches.append(m.reshape(1,2))
  if(len(matches)==0):
    matches = np.empty((0,2),dtype=int)
  else:
    matches = np.concatenate(matches,axis=0)

  return matches, np.array(unmatched_detections), np.array(unmatched_trackers)


# max_age = 15 (last change done to max_age = 20)
class Sort(object):
  def __init__(self,max_age=15,min_hits=3,class_hits=3):
    """
    Sets key parameters for SORT
    """
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.trackers_class = []
    self.trackers_class_hits = class_hits
    self.trk_prob = []
  
  def predict(self):
    trks = np.zeros((len(self.trackers),5))
    trks_cls = np.zeros(len(self.trackers_class))
    to_del = []
    #rett = []
  
    for t,trk in enumerate(trks):
        pos = self.trackers[t].predict()[0]
        #rett.append(pos)
        trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
        if(np.any(np.isnan(pos))):
            to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    # rett = np.asarray(rett)
    # trks and rett are the same (just return either of them)
    
    return trks, to_del
    

  def update(self, dets, dets_class, new_probs, trks, to_del):
    """
    Params:
      dets1 - a numpy array of detections in the format [[x,y,w,h,score],[x,y,w,h,score],...]
    Requires: this method must be called once for each frame even with empty detections.
    Returns the a similar array, where the last column is the object ID.

    NOTE: The number of objects returned may differ from the number of detections provided.
    """

    self.frame_count += 1
    # get predicted locations from existing trackers.
    ret = []
    
    for t in reversed(to_del):
        self.trackers.pop(t)
        self.trackers_class.pop(t)
        self.trk_prob.pop(t)
    
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
    # passed from associate_detections_to_trackers()
    print('matched update', matched)
    print('unmatched dets', unmatched_dets)
    print('unmatched trks', unmatched_trks)
    # print('self frame count', self.frame_count)
    

    # update matched trackers with assigned detections
    for t,trk in enumerate(self.trackers):
        #print('t',t) this is just the serial # of object
        #print('trk',trk) this is the KalmanBoxTracker object according to the serial # of object
        if(t not in unmatched_trks):
            d = matched[np.where(matched[:,1]==t)[0],0]
            trk.update(dets[d,:][0]) #Actual bbs
            #print('d',d)
            #print('dets[d,:][0]',dets[d,:][0])
            print('t, d[0]',t, d[0])
            #print('new_probs',new_probs)
            #print('nn_probs', new_probs[d[0]])

            if (self.trackers_class[t]==dets_class[d[0]]):
                self.trackers_class[t] = dets_class[d[0]]
                self.trk_prob[t] = (float(self.trk_prob[t]) + float(new_probs[d[0]]))/2.0
            else:
                self.trk_prob[t] = new_probs[d[0]]
                if (trk.class_hit_streak > self.trackers_class_hits): 
                    self.trackers_class[t] = self.trackers_class[t]  #this is simply using the detection class (for first 3 frames)
                    

    # I would really like to use some kind of weighted class of both the classes 

    #create and initialise new trackers for unmatched detections
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:])
        self.trackers.append(trk)
        self.trackers_class.append(dets_class[i])
        self.trk_prob.append(new_probs[i])
   
    i = len(self.trackers)
    tt_cl = self.trackers_class
    tt_prob = self.trk_prob
    #self.trackers_class.append(dets_class)
    #print('trk_cls_ret', self.trackers_class)

    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        i -= 1
        #print('d',d)
        #print('tt_cl',tt_cl[i])

        #the 'or' condition in the if statement in the next line is useless. self.frame_count is always increasing from 0
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
            ret.append(np.concatenate((d,[trk.id+1],[tt_cl[i]],[tt_prob[i]])).reshape(1,-1)) # +1 as MOT benchmark requires positive
       
        #remove dead tracklet
        if(trk.time_since_update > self.max_age):
            self.trackers.pop(i)
            self.trackers_class.pop(i)
            self.trk_prob.pop(i)

    print('ret',ret)
    if(len(ret)>0):
        rrtmp = np.concatenate(ret)
        return rrtmp
    return np.empty((0,5)) 

if __name__ == '__main__':
    trackers = mot_tracker.update(dets, dets_class, new_probs, trks, to_del)
    trackers_p = mot_tracker.predict()
