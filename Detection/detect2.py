from __future__ import print_function
#==================================================================================================================
#Importing Necessarry or Required APIS or Packages:-
#==================================================================================================================
#For Some Array Operation:-
import numpy as np
#For Some Graphical Purpose:-
import matplotlib.pyplot as plt
import matplotlib.patches as patches
#For Composing the Pictures ofr Images in Different Configuration:-
from skimage import io
#To define techniques to match specified patterns according to rules related to Linux:- 
import glob
#For Conversion of Nd Array's (Tensors) to Maatrices:-
from sklearn.utils.linear_assignment_ import linear_assignment
#We have to Work with Video For that time Functions and Qualities are Required:-
import time
#This is Suitable for Current Operating System of  Any Computer that's Why For Some File Manipulation it is used:-
import os.path
#To make sense of how to parse Arguements out of sys.argv:- 
import argparse
#It is a just-in-time compiler for Python that works best on
#code that uses NumPy arrays and functions, and loops:-
from numba import jit
#To Create Some Constant Velocity Model For Common Vehicles:-
from filterpy.kalman import KalmanFilter
#================================================================================================
#To Computes IUO between two bboxes in the form [x1,y1,x2,y2]:-
#================================================================================================
def iou(bb_test,bb_gt):
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
#==============================================================================================
#To Takes a bounding box in the form [x1,y1,x2,y2] and returns z in the form [x,y,s,r] where
#x,y is the centre of the box and s is the scale/area and r is the Aspect Ratio:-
#==============================================================================================
def convert_bbox_to_z(bbox):
  w = bbox[2]-bbox[0]
  h = bbox[3]-bbox[1]
  x = bbox[0]+w/2.
  y = bbox[1]+h/2.
  s = w*h #Here,This Scale is just Area:-
  r = w/float(h)
  return np.array([x,y,s,r]).reshape((4,1))
#==============================================================================================
#To Takes a bounding box in the centre form [x,y,s,r] and returns it in the form [x1,y1,x2,y2]
#Where x1,y1 is the top left and x2,y2 is the bottom right:-
#==============================================================================================
def convert_x_to_bbox(x,score=None):
  w = np.sqrt(x[2]*x[3])
  h = x[2]/w
  if(score==None):
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.]).reshape((1,4))
  else:
    return np.array([x[0]-w/2.,x[1]-h/2.,x[0]+w/2.,x[1]+h/2.,score]).reshape((1,5))
#==============================================================================================
#Here,This class represents the internel state of individual tracked objects observed as bbox:-
#==============================================================================================
class KalmanBoxTracker(object):
  count = 0
#==============================================================================================
#Initialises a tracker using initial bounding box:-
#==============================================================================================
  def __init__(self,bbox):
    #To Define Constant Velocity Model:-
    self.kf = KalmanFilter(dim_x=7, dim_z=4)
    self.kf.F = np.array([[1,0,0,0,1,0,0],[0,1,0,0,0,1,0],[0,0,1,0,0,0,1],[0,0,0,1,0,0,0],  [0,0,0,0,1,0,0],[0,0,0,0,0,1,0],[0,0,0,0,0,0,1]])
    self.kf.H = np.array([[1,0,0,0,0,0,0],[0,1,0,0,0,0,0],[0,0,1,0,0,0,0],[0,0,0,1,0,0,0]])
    self.kf.R[2:,2:] *= 10.
    self.kf.P[4:,4:] *= 1000. #To Give High Uncertainty to the Unobservable Initial Velocities:-
    self.kf.P *= 10.
    self.kf.Q[-1,-1] *= 0.01
    self.kf.Q[4:,4:] *= 0.01
    self.kf.x[:4] = convert_bbox_to_z(bbox)
    self.time_since_update = 0
    self.id = KalmanBoxTracker.count
    KalmanBoxTracker.count += 1
    self.history = []
    self.hits = 0
    self.hit_streak = 0
    self.age = 0
#================================================================================================
#To Updates the state vector with observed bbox.:-
#================================================================================================    
  def update(self,bbox):
    self.time_since_update = 0
    self.history = []
    self.hits += 1
    self.hit_streak += 1
    self.kf.update(convert_bbox_to_z(bbox))
#================================================================================================
#To Advances the state vector and returns the predicted bounding box estimate.:-
#================================================================================================
  def predict(self):
    if((self.kf.x[6]+self.kf.x[2])<=0):
      self.kf.x[6] *= 0.0
    self.kf.predict()
    self.age += 1
    if(self.time_since_update>0):
      self.hit_streak = 0
    self.time_since_update += 1
    self.history.append(convert_x_to_bbox(self.kf.x))
    return self.history[-1]
#================================================================================================
#To Returns the Current Bounding box Estimate:-
#================================================================================================
  def get_state(self):
    return convert_x_to_bbox(self.kf.x)
#================================================================================================
#For Assigns detections to tracked object(both represented as bounding boxes)and Returns 3 lists
#of matches, unmatched_detections and unmatched_trackers
#================================================================================================
def associate_detections_to_trackers(detections,trackers,iou_threshold = 0.3):
  if(len(trackers)==0):
    return np.empty((0,2),dtype=int), np.arange(len(detections)), np.empty((0,5),dtype=int)
  iou_matrix = np.zeros((len(detections),len(trackers)),dtype=np.float32)
  for d,det in enumerate(detections):
    for t,trk in enumerate(trackers):
      iou_matrix[d,t] = iou(det,trk)
  matched_indices = linear_assignment(-iou_matrix)
  #We can do this also:-matched_indices = np.array(matched_indices).reshape((-1,2))
  #We can do this also:-print(iou_matrix.shape,matched_indices.shape)
  unmatched_detections = []
  for d,det in enumerate(detections):
    if(d not in matched_indices[:,0]):
      unmatched_detections.append(d)
  unmatched_trackers = []
  for t,trk in enumerate(trackers):
    if(t not in matched_indices[:,1]):
      unmatched_trackers.append(t)
  #For Filter out matched with low IOU:-
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
#Some Sets Key Parameters For SORT:- 
class Sort(object):
  def __init__(self,max_age=10,min_hits=3):
    self.max_age = max_age
    self.min_hits = min_hits
    self.trackers = []
    self.frame_count = 0
    self.counts = 0
#========================================================================================================
#dets:-A Numpy array of detections in the format [[x1,y1,x2,y2,score],[x1,y1,x2,y2,score],...]
#Requires:-Here,This method must be called once for each frame even with empty detections.
#It Returns the a similar array, where the last column is the object ID.
#NOTE:-Here,The Number of objects returned may differ from the number of detections provided.
#========================================================================================================
  def update(self,dets):
    self.frame_count += 1
    #To Get Predicted Locations From Existing Trackers:-
    trks = np.zeros((len(self.trackers),5))
    to_del = []
    ret = []
    for t,trk in enumerate(trks):
      pos = self.trackers[t].predict()[0]
      trk[:] = [pos[0], pos[1], pos[2], pos[3], 0]
      if(np.any(np.isnan(pos))):
        to_del.append(t)
    trks = np.ma.compress_rows(np.ma.masked_invalid(trks))
    for t in reversed(to_del):
      self.trackers.pop(t)
    matched, unmatched_dets, unmatched_trks = associate_detections_to_trackers(dets,trks)
    #To Update matched trackers with assigned detections:-
    for t,trk in enumerate(self.trackers):
      if(t not in unmatched_trks):
        d = matched[np.where(matched[:,1]==t)[0],0]
        trk.update(dets[d,:][0])
    #To Create and Initialise new trackers for unmatched detections:-
    for i in unmatched_dets:
        trk = KalmanBoxTracker(dets[i,:]) 
        self.trackers.append(trk)
    i = len(self.trackers)
    for trk in reversed(self.trackers):
        d = trk.get_state()[0]
        if((trk.time_since_update < 1) and (trk.hit_streak >= self.min_hits or self.frame_count <= self.min_hits)):
          ret.append(np.concatenate((d,[trk.id+1])).reshape(1,-1)) # +1 as MOT benchmark requires positive
        i -= 1
        #Here,This is For Remove dead tracklet:-
        if(trk.time_since_update > self.max_age):
          self.trackers.pop(i)
    self.counts = KalmanBoxTracker.count
    if(len(ret)>0):
      return np.concatenate(ret)
    return np.empty((0,5))
#==================================================================================================
#For Parse Input Arguements:-
#==================================================================================================
def parse_args():
    parser = argparse.ArgumentParser(description='SORT demo')
    parser.add_argument('--display', dest='display', help='Display online tracker output (slow) [False]',action='store_true')
    args = parser.parse_args()
    return args
if __name__ == '__main__':
  #For All Train:-
  args = parse_args()
  display = args.display
  phase = 'train'
  total_time = 0.0
  total_frames = 0
  colours = np.random.rand(32,3) #Here, It is used only for display:-
  if(display):
    if not os.path.exists('mot_benchmark'):
      print('\n\tERROR: mot_benchmark link not found!\n\n    Create a symbolic link to the MOT benchmark\n    (https://motchallenge.net/data/2D_MOT_2015/#download). E.g.:\n\n    $ ln -s /path/to/MOT2015_challenge/2DMOT2015 mot_benchmark\n\n')
      exit()
    plt.ion()
    fig = plt.figure() 
  if not os.path.exists('output'):
    os.makedirs('output')
  for seq in sequences:
    mot_tracker = Sort() #To Create instance of the SORT tracker:-
    seq_dets = np.loadtxt('data/%s/det.txt'%(seq),delimiter=',') #load detections
    with open('output/%s.txt'%(seq),'w') as out_file:
      print("Processing %s."%(seq))
      for frame in range(int(seq_dets[:,0].max())):
        frame += 1 #For detection and frame numbers begin at 1:-
        dets = seq_dets[seq_dets[:,0]==frame,2:7]
        dets[:,2:4] += dets[:,0:2] #To Convert to [x1,y1,w,h] to [x1,y1,x2,y2]:-
        total_frames += 1
        if(display):
          ax1 = fig.add_subplot(111, aspect='equal')
          fn = 'mot_benchmark/%s/%s/img1/%06d.jpg'%(phase,seq,frame)
          im =io.imread(fn)
          ax1.imshow(im)
          plt.title(seq+' Tracked Targets')
        start_time = time.time()
        trackers = mot_tracker.update(dets)
        cycle_time = time.time() - start_time
        total_time += cycle_time
        for d in trackers:
          print('%d,%d,%.2f,%.2f,%.2f,%.2f,1,-1,-1,-1'%(frame,d[4],d[0],d[1],d[2]-d[0],d[3]-d[1]),file=out_file)
          if(display):
            d = d.astype(np.int32)
            ax1.add_patch(patches.Rectangle((d[0],d[1]),d[2]-d[0],d[3]-d[1],fill=False,lw=3,ec=colours[d[4]%32,:]))
            ax1.set_adjustable('box-forced')
        if(display):
          fig.canvas.flush_events()
          plt.draw()
          ax1.cla()
  print("Total Tracking took: %.3f for %d frames or %.1f FPS"%(total_time,total_frames,total_frames/total_time))
  if(display):
    print("Note: to get real runtime results run without the option: --display")
#===================================================================================================================
