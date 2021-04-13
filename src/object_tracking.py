#!/usr/bin/env python
from __future__ import division, print_function, absolute_import

import cv2
import numpy as np

# from deep_sort.application_util import preprocessing
# from deep_sort.application_util import visualization
from deep_sort import nn_matching
from deep_sort.detection import Detection
from deep_sort.tracker import Tracker
from object_detector_msgs.msg import ObjectArray, Object
import rospy


class ObjectTracking(object):

    def __init__(self, name):
        self.max_iou_distance = rospy.get_param('max_iou_distance', 0.7)
        self.max_age = rospy.get_param('max_age', 30)
        self.n_init = rospy.get_param('n_init', 3)
        self.metric = nn_matching.NearestNeighborDistanceMetric("cosine", 0.2, None)
        self.tracker = Tracker(self.metric, self.max_iou_distance, self.max_age, self.n_init)
        rospy.Subscriber("image_object", ObjectArray, self.__callback)
        self.pub = rospy.Publisher('image_object_tracked', ObjectArray, queue_size=10)

    def __callback(self, data):

        def create_detections(objects):
            detection_list = []
            for obj in len(objects):
                bbox = np.array([obj.x_offset, obj.y_offset, obj.width, obj.height])
                confidence = obj.score
                detection_list.append(Detection(bbox, confidence, []))
            return detection_list

        print("Received detection results")

        # Load image and generate detections.
        detections = create_detections(data.objects)
        
        # Update tracker.
        self.tracker.predict()
        new_ids = self.tracker.update(detections)

        for i in range(data.size):
            data.objects[i].id = new_ids[i]
        
        self.pub.publish(data)
    
def object_tracking():
    rospy.init_node('object_tracking', anonymous=True)
    tracking = ObjectTracking(rospy.get_name())
    rospy.spin()

if __name__ == '__main__':
    try:
        object_tracking()
    except rospy.ROSInterruptException:
        pass