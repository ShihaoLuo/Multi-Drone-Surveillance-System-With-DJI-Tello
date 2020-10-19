import pose_estimater
import numpy as np

obj = pose_estimater.PoseEstimater()
obj.loaddata('dataset/')
#obj.showdataset()
wpt = obj.read_from_npy('dataset/post/wpoint.npy')
print(wpt)
wpt = np.array([0.0, 0.0, 0.0, -140.0, 0.0, 0.0, -140.0, -80.0, 0.0, 0.0, -80.0, 0.0])
obj.save_2_npy('dataset/post/wpoint.npy', wpt)