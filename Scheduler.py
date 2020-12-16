import numpy as np
import time
from pose_estimater import pose_estimater
import threading
#import matplotlib.pyplot as plt

class Scheduler:
    def __init__(self, controller, video):
        self.controller = controller
        self.video = video
        self.pose_estimater = pose_estimater.PoseEstimater('SIFT', 15)
        self.pose_estimater.loaddata('pose_estimater/dataset/')
        self.tello_list = controller.tello_list
        self.path = {}
        self.id_sn = controller.id_sn_dict
        self.sn_ip = controller.sn_ip_dict
        self.id_ip = {}
        self.timeflag = {}
        for key in self.id_sn.keys():
            self.id_ip[key] = self.sn_ip[self.id_sn[key]]
        for key in self.id_sn.keys():
            self.timeflag[key] = 0
        self.pose = {}
        self.permission = {}
        for key in self.id_ip.keys():
            self.permission[key] = 0
        self.videoflag = {}
        for key in self.id_ip.keys():
            self.videoflag[key] = 0

    def drone_init(self):
        for i in self.id_ip.keys():
            # self.controller.command(str(i)+'>setfps high')
            # self.controller.command(str(i)+'>setresolution high')
            # self.controller.command(str(i)+'>setbitrate 5')
            self.controller.command(str(i)+'>streamon')
#
    def init_path(self, loopcount, _id, path, pose):
        tmp = np.array(path)
        self.path[self.id_ip[_id]] = np.tile(tmp, (loopcount, 1))
        self.pose[self.id_ip[_id]] = pose

    def updatepos(self, _last_cmd, _last_pose, _Video, _id):
        ip = self.id_ip[_id]
        pose = np.array([0, 0, 0, 0])
        #if _last_pose[1] < 100 and (time.time() - self.timeflag[_id] > 20 or self.timeflag[_id] == 0):
        img = _Video.get_frame(ip)
        #plt.imshow(img)
        if img is not None:
            print('in pose estimater')
            _pose, yaw = self.pose_estimater.estimate_pose(img)
            #print(_pose)
            if _pose is not None:
                print('in img1')
                time.sleep(2.5)
                while True:
                    img = _Video.get_frame(ip)
                    if img is not None:
                        break
                _pose, yaw = self.pose_estimater.estimate_pose(img)
                if _pose is not None:
                    print('in img2')
                    pose[0] = _pose[0]
                    pose[1] = _pose[1]
                    if _pose[2] == 0:
                        pose[2] = _last_pose[2]
                    else:
                        pose[2] = _pose[2]
                    pose[3] = yaw
                    print('update pose of:', _id, pose)
                    self.timeflag[_id] = int(time.time())
                    return pose
        if 'ccw' in _last_cmd:
            angle = float(_last_cmd.partition(' ')[2])
            print('angle:{}'.format(angle))
            pose[3] = angle + _last_pose[3]
            pose[0:3] = _last_pose[0:3]
        elif 'go' in _last_cmd:
            tmp = _last_cmd.split(' ')[1:4]
            tmp = [int(i) for i in tmp]
            alpha = _last_pose[3] * 3.1416 / 180
            M = np.array([[np.cos(alpha), np.sin(alpha), 0],
                          [-np.sin(alpha), np.cos(alpha), 0],
                          [0, 0, 1]])
            tmp = np.dot(np.linalg.inv(M), tmp)
            tmp = np.append(tmp, 0)
            pose = _last_pose + tmp
        else:
            pose = _last_pose
        print('update pose of:' ,_id, pose)
        return pose

    def start(self):
        #print('path:', self.path)
        #print('pose:', self.pose)
        run_thread = {}
        for _id in self.id_ip.keys():
            run_thread[_id] = threading.Thread(target=self.run, args=(_id,))
            run_thread[_id].start()
        while True:
            for key in self.id_ip.keys():
                self.permission[key] = 0
            if len(self.id_ip) == 1:
                self.permission[list(self.id_ip.keys())[0]] = 1
                print('permision:', self.permission)
                print('start living...')
            else:
                id1 = list(self.id_ip.keys())
                id2 = list(self.id_ip.keys())
                #print('id1, id2', id1, id2)
                for _id in id1:
                    id2.remove(_id)
                    if len(id2) == 0:
                        self.permission[_id] = 1
                    else:
                        for _id2 in id2:
                            #print('ip', self.id_ip[_id])
                            d = np.linalg.norm(np.array((self.pose[self.id_ip[_id]])[0:3]) - np.array((self.pose[self.id_ip[_id2]])[0:3]), 2)
                            #print('d:', d)
                            if d > 300:
                                self.permission[_id] = 1

                print('permision:', self.permission)
                print('start living...')
            time.sleep(0.05)


    def run(self, _id):
        _len = len(self.path[self.id_ip[_id]])
        cnt = 0
        print('len ', _len)
        #print('pose,', self.pose)
        if self.pose[self.id_ip[_id]][2] == 0:
            while self.permission[_id] == 0:
                print('run thread, in the while...')
                self.controller.command(str(_id) + 'wait 0.5')
                self.pose[self.id_ip[_id]] = self.updatepos(' ', self.pose[self.id_ip[_id]], self.video, _id)
            #self.controller.command(str(_id) + 'wait 5')
            self.controller.command(str(_id) + '>takeoff')
            self.controller.command(str(_id) + '>up 170')
            self.pose[self.id_ip[_id]][2] = 100
            time.sleep(2.5)
            self.pose[self.id_ip[_id]] = self.updatepos(' ', self.pose[self.id_ip[_id]], self.video,  _id)
        for target in self.path[self.id_ip[_id]]:
            print('in target for loop...', _id)
            if self.videoflag[_id] == 0 and self.pose[self.id_ip[_id]][1] < 100:
                self.controller.command(str(_id) + '>streamon')
                self.videoflag[_id] = 1
            if self.videoflag[_id] == 1 and self.pose[self.id_ip[_id]][1] >= 100:
                self.controller.command(str(_id) + '>streamoff')
                self.videoflag[_id] = 0
            while self.permission[_id] == 0:
                print('run thread, in the while...', _id)
                self.controller.command(str(_id) + 'wait 2')
            print("--------------------------")
            print("target:{}".format(target))
            theta = target[3] - self.pose[self.id_ip[_id]][3]
            if abs(theta) > 30:
                if abs(theta) > 180:
                    cmd = 'cw ' + str(theta+180)
                    self.controller.command(str(_id) + ">" + cmd)
                    cmd = 'ccw ' + str(theta)
                    self.pose[self.id_ip[_id]] = self.updatepos(cmd, self.pose[self.id_ip[_id]], self.video, _id)
                    print("Pose in the drone_world is {}".format(self.pose[self.id_ip[_id]]))
                else:
                    cmd = 'ccw ' + str(theta)
                    self.controller.command(str(_id)+">"+cmd)
                    self.pose[self.id_ip[_id]] = self.updatepos(cmd, self.pose[self.id_ip[_id]], self.video, _id)
                    print("Pose in the drone_world is {}".format(self.pose[self.id_ip[_id]]))
            if np.linalg.norm(target[0:3] - self.pose[self.id_ip[_id]][0:3]) < 50:
                pass
            else:
                alpha = self.pose[self.id_ip[_id]][3] * 3.1416 / 180
                #print("alpha:{}".format(alpha))
                M = np.array([[np.cos(alpha), np.sin(alpha), 0],
                              [-np.sin(alpha), np.cos(alpha), 0],
                              [0, 0, 1]])
                #print("M: {}".format(M))
                tmp = target[0:3]-self.pose[self.id_ip[_id]][0:3]
                tmp = np.dot(M, tmp)
                #tmp = [int(i) for i in tmp]
                tmp = np.append(tmp, 100)
                tmp = [int(i) for i in tmp]
                tmp = [str(i) for i in tmp]
                cmd = 'go ' + ' '.join(tmp)
                self.controller.command(str(_id) + '>' + cmd)
                self.pose[self.id_ip[_id]] = self.updatepos(cmd, self.pose[self.id_ip[_id]], self.video, _id)
                print("Pose in the drone_world is {}".format(self.pose[self.id_ip[_id]]))
            cnt += 1
            print('run living...', _id)
            if cnt == _len:
                del self.id_ip[_id]
                self.controller.command(str(_id) + ">land")
                self.controller.command(str(_id) + ">streamoff")
        print('run dying...', _id)