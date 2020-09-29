#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Fri Sep 25 09:33:51 2020

@author: jake
"""

import tello_video
import tello_controller
import time

controller = tello_controller.Tell_Controller()
controller.scan(1)
controller.command("battery_check 20")
controller.command("correct_ip")
video = tello_video.Tello_Video(controller.tello_list)

x = 0
y = 1.7
theta = 250

pic_folder = './test_pic/'

init_command = ['streamon']
move_command = ['right 100']*8

try:
    for i in range(len(controller.sn_list)):
        controller.command(str(i + 1) + "=" + controller.sn_list[i])
    for init_c in init_command:
        controller.command('*>'+init_c)
    time.sleep(10)
    for i in range(10):
        video.take_pic(pic_folder+str(x)+'_'+str(y)+'_'+str(theta)+'_'+str(i)+'.jpg')
        time.sleep(2)
    controller.save_log(controller.manager)
    controller.manager.close()
    video.close()
except KeyboardInterrupt:
    print ('[Quit_ALL]Multi_Tello_Task got exception. \
           Sending land to all drones...\n')
    for ip in controller.manager.tello_ip_list:
        controller.manager.socket.sendto('streamoff'.encode('utf-8'),
                                         (ip, 9001))
        controller.manager.socket.sendto('land'.encode('utf-8'), (ip, 9001))
    controller.save_log(controller.manager)
    controller.manager.close()
    video.close()
