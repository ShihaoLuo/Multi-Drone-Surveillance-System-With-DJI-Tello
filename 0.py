# -*- coding: utf-8 -*-
# @Time    : 2021/1/1 下午4:50
# @Author  : JakeShihao Luo
# @Email   : jakeshihaoluo@gmail.com
# @File    : 0.py
# @Software: PyCharm

from scanner import *
import socket
from tello_node import *
import multiprocessing


def received_ok(c_res):
    soc_res= socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    soc_res.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    soc_res.bind(('', 8889))
    while True:
        try:
            response, ip = soc_res.recvfrom(1024)
            ip = ''.join(str(ip[0]))
            if response.decode(encoding='utf-8', errors='ignore').upper() == 'OK':
                cmd_res.put(ip)
            time.sleep(0.01)
        except socket.error as exc:
            print("[Exception_Error(rev)]Caught exception socket.error : %s\n" % exc)


Node = {}
cmd_res = multiprocessing.Queue()
scanner = Scanner()
scanner.find_available_tello(2)
Node[scanner.get_tello_info()[0][0]] = TelloNode(scanner.get_tello_info()[0])
Node[scanner.get_tello_info()[1][0]] = TelloNode(scanner.get_tello_info()[1])
rec_thread = multiprocessing.Process(target=received_ok, args=(cmd_res,), daemon=True)
rec_thread.start()
Node[scanner.get_tello_info()[0][0]].run()
Node[scanner.get_tello_info()[1][0]].run()
old_time = time.time()
while True:
    if cmd_res.empty() is False:
        tmp = cmd_res.get()
        Node[tmp].update_res()
        old_time = time.time()
    if time.time() - old_time >= 10:
        print('Main thread die!')
        break


