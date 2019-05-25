#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from pytopic import Pytopic

app = Pytopic('talker')

@app.publish('chatter', 10, String) # 10hz
def talker():
    hello_str = "hello world %s" % rospy.get_time()
    rospy.loginfo(hello_str)
    return hello_str

if __name__ == '__main__':
    app.run()
