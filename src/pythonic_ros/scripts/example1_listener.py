#!/usr/bin/env python

import rospy
from std_msgs.msg import String
from pytopic import Pytopic

app = Pytopic('listener')

@app.listen('chatter', String)
def callback(data):
    rospy.loginfo(rospy.get_caller_id() + "I heard %s", data.data)
    
if __name__ == '__main__':
    app.run()