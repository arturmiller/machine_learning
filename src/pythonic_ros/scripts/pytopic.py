#!/usr/bin/env python

import rospy


class Pytopic:
    def __init__(self, name):
        rospy.init_node(name, anonymous=True)
        

    def decorator_func(self, func):
        
        def new_func():
            rate = rospy.Rate(self.frequency)
            while not rospy.is_shutdown():
                self.pub.publish(func())
                rate.sleep()

        self.new_func = new_func
        return new_func

    def publish(self, topic_name, frequency, msg_type):
        self.pub = rospy.Publisher(topic_name, msg_type, queue_size=10)
        self.frequency = frequency
        return self.decorator_func

    def decorator_listener_func(self, func):
        print('b')
        def new_func():
            rospy.Subscriber(self.listener_topic_name, self.listener_msg_type, func)
            rospy.spin()

        self.new_func = new_func
        return new_func


    def listen(self, topic_name, msg_type):
        self.listener_topic_name = topic_name
        self.listener_msg_type = msg_type
        print('a')
        return self.decorator_listener_func

    def run(self):
        try:
            self.new_func()
        except rospy.ROSInterruptException:
            pass
