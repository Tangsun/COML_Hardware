#!/usr/bin/env python
"""
vimfly - vim keybindings for your multirotor!

Teleoperated flying from your keyboard. Command u, v, w, and psidot
Simulates joystick commands as a sensor_msgs/Joy message.

The following keybindings are used:
    - a: CCW (-psidot)
    - s: CW (+psidot)
    - d: Down (-w)
    - f: Up (+w)
    - h: Left (+v)
    - j: Backward (-u)
    - k: Forward (+u)
    - l: Right (-v)


https://xkcd.com/1823/

Parker Lusk
22 Oct 2017
"""
import rospy

import numpy as np
import pygame

from sensor_msgs.msg import Joy

class VimFly:
    def __init__(self):

        # initialize pygame display
        pygame.init()
        pygame.display.set_caption('vimfly')
        self.screen = pygame.display.set_mode((550, 200))
        self.font = pygame.font.SysFont("monospace", 18)

        # Joystick commands
        self.pub_joy = rospy.Publisher('joy', Joy, queue_size=1)

        self.joy_kx = rospy.get_param('~kx', 2.0)
        self.joy_ky = rospy.get_param('~ky', 2.0)
        self.joy_kz = rospy.get_param('~kz', 0.5)
        self.joy_kr = rospy.get_param('~kr', 2.0)

        # retrieve vimfly parameters from the rosparam server
        self.params = {
            'vx_cmd': rospy.get_param('~vx_cmd', 0.5),
            'vy_cmd': rospy.get_param('~vy_cmd', 0.5),
            'vz_cmd': rospy.get_param('~vz_cmd', 0.5),
            'psidot_cmd': rospy.get_param('~psidot_cmd', np.pi/2.0),
        }

        # Send commands 'til I die
        self.run()


    def run(self):
        rate = rospy.Rate(30)
        while not rospy.is_shutdown():

            keys = pygame.key.get_pressed()

            # LEFT -- h
            if keys[pygame.K_h]:
                vy = self.params['vy_cmd']

            # RIGHT -- l
            elif keys[pygame.K_l]:
                vy = -self.params['vy_cmd']

            else:
                vy = 0


            # FORWARD -- k
            if keys[pygame.K_k]:
                vx = self.params['vx_cmd']

            # BACKWARD -- j
            elif keys[pygame.K_j]:
                vx = -self.params['vx_cmd']

            else:
                vx = 0


            # DOWN -- d
            if keys[pygame.K_d]:
                vz = -self.params['vz_cmd']

            # UP -- f
            elif keys[pygame.K_f]:
                vz = self.params['vz_cmd']

            else:
                vz = 0


            # CCW -- a
            if keys[pygame.K_a]:
                psidot = self.params['psidot_cmd']

            # CW -- s
            elif keys[pygame.K_s]:
                psidot = -self.params['psidot_cmd']

            else:
                psidot = 0


            A = 1 if keys[pygame.K_1] else 0 # takeoff
            X = 1 if keys[pygame.K_2] else 0 # land
            B = 1 if keys[pygame.K_3] else 0 # kill
            Y = 1 if keys[pygame.K_4] else 0 # left button
            LB = 1 if keys[pygame.K_5] else 0 # left button
            RB = 1 if keys[pygame.K_6] else 0 # right button


            # Pack up message and ship it. Note: the order is hardcoded
            # to match what is being expected in joy.py
            msg = Joy()
            msg.header.stamp = rospy.Time.now()
            msg.axes = [psidot / self.joy_kr, vz / self.joy_kz, 0.0,
                            vy / self.joy_ky, vx / self.joy_kx]
            msg.buttons = [A, B, X, Y, LB, RB]
            self.pub_joy.publish(msg)

            # Update the display with the current commands
            self.update_display(vx, vy, vz, psidot)

            # process event queue and throttle the while loop
            pygame.event.pump()
            rate.sleep()


    def update_display(self, vx, vy, vz, psidot):
        self.display_help()

        msgText = "vx: {:.2f}, vy: {:.2f}, vz: {:.2f}, psidot: {:.2f}".format(vx, vy, vz, psidot)
        self.render(msgText, (0,140))
        
        pygame.display.flip()


    def display_help(self):
        self.screen.fill((0,0,0))

        LINE=20

        self.render("vimfly keybindings:", (0,0))
        self.render("- a: CCW (+psidot)", (0,1*LINE)); self.render("- h: Left (+vy)", (250,1*LINE))
        self.render("- s: CW (-psidot)", (0,2*LINE)); self.render("- j: Backward (-vx)", (250,2*LINE))
        self.render("- d: Down (-vz)", (0,3*LINE)); self.render("- k: Forward (+vx)", (250,3*LINE))
        self.render("- f: Up (+vz)", (0,4*LINE)); self.render("- l: Right (-vy)", (250,4*LINE))

        self.render("1: take off", (0,5*LINE))
        self.render("2: land", (150,5*LINE))
        self.render("3: kill", (250,5*LINE))
        self.render("4: Y", (350,5*LINE))
        self.render("5: LB", (415,5*LINE))
        self.render("6: RB", (485,5*LINE))


    def render(self, text, loc):
        txt = self.font.render(text, 1, (255,255,255))
        self.screen.blit(txt, loc)



if __name__ == '__main__':
    rospy.init_node('vimfly', anonymous=False)
    try:
        teleop = VimFly()
    except rospy.ROSInterruptException:
        pass
