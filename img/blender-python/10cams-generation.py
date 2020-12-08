import bpy
import os
from math import *
from mathutils import *

#set your own target here
target = bpy.data.objects['Cube']
cam = bpy.data.objects['Camera']
t_loc_x = target.location.x
t_loc_y = target.location.y
cam_loc_x = cam.location.x
cam_loc_y = cam.location.y
"""
The default pose is
X: 7m
Y: 1m
Z: 1m
Rotation_Z = 90 degrees (a.k.a cam.rotation_euler[2])
"""

R = (target.location.xy-cam.location.xy).length # Radius

init_angle  = (1-2*bool((cam_loc_y-t_loc_y)<0))*acos((cam_loc_x-t_loc_x)/R)-2*pi*bool((cam_loc_y-t_loc_y)<0) # 8.13 degrees
target_angle = (pi/2 - init_angle) # Go 90-8 deg more
num_steps = 10 #how many rotation steps

for x in range(num_steps):
    alpha = init_angle + (x)*target_angle/num_steps
    cam.rotation_euler[2] = pi/2 + alpha #
    cam.location.x = t_loc_x+cos(alpha)*R
    cam.location.y = t_loc_y+sin(alpha)*R
    """
    # Define SAVEPATH and output filename
    file = os.path.join('renders/', str(x)+'_'+str(round(alpha,2))+'_'+str(round(cam.location.x, 2))+'_'+str(round(cam.location.y, 2)))

    # Render
    bpy.context.scene.render.filepath = file
    bpy.ops.render.render(write_still=True)
    """
    # Randomly place the camera on a circle around the object at the same height as the main camera
    new_camera_pos = Vector((R * cos(alpha), R * sin(alpha), cam.location.z))
    bpy.ops.object.camera_add(enter_editmode=False, location=new_camera_pos)
    # Set the new camera as active
    bpy.context.scene.camera = bpy.context.object
