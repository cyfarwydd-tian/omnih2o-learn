import torch 
import numpy as np
import phc.utils.rotation_conversions as tRot
import xml.etree.ElementTree as ETree
from easydict import EasyDict
import scipy.ndimage.filters as filters
import smpl_sim.poselib.core.rotation3d as pRot


G1_ROTATION_AXIS = torch.tensor([[
    [0, 0, 1], # l_hip_yaw
    [1, 0, 0], # l_hip_roll
    [0, 1, 0], # l_hip_pitch
    
    [0, 1, 0], # kneel
    [0, 1, 0], # ankle
    
    [0, 0, 1], # r_hip_yaw
    [1, 0, 0], # r_hip_roll
    [0, 1, 0], # r_hip_pitch
    
    [0, 1, 0], # kneel
    [0, 1, 0], # ankle
    
    [0, 0, 1], # torso
    
    [0, 1, 0], # l_shoulder_pitch
    [1, 0, 0], # l_roll_pitch
    [0, 0, 1], # l_yaw_pitch
    
    [0, 1, 0], # l_elbow
    
    [0, 1, 0], # r_shoulder_pitch
    [1, 0, 0], # r_roll_pitch
    [0, 0, 1], # r_yaw_pitch
    
    [0, 1, 0], # r_elbow
]])

class Humanoid_Batch:
     def __init__(self, mjcf_file = f"resources/robots/g1/g1.xml", extend_hand = True, extend_head = False, device = torch.device("cpu")):
        self.mjcf_data = mjcf_data = self.from_mjcf(mjcf_file)
        self.extend_hand = extend_hand
        self.extend_head = extend_head
        if extend_hand:
            self.model_names = mjcf_data['node_names'] + ["left_hand_link", "right_hand_link"]
            self._parents = torch.cat((mjcf_data['parent_indices'], torch.tensor([15, 19]))).to(device) # Adding the hands joints
            arm_length = 0.3
            self._offsets = torch.cat((mjcf_data['local_translation'], torch.tensor([[arm_length, 0, 0], [arm_length, 0, 0]])), dim = 0)[None, ].to(device)
            self._local_rotation = torch.cat((mjcf_data['local_rotation'], torch.tensor([[1, 0, 0, 0], [1, 0, 0, 0]])), dim = 0)[None, ].to(device)
            self._remove_idx = 2
        else:
            self._parents = mjcf_data['parent_indices']
            self.model_names = mjcf_data['node_names']
            self._offsets = mjcf_data['local_translation'][None, ].to(device)
            self._local_rotation = mjcf_data['local_rotation'][None, ].to(device)
            
        if extend_head:
            self._remove_idx = 3
            self.model_names = self.model_names + ["head_link"]
            self._parents = torch.cat((self._parents, torch.tensor([0]).to(device))).to(device) # Adding the hands joints
            head_length = 0.75
            self._offsets = torch.cat((self._offsets, torch.tensor([[[0, 0, head_length]]]).to(device)), dim = 1).to(device)
            self._local_rotation = torch.cat((self._local_rotation, torch.tensor([[[1, 0, 0, 0]]]).to(device)), dim = 1).to(device)
            
        
        self.joints_range = mjcf_data['joints_range'].to(device)
        self._local_rotation_mat = tRot.quaternion_to_matrix(self._local_rotation).float() # w, x, y ,z