import sys
import os
import numpy as np
import scipy.interpolate as interpolate
import scipy.ndimage.filters as filters

from skeletondef import Choose as skd
from skeletondef import Games
from skeletondef import Original

sys.path.append('./motion')

import BVH as BVH
import Animation as Animation
from Quaternions import Quaternions
from Pivots import Pivots
from Learning import RBF

""" Options """

rng = np.random.RandomState(1234)
to_meters = skd.JOINT_SCALE
window = skd.WINDOW
njoints = skd.JOINT_NUM
prev_rotations = None

""" Data """

data_terrain = [
    './data/animations/LocomotionFlat01_000.bvh',
    './data/animations/LocomotionFlat02_000.bvh',
    './data/animations/LocomotionFlat02_001.bvh',
    './data/animations/LocomotionFlat03_000.bvh',
    './data/animations/LocomotionFlat04_000.bvh',
    './data/animations/LocomotionFlat05_000.bvh',
    './data/animations/LocomotionFlat06_000.bvh',
    './data/animations/LocomotionFlat06_001.bvh',
    './data/animations/LocomotionFlat07_000.bvh',
    './data/animations/LocomotionFlat08_000.bvh',
    './data/animations/LocomotionFlat08_001.bvh',
    './data/animations/LocomotionFlat09_000.bvh',
    './data/animations/LocomotionFlat10_000.bvh',
    './data/animations/LocomotionFlat11_000.bvh',
    './data/animations/LocomotionFlat12_000.bvh',

    './data/animations/LocomotionFlat01_000_mirror.bvh',
    './data/animations/LocomotionFlat02_000_mirror.bvh',
    './data/animations/LocomotionFlat02_001_mirror.bvh',
    './data/animations/LocomotionFlat03_000_mirror.bvh',
    './data/animations/LocomotionFlat04_000_mirror.bvh',
    './data/animations/LocomotionFlat05_000_mirror.bvh',
    './data/animations/LocomotionFlat06_000_mirror.bvh',
    './data/animations/LocomotionFlat06_001_mirror.bvh',
    './data/animations/LocomotionFlat07_000_mirror.bvh',
    './data/animations/LocomotionFlat08_000_mirror.bvh',
    './data/animations/LocomotionFlat08_001_mirror.bvh',
    './data/animations/LocomotionFlat09_000_mirror.bvh',
    './data/animations/LocomotionFlat10_000_mirror.bvh',
    './data/animations/LocomotionFlat11_000_mirror.bvh',
    './data/animations/LocomotionFlat12_000_mirror.bvh',

    './data/animations/WalkingUpSteps01_000.bvh',
    './data/animations/WalkingUpSteps02_000.bvh',
    './data/animations/WalkingUpSteps03_000.bvh',
    './data/animations/WalkingUpSteps04_000.bvh',
    './data/animations/WalkingUpSteps04_001.bvh',
    './data/animations/WalkingUpSteps05_000.bvh',
    './data/animations/WalkingUpSteps06_000.bvh',
    './data/animations/WalkingUpSteps07_000.bvh',
    './data/animations/WalkingUpSteps08_000.bvh',
    './data/animations/WalkingUpSteps09_000.bvh',
    './data/animations/WalkingUpSteps10_000.bvh',
    './data/animations/WalkingUpSteps11_000.bvh',
    './data/animations/WalkingUpSteps12_000.bvh',

    './data/animations/WalkingUpSteps01_000_mirror.bvh',
    './data/animations/WalkingUpSteps02_000_mirror.bvh',
    './data/animations/WalkingUpSteps03_000_mirror.bvh',
    './data/animations/WalkingUpSteps04_000_mirror.bvh',
    './data/animations/WalkingUpSteps04_001_mirror.bvh',
    './data/animations/WalkingUpSteps05_000_mirror.bvh',
    './data/animations/WalkingUpSteps06_000_mirror.bvh',
    './data/animations/WalkingUpSteps07_000_mirror.bvh',
    './data/animations/WalkingUpSteps08_000_mirror.bvh',
    './data/animations/WalkingUpSteps09_000_mirror.bvh',
    './data/animations/WalkingUpSteps10_000_mirror.bvh',
    './data/animations/WalkingUpSteps11_000_mirror.bvh',
    './data/animations/WalkingUpSteps12_000_mirror.bvh',

    './data/animations/NewCaptures01_000.bvh',
    './data/animations/NewCaptures02_000.bvh',
    './data/animations/NewCaptures03_000.bvh',
    './data/animations/NewCaptures03_001.bvh',
    './data/animations/NewCaptures03_002.bvh',
    './data/animations/NewCaptures04_000.bvh',
    './data/animations/NewCaptures05_000.bvh',
    './data/animations/NewCaptures07_000.bvh',
    './data/animations/NewCaptures08_000.bvh',
    './data/animations/NewCaptures09_000.bvh',
    './data/animations/NewCaptures10_000.bvh',
    './data/animations/NewCaptures11_000.bvh',

    './data/animations/NewCaptures01_000_mirror.bvh',
    './data/animations/NewCaptures02_000_mirror.bvh',
    './data/animations/NewCaptures03_000_mirror.bvh',
    './data/animations/NewCaptures03_001_mirror.bvh',
    './data/animations/NewCaptures03_002_mirror.bvh',
    './data/animations/NewCaptures04_000_mirror.bvh',
    './data/animations/NewCaptures05_000_mirror.bvh',
    './data/animations/NewCaptures07_000_mirror.bvh',
    './data/animations/NewCaptures08_000_mirror.bvh',
    './data/animations/NewCaptures09_000_mirror.bvh',
    './data/animations/NewCaptures10_000_mirror.bvh',
    './data/animations/NewCaptures11_000_mirror.bvh',
]

if skd == Original:
    data_terrain = ['./data/animations/LocomotionFlat01_000.bvh']
else:
    data_terrain = [f'./data/reference/{skd.FILE_NAME}.bvh']

database_name = skd.DATABASE_NAME

""" filter out joints """
def filter_joints(anim, names):
    indices = []
    for j in range(len(skd.FILTER_OUT)):
        for i in range(len(names)):
            if skd.FILTER_OUT[j] in names[i]:
                indices.append(i)
    indices = sorted(indices, reverse=True)
    
    #detect in-middle bones
    # for j in indices:
        # print("delete:" + names[j])
        # if j in anim.parents:
            # print("%s has children!" % names[j])
    
    """
    #note: if middle joints are filtered out, we need re-calc local xforms
    #now there's not any, so skip
    #cache global xform
    global_xforms = Animation.transforms_global(anim)
    global_xforms = np.delete(global_xforms, indices)
    local_xforms = np.zeros(global_xforms.shape)
    """
    
    anim2 = anim.copy()
    anim2.orients = np.delete(anim2.orients, indices, 0)
    anim2.offsets = np.delete(anim2.offsets, indices, 0)
    anim2.parents = np.delete(anim2.parents, indices, 0)
    anim2.rotations = Quaternions(np.delete(anim2.rotations, indices, 1))
    anim2.positions = np.delete(anim2.positions, indices, 1)

    names2 = names.copy()
    for i in indices:
        del names2[i]
        for j in range(len(anim2.parents)):
            if anim2.parents[j] >= i:
                anim2.parents[j] -= 1
    
    """
    #calc local xform based on stripped global xform
    invf = lambda x : Animation.transforms_inv(x)
    inv_xforms = list(map(invf, global_xforms))
    
    for i in range(1, anim.shape[1]):
        local_xforms[:,i] = Animation.transforms_multiply(inv_xforms[:,anim.parents[i]], global_xforms[:,i])
    
    anim.rotations = Quaternions.from_transforms(local_xforms)
    anim.positions = local_xforms[:,:,:3,3] / local_xforms[:,:,3:,3]
    """
    
    return anim2, names2


""" Load Terrain Patches """

patches_database = np.load(skd.PATCHES_NAME)
patches = patches_database['X'].astype(np.float32)
patches_coord = patches_database['C'].astype(np.float32)

""" Processing Functions """

def process_data(anim, phase, gait, type='flat'):
    
    """ Do FK 前向动力学 """
    # 每帧 每关节的全局旋转矩阵 根据各自的局部旋转得到全局旋转
    global_xforms = Animation.transforms_global(anim)
    # 每帧 每关节的全局位置
    global_positions = global_xforms[:, :, :3, 3] / global_xforms[:, :, 3:, 3]
    # 每帧 每关节的全局旋转四元数表示
    global_rotations = Quaternions.from_transforms(global_xforms)
    
    """ Extract Forward Direction """
    # 肩和髋向量之和
    across = (
        (global_positions[:,skd.SDR_L] - global_positions[:,skd.SDR_R]) + 
        (global_positions[:,skd.HIP_L] - global_positions[:,skd.HIP_R]))
    # 取单位向量
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    """ Smooth Forward Direction """
    
    # 顺滑 + 与Y轴做叉乘
    # 得到 Forward
    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0,1,0]])), direction_filterwidth, axis=0, mode='nearest')    
    # 取单位向量
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    root_rotation = Quaternions.between(forward, 
        np.array([[0,0,1]]).repeat(len(forward), axis=0))[:,np.newaxis] 
    
    """ Local Space 计算局部坐标信息 """
    
    local_positions = global_positions.copy()
    
    # 局部位置
    # 各帧中所有关节x,z - 根关节x,z 
    # 局部坐标信息的定义和 Games105不同
    # 根据根关节，而不是根据父关节
    local_positions[:,:,0] = local_positions[:,:,0] - local_positions[:,0:1,0]
    local_positions[:,:,2] = local_positions[:,:,2] - local_positions[:,0:1,2]
    
    # 左乘 root_rotation[:-1] 意义是 "以第一帧作为参考帧" 消除绝对方向的影响
    local_positions = root_rotation[:-1] * local_positions[:-1]
    
    # 局部速度
    # 以第一帧作为参考帧
    global_velocities = global_positions[1:] - global_positions[:-1] # 下一帧 - 上一帧 = 速度/方向
    local_velocities = root_rotation[:-1] * global_velocities
    
    # 局部旋转
    # 以第一帧作为参考帧
    local_rotations = ((root_rotation[:-1] * global_rotations[:-1]))
    
    #print('hips (w,x,y,z)=' + str(abs((root_rotation[:-1] * global_rotations[:-1]))[0][0]))
    #print('hips log(w,x,y,z)=' + str(local_rotations[0][0]))
    
    # 根节点 速度 & 旋转速度
    root_velocity = root_rotation[:-1] * (global_positions[1:,0:1] - global_positions[:-1,0:1])
    root_rvelocity = Pivots.from_quaternions(root_rotation[1:] * -root_rotation[:-1]).ps
    
    """ Foot Contacts 计算脚触地状态 """

    # 针对每只脚的脚跟和脚尖, 利用相邻帧的差值是否小于某个阈值来判断
    # 分别判断左脚的脚跟和脚尖的触地状态, 右脚的脚跟和脚尖的触地状态, 触地为1, 否则为0
    # 分别存储脚跟和脚尖的触地信息
    fid_l, fid_r = np.array(skd.FOOT_L), np.array(skd.FOOT_R)
    velfactor = np.array([0.02, 0.02])
    
    # [1:,_,_], [:-1,_,_] 代表相邻帧
    # [_,fdl,_] 代表左脚
    # [_,_,0] 代表关节的x坐标
    feet_l_x = (global_positions[1:,fid_l,0] - global_positions[:-1,fid_l,0])**2
    feet_l_y = (global_positions[1:,fid_l,1] - global_positions[:-1,fid_l,1])**2
    feet_l_z = (global_positions[1:,fid_l,2] - global_positions[:-1,fid_l,2])**2
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor)).astype(np.float)
    
    feet_r_x = (global_positions[1:,fid_r,0] - global_positions[:-1,fid_r,0])**2
    feet_r_y = (global_positions[1:,fid_r,1] - global_positions[:-1,fid_r,1])**2
    feet_r_z = (global_positions[1:,fid_r,2] - global_positions[:-1,fid_r,2])**2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor)).astype(np.float)
    
    """ Phase 相位 """
    
    # 相邻帧的相位变化量delta
    dphase = phase[1:] - phase[:-1]
    dphase[dphase < 0] = (1.0-phase[:-1]+phase[1:])[dphase < 0]
    
    """ Adjust Crouching Gait Value """
    
    # if type == 'flat':
    #     crouch_low, crouch_high = 80, 130
    #     head = skd.HEAD
    #     gait[:-1,3] = 1 - np.clip((global_positions[:-1,head,1] - 80) / (130 - 80), 0, 1)
    #     gait[-1,3] = gait[-2,3]
        
    """ Load prev rotations across files (poles maybe adjusted) """
    global prev_rotations
    if prev_rotations is None:
        prev_rotations = local_rotations[window-1]

    """ Start Windows """
    
    Pc, Xc, Yc = [], [], []
    
    # 窗口中心是 i ~~ [window, len(anim) - window-1]
    # 窗口大小是 2 * window
    for i in range(window, len(anim)-window-1, 1):
        
        # global_positions[i-window:i+window : 10, 0]
        # i-window:i+window  范围切片
        # 10  步长
        # 0  第二个维度的数据 根节点
        # 第i帧前后共2*window大小 步长10 的数据
        # 减去 第i帧的数据
        # 如果window = 60, 则window * 2 = 120, 120 / 10 = 12帧
        rootposs = root_rotation[i:i+1,0] * (global_positions[i-window:i+window:10,0] - global_positions[i:i+1,0])
        rootdirs = root_rotation[i:i+1,0] * forward[i-window:i+window:10]
        rootgait = gait[i-window:i+window:10]
        
        # 第一帧下标是 Window
        Pc.append(phase[i])
        
        """ Unify Quaternions To Single Pole """
        """ Between Prev vs Next Keys """
        
        """ [Grassia1998]:
            The procedure followed by Yahia [13] of limiting the range of the log map to |log(r)| ≤ π does not suffice.
            A log mapping that does guarantee the geodesic approximation picks the mapping for each successive key that
            minimizes the Euclidean distance to the mapping of the previous key.
            Given such a log map that considers the previous mapping when calculating the current mapping, the results of interpolating
            in S3 and R3 may be visually indistinguishable for many applications, including keyframing.  """
        antipode = prev_rotations.dot(local_rotations[i]) < 0
        local_rotations[i][antipode] = Quaternions(-local_rotations[i][antipode].qs)
        prev_rotations = local_rotations[i]

        # print("*********************************")
        # print(i)
        # #print(*antipode, sep=' ')
        # for j in range(antipode.size):
        #     print("%6d " % (j+1), end='' if j < antipode.size-1 else '\n')
        # for j in range(antipode.size):
        #     print("%6s " % antipode[j], end='' if j < antipode.size-1 else '\n')
        # for j in range(antipode.size):
        #     print("%06.3f " % local_rotations[i-1][j].reals, end='' if j < antipode.size-1 else '\n')
        # for j in range(antipode.size):
        #     print("%06.3f " % local_rotations[i][j].reals, end='' if j < antipode.size-1 else '\n')
            
        # print(*local_rotations[i-1][antipode], sep=' ')
        # print(*local_rotations[i][antipode], sep=' ')

        # # print("*********************************")
        # # print(local_rotations[i][antipode])
        # # print("*********************************")
        
        Xc.append(np.hstack([
                rootposs[:,0].ravel(), rootposs[:,2].ravel(),   # Trajectory Pos, t_i_p
                rootdirs[:,0].ravel(), rootdirs[:,2].ravel(),   # Trajectory Dir, t_i_d
                rootgait[:,0].ravel(), rootgait[:,1].ravel(),   # Trajectory Gait, t_i_g
                rootgait[:,2].ravel(), rootgait[:,3].ravel(),
                rootgait[:,4].ravel(), rootgait[:,5].ravel(), 
                local_positions[i-1].ravel(),                   # Joint Pos, j_i-1_p
                local_velocities[i-1].ravel(),                  # Joint Vel, j_i-1_v
                ]))
        
        rootposs_next = root_rotation[i+1:i+2,0] * (global_positions[i+1:i+window+1:10,0] - global_positions[i+1:i+2,0])
        rootdirs_next = root_rotation[i+1:i+2,0] * forward[i+1:i+window+1:10]   
        
        Yc.append(np.hstack([
                root_velocity[i,0,0].ravel(),                                   # Root Vel X, r_i_x
                root_velocity[i,0,2].ravel(),                                   # Root Vel Y, r_i_z
                root_rvelocity[i].ravel(),                                      # Root Rot Vel, r_i_a
                dphase[i],                                                      # Change in Phase, p_i
                np.concatenate([feet_l[i], feet_r[i]], axis=-1),                # Contacts, c_i
                rootposs_next[:,0].ravel(), rootposs_next[:,2].ravel(),         # Next Trajectory Pos, t_i+1_p
                rootdirs_next[:,0].ravel(), rootdirs_next[:,2].ravel(),         # Next Trajectory Dir, t_i+1_d
                local_positions[i].ravel(),                                     # Joint Pos, j_i_p
                local_velocities[i].ravel(),                                    # Joint Vel, j_i_v
                local_rotations[i].log().ravel()                                # Joint Rot, j_i_a
                ]))
                                                
    return np.array(Pc), np.array(Xc), np.array(Yc)
    

""" Sampling Patch Heightmap """    

def patchfunc(P, Xp, hscale=3.937007874, vscale=3.0):
    
    Xp = Xp / hscale + np.array([P.shape[1]//2, P.shape[2]//2])
    
    A = np.fmod(Xp, 1.0)
    X0 = np.clip(np.floor(Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1]))
    X1 = np.clip(np.ceil (Xp).astype(np.int), 0, np.array([P.shape[1]-1, P.shape[2]-1]))
    
    H0 = P[:,X0[:,0],X0[:,1]]
    H1 = P[:,X0[:,0],X1[:,1]]
    H2 = P[:,X1[:,0],X0[:,1]]
    H3 = P[:,X1[:,0],X1[:,1]]
    
    HL = (1-A[:,0]) * H0 + (A[:,0]) * H2
    HR = (1-A[:,0]) * H1 + (A[:,0]) * H3
    
    return (vscale * ((1-A[:,1]) * HL + (A[:,1]) * HR))[...,np.newaxis]
    

def process_heights(anim, nsamples=10, type='flat'):
    
    """ Do FK """
    
    global_xforms = Animation.transforms_global(anim)
    global_positions = global_xforms[:,:,:3,3] / global_xforms[:,:,3:,3]
    global_rotations = Quaternions.from_transforms(global_xforms)
    
    """ Extract Forward Direction """
    
    across = (
        (global_positions[:,skd.SDR_L] - global_positions[:,skd.SDR_R]) + 
        (global_positions[:,skd.HIP_L] - global_positions[:,skd.HIP_R]))
    across = across / np.sqrt((across**2).sum(axis=-1))[...,np.newaxis]
    
    """ Smooth Forward Direction """
    
    direction_filterwidth = 20
    forward = filters.gaussian_filter1d(
        np.cross(across, np.array([[0,1,0]])), direction_filterwidth, axis=0, mode='nearest')    
    forward = forward / np.sqrt((forward**2).sum(axis=-1))[...,np.newaxis]

    root_rotation = Quaternions.between(forward, 
        np.array([[0,0,1]]).repeat(len(forward), axis=0))[:,np.newaxis] 

    """ Foot Contacts """
    
    fid_l, fid_r = np.array(skd.FOOT_L), np.array(skd.FOOT_R)
    velfactor = np.array([0.02, 0.02])
    
    feet_l_x = (global_positions[1:,fid_l,0] - global_positions[:-1,fid_l,0])**2
    feet_l_y = (global_positions[1:,fid_l,1] - global_positions[:-1,fid_l,1])**2
    feet_l_z = (global_positions[1:,fid_l,2] - global_positions[:-1,fid_l,2])**2
    feet_l = (((feet_l_x + feet_l_y + feet_l_z) < velfactor))
    
    feet_r_x = (global_positions[1:,fid_r,0] - global_positions[:-1,fid_r,0])**2
    feet_r_y = (global_positions[1:,fid_r,1] - global_positions[:-1,fid_r,1])**2
    feet_r_z = (global_positions[1:,fid_r,2] - global_positions[:-1,fid_r,2])**2
    feet_r = (((feet_r_x + feet_r_y + feet_r_z) < velfactor))
    
    feet_l = np.concatenate([feet_l, feet_l[-1:]], axis=0)
    feet_r = np.concatenate([feet_r, feet_r[-1:]], axis=0)
    
    """ Toe and Heel Heights """
    
    toe_h, heel_h = 4.0, 5.0
    
    """ Foot Down Positions """
    
    feet_down = np.concatenate([
        global_positions[feet_l[:,0],fid_l[0]] - np.array([0, heel_h, 0]),
        global_positions[feet_l[:,1],fid_l[1]] - np.array([0,  toe_h, 0]),
        global_positions[feet_r[:,0],fid_r[0]] - np.array([0, heel_h, 0]),
        global_positions[feet_r[:,1],fid_r[1]] - np.array([0,  toe_h, 0])
    ], axis=0)
    
    """ Foot Up Positions """
    
    feet_up = np.concatenate([
        global_positions[~feet_l[:,0],fid_l[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_l[:,1],fid_l[1]] - np.array([0,  toe_h, 0]),
        global_positions[~feet_r[:,0],fid_r[0]] - np.array([0, heel_h, 0]),
        global_positions[~feet_r[:,1],fid_r[1]] - np.array([0,  toe_h, 0])
    ], axis=0)
    
    """ Down Locations """
    feet_down_xz = np.concatenate([feet_down[:,0:1], feet_down[:,2:3]], axis=-1)
    feet_down_xz_mean = feet_down_xz.mean(axis=0)
    feet_down_y = feet_down[:,1:2]
    feet_down_y_mean = feet_down_y.mean(axis=0)
    feet_down_y_std  = feet_down_y.std(axis=0)
    """ Up Locations """
        
    feet_up_xz = np.concatenate([feet_up[:,0:1], feet_up[:,2:3]], axis=-1)
    feet_up_y = feet_up[:,1:2]
    
    if len(feet_down_xz) == 0:
    
        """ No Contacts """
    
        terr_func = lambda Xp: np.zeros_like(Xp)[:,:1][np.newaxis].repeat(nsamples, axis=0)
        
    elif type == 'flat':
        
        """ Flat """
        
        terr_func = lambda Xp: np.zeros_like(Xp)[:,:1][np.newaxis].repeat(nsamples, axis=0) + feet_down_y_mean
    
    else:
        
        """ Terrain Heights """
        
        terr_down_y = patchfunc(patches, feet_down_xz - feet_down_xz_mean)
        terr_down_y_mean = terr_down_y.mean(axis=1)
        terr_down_y_std  = terr_down_y.std(axis=1)
        terr_up_y = patchfunc(patches, feet_up_xz - feet_down_xz_mean)
        
        """ Fitting Error """
        
        terr_down_err = 0.1 * ((
            (terr_down_y - terr_down_y_mean[:,np.newaxis]) -
            (feet_down_y - feet_down_y_mean)[np.newaxis])**2)[...,0].mean(axis=1)
        
        terr_up_err = (np.maximum(
            (terr_up_y - terr_down_y_mean[:,np.newaxis]) -
            (feet_up_y - feet_down_y_mean)[np.newaxis], 0.0)**2)[...,0].mean(axis=1)
        
        """ Jumping Error """
        
        if type == 'jumpy':
            terr_over_minh = 5.0
            terr_over_err = (np.maximum(
                ((feet_up_y - feet_down_y_mean)[np.newaxis] - terr_over_minh) -
                (terr_up_y - terr_down_y_mean[:,np.newaxis]), 0.0)**2)[...,0].mean(axis=1)
        else:
            terr_over_err = 0.0
        
        """ Fitting Terrain to Walking on Beam """
        
        if type == 'beam':

            beam_samples = 1
            beam_min_height = 40.0

            beam_c = global_positions[:,0]
            beam_c_xz = np.concatenate([beam_c[:,0:1], beam_c[:,2:3]], axis=-1)
            beam_c_y = patchfunc(patches, beam_c_xz - feet_down_xz_mean)

            beam_o = (
                beam_c.repeat(beam_samples, axis=0) + np.array([50, 0, 50]) * 
                rng.normal(size=(len(beam_c)*beam_samples, 3)))

            beam_o_xz = np.concatenate([beam_o[:,0:1], beam_o[:,2:3]], axis=-1)
            beam_o_y = patchfunc(patches, beam_o_xz - feet_down_xz_mean)

            beam_pdist = np.sqrt(((beam_o[:,np.newaxis] - beam_c[np.newaxis,:])**2).sum(axis=-1))
            beam_far = (beam_pdist > 15).all(axis=1)

            terr_beam_err = (np.maximum(beam_o_y[:,beam_far] - 
                (beam_c_y.repeat(beam_samples, axis=1)[:,beam_far] - 
                 beam_min_height), 0.0)**2)[...,0].mean(axis=1)

        else:
            terr_beam_err = 0.0
        
        """ Final Fitting Error """
        
        terr = terr_down_err + terr_up_err + terr_over_err + terr_beam_err
        
        """ Best Fitting Terrains """
        
        terr_ids = np.argsort(terr)[:nsamples]
        terr_patches = patches[terr_ids]
        terr_basic_func = lambda Xp: (
            (patchfunc(terr_patches, Xp - feet_down_xz_mean) - 
            terr_down_y_mean[terr_ids][:,np.newaxis]) + feet_down_y_mean)
        
        """ Terrain Fit Editing """
        terr_residuals = feet_down_y - terr_basic_func(feet_down_xz)
        terr_fine_func = [RBF(smooth=0.1, function='linear', epsilon=1e-10) for _ in range(nsamples)]
        for i in range(nsamples): terr_fine_func[i].fit(feet_down_xz, terr_residuals[i])
        terr_func = lambda Xp: (terr_basic_func(Xp) + np.array([ff(Xp) for ff in terr_fine_func]))
        
        
    """ Get Trajectory Terrain Heights """
    
    root_offsets_c = global_positions[:,0]
    root_offsets_r = (-root_rotation[:,0] * np.array([[+25, 0, 0]])) + root_offsets_c
    root_offsets_l = (-root_rotation[:,0] * np.array([[-25, 0, 0]])) + root_offsets_c

    root_heights_c = terr_func(root_offsets_c[:,np.array([0,2])])[...,0]
    root_heights_r = terr_func(root_offsets_r[:,np.array([0,2])])[...,0]
    root_heights_l = terr_func(root_offsets_l[:,np.array([0,2])])[...,0]
    
    """ Find Trajectory Heights at each Window """
    
    root_terrains = []
    root_averages = []
    for i in range(window, len(anim)-window, 1): 
        root_terrains.append(
            np.concatenate([
                root_heights_r[:,i-window:i+window:10],
                root_heights_c[:,i-window:i+window:10],
                root_heights_l[:,i-window:i+window:10]], axis=1))
        root_averages.append(root_heights_c[:,i-window:i+window:10].mean(axis=1))
     
    root_terrains = np.swapaxes(np.array(root_terrains), 0, 1)
    root_averages = np.swapaxes(np.array(root_averages), 0, 1)
    
    return root_terrains, root_averages

""" Phases, Inputs, Outputs """
    
P, X, Y = [], [], []
            
for data in data_terrain:
    
    print('Processing Clip %s' % data)
    
    """ Data Types """
    
    if   'LocomotionFlat12_000' in data: type = 'jumpy'
    elif 'NewCaptures01_000'    in data: type = 'flat'
    elif 'NewCaptures02_000'    in data: type = 'flat'
    elif 'NewCaptures03_000'    in data: type = 'jumpy'
    elif 'NewCaptures03_001'    in data: type = 'jumpy'
    elif 'NewCaptures03_002'    in data: type = 'jumpy'
    elif 'NewCaptures04_000'    in data: type = 'jumpy'
    elif 'WalkingUpSteps06_000' in data: type = 'beam'
    elif 'WalkingUpSteps09_000' in data: type = 'flat'
    elif 'WalkingUpSteps10_000' in data: type = 'flat'
    elif 'WalkingUpSteps11_000' in data: type = 'flat'
    elif 'Flat' in data: type = 'flat'
    elif 'walk' in data: type = 'flat'
    else: type = 'rocky'
    
    """ Load Data """
    
    anim, names, _ = BVH.load(data)
    anim2, names2 = filter_joints(anim, names)
    
    # print(names)
    # print(names2)
    # d = dict()
    # for i in range(len(anim.parents)):
    #     if anim.parents[i] not in d:
    #         d[anim.parents[i]] = []
    #     d[anim.parents[i]].append(i)
    # print(len(anim.parents), len(names))
    # for key in d:
    #     print(f"{names[key]} ->\n{[names[i] for i in d[key]]}")
    
    """ Dump joint lists"""
    #BVH.save("stripped.bvh", anim, names, 1.0/120.0, 'zyx', True, True)
    
    with open("jointlist.txt", "w") as f:
        for j in range(len(names2)):
            jname = names2[j]
            p = anim.parents[names.index(jname)]
            while p!=-1:
                jname = names[p] + "/" + jname
                p = anim.parents[p]
            f.writelines("\"" + jname + "\",\n")

    anim = anim2
    names = names2
    
    anim.offsets *= to_meters
    anim.positions *= to_meters
    anim = anim[::2]
    """ Load Phase / Gait """
    
    # 相位数据
    phase = np.loadtxt(data.replace('.bvh', '.phase'))[::2]
    gait = np.loadtxt(data.replace('.bvh', '.gait'))[::2]

    """ Merge Jog / Run and Crouch / Crawl """
    
    # 8种运动风格 合并成6种 不是binary-vector
    # 1.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000 0.00000
    # 0.30667 0.00000 0.00000 0.69333 0.00000 0.00000 0.00000 0.00000
    # 0.00000 0.00000 0.00000 1.00000 0.00000 0.00000 0.00000 0.00000
    gait = np.concatenate([
        gait[:,0:1],
        gait[:,1:2],
        gait[:,2:3] + gait[:,3:4],
        gait[:,4:5] + gait[:,6:7],
        gait[:,5:6],
        gait[:,7:8]
    ], axis=-1)

    """ Preprocess Data """
    
    Pc, Xc, Yc = process_data(anim, phase, gait, type=type)

    # 随机选取的帧段 用于后续的地形矫正 无实际意义
    with open(data.replace('.bvh', '_footsteps.txt'), 'r') as f:
        footsteps = f.readlines()
        
    # print(f"Footsteps: {footsteps} Length: {len(footsteps)-1}")
    
    """ For each Locomotion Cycle fit Terrains """
    
    for li in range(len(footsteps)-1):
    
        curr, next = footsteps[li+0].split(' '), footsteps[li+1].split(' ')
        # print(f"Curr: {curr}, Next: {next}, Window:{window}")
        
        """ Ignore Cycles marked with '*' or not in range """
        
        if len(curr) == 3 and curr[2].strip().endswith('*'): continue
        if len(next) == 3 and next[2].strip().endswith('*'): continue
        if len(next) <  2: continue
        if int(curr[0])//2-window < 0: continue
        if int(next[0])//2-window >= len(Xc): continue 
        
        """ Fit Heightmaps """
        
        slc = slice(int(curr[0])//2-window, int(next[0])//2-window+1)
        H, Hmean = process_heights(anim[
            int(curr[0])//2-window:
            int(next[0])//2+window+1], type=type)
        
        # print(H, Hmean)

        for h, hmean in zip(H, Hmean):
            
            Xh, Yh = Xc[slc].copy(), Yc[slc].copy()
            
            """ Reduce Heights in Input/Output to Match"""
            
            xo_s, xo_e = ((window*2)//10)*10+1, ((window*2)//10)*10+njoints*3+1
            yo_s, yo_e = 8+(window//10)*4+1, 8+(window//10)*4+njoints*3+1
            Xh[:,xo_s:xo_e:3] -= hmean[...,np.newaxis]
            Yh[:,yo_s:yo_e:3] -= hmean[...,np.newaxis]
            Xh = np.concatenate([Xh, h - hmean[...,np.newaxis]], axis=-1)
            
            """ Append to Data """
            
            P.append(np.hstack([0.0, Pc[slc][1:-1], 1.0]).astype(np.float32))
            X.append(Xh.astype(np.float32))
            Y.append(Yh.astype(np.float32))
  
""" Clip Statistics """
  
print('Total Clips: %i' % len(X))
print('Shortest Clip: %i' % min(map(len,X)))
print('Longest Clip: %i' % max(map(len,X)))
print('Average Clip: %i' % np.mean(list(map(len,X))))
#print("YP0: %3.3f %3.3f %3.3f" %(float(Y[0][0][32]), float(Y[0][0][33]), float(Y[0][0][34])) )
#print("YR0: %3.4f %3.4f %3.4f" %(float(Y[0][0][218]), float(Y[0][0][219]), float(Y[0][0][220])) )

""" Merge Clips """

print('Merging Clips...')

Xun = np.concatenate(X, axis=0)
Yun = np.concatenate(Y, axis=0)
Pun = np.concatenate(P, axis=0)

print(Xun.shape, Yun.shape, Pun.shape)

print('Saving Database...')

np.savez_compressed(database_name, Xun=Xun, Yun=Yun, Pun=Pun)

