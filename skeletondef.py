class Original:
    """
    ROOT Hips
    JOINT LHipJoint
        JOINT LeftUpLeg
            JOINT LeftLeg
                JOINT LeftFoot
                    JOINT LeftToeBase
    JOINT RHipJoint
        JOINT RightUpLeg
            JOINT RightLeg
                JOINT RightFoot
                    JOINT RightToeBase
    JOINT LowerBack
        JOINT Spine
            JOINT Spine1
                JOINT Neck
                    JOINT Neck1
                        JOINT Head
                JOINT LeftShoulder
                    JOINT LeftArm
                        JOINT LeftForeArm
                            JOINT LeftHand
                                JOINT LeftFingerBase
                                    JOINT LeftHandIndex1
                                JOINT LThumb
                JOINT RightShoulder
                    JOINT RightArm
                        JOINT RightForeArm
                            JOINT RightHand
                                JOINT RightFingerBase
                                    JOINT RightHandIndex1
                                JOINT RThumb
    """
    PATCHES_NAME = 'patches.npz'
    DATABASE_NAME = 'database.npz'
    FILE_NAME = None
    WINDOW = 60
    JOINT_NUM = 31

    SDR_L, SDR_R, HIP_L, HIP_R = 18, 25, 2, 7
    FOOT_L = [4,5]
    FOOT_R = [9,10]
    HEAD = 16
    FILTER_OUT = []
    # JOINT_SCALE = 1
    JOINT_SCALE = 5.644

    JOINT_WEIGHTS = [
        1,
        1e-10, 1, 1, 1, 1,
        1e-10, 1, 1, 1, 1,
        1e-10, 1, 1,
        1e-10, 1, 1,
        1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10,
        1e-10, 1, 1, 1, 1e-10, 1e-10, 1e-10 ]
    # JOINT_WEIGHTS = [1e-10] * JOINT_NUM

class Games:
    """ Skeleton for walk_and_turn_left.bvh
    0 ROOT RootJoint
    1    JOINT lHip
    2        JOINT lKnee
    3            JOINT lAnkle
    4                JOINT lToeJoint
    5    JOINT rHip
    6        JOINT rKnee
    7            JOINT rAnkle
    8                JOINT rToeJoint
    9    JOINT pelvis_lowerback
    10        JOINT lowerback_torso
    11            JOINT torso_head
    12            JOINT lTorso_Clavicle
    13                JOINT lShoulder
    14                    JOINT lElbow
    15                        JOINT lWrist
    16            JOINT rTorso_Clavicle
    17                JOINT rShoulder
    18                    JOINT rElbow
    19                        JOINT rWrist
    """
    # SDR_L, SDR_R, HIP_L, HIP_R = 13, 17, 1, 5
    # FOOT_L = [3,4]
    # FOOT_R = [7,8]
    # HEAD = 11
    # WINDOW = 30
    # FILE_NAME = 'walk_and_turn_left'
    
    
    """ Skeleton for WalkF.bvh
    0 ROOT RootJoint
    1     JOINT lHip
    2         JOINT lKnee
    3             JOINT lAnkle
    4                 JOINT lToeJoint
    5     JOINT pelvis_lowerback
    6         JOINT lowerback_torso
    7             JOINT lTorso_Clavicle
    8                 JOINT lShoulder
    9                     JOINT lElbow
    10                         JOINT lWrist
    11             JOINT rTorso_Clavicle
    12                 JOINT rShoulder
    13                     JOINT rElbow
    14                         JOINT rWrist
    15             JOINT torso_head
    16     JOINT rHip
    17         JOINT rKnee
    18             JOINT rAnkle
    19                 JOINT rToeJoint
    """
    # SDR_L, SDR_R, HIP_L, HIP_R = 8, 12, 1, 16
    # FOOT_L = [3,4]
    # FOOT_R = [18,19]
    # HEAD = 15
    # WINDOW = 30
    # FILE_NAME = 'walkF'
    
    
    """ Skeleton for long_walk.bvh
    0 ROOT RootJoint
    1    JOINT pelvis_lowerback
    2        JOINT lowerback_torso
    3            JOINT torso_head
    4            JOINT rTorso_Clavicle
    5                JOINT rShoulder
    6                    JOINT rElbow
    7                        JOINT rWrist
    8                JOINT lTorso_Clavicle
    9                    JOINT lShoulder
    10                        JOINT lElbow
    11                            JOINT lWrist
    12        JOINT rHip
    13            JOINT rKnee
    14                JOINT rAnkle
    15                    JOINT rToeJoint
    16        JOINT lHip
    17            JOINT lKnee
    18                JOINT lAnkle
    19                    JOINT lToeJoint
    """
    SDR_L, SDR_R, HIP_L, HIP_R = 9, 5, 16, 12
    FOOT_L = [18,19]
    FOOT_R = [14,15]
    HEAD = 3
    WINDOW = 60
    FILE_NAME = 'long_walk'
    
    
    PATCHES_NAME = 'patches_games.npz'
    DATABASE_NAME = 'database_games.npz'
    JOINT_NUM = 20
    FILTER_OUT = []
    # JOINT_SCALE = 5.644
    # JOINT_SCALE = 1
    JOINT_SCALE = 100
    # JOINT_WEIGHTS = [1] * JOINT_NUM
    JOINT_WEIGHTS = [1, 1, 1, 1, 1e-10, 1, 1, 1, 1e-10, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]

# TODO 1: 选择骨骼
# Choose = Original
Choose = Games


# """
# ROOT
# JOINT Hip_R
#   JOINT HipPart1_R
#     JOINT HipPart2_R
#       JOINT Knee_R
#         JOINT KneePart1_R
#           JOINT KneePart2_R
#             JOINT Ankle_R
#               JOINT Toes_R
#                 JOINT ToesEnd_R
# JOINT Spine1_M
#   JOINT Spine2_M
#     JOINT Spine3_M
#       JOINT Chest_M
#         JOINT Scapula_R
#           JOINT Shoulder_R
#             JOINT ShoulderPart1_R
#               JOINT ShoulderPart2_R
#                 JOINT Elbow_R
#                   JOINT ElbowPart1_R
#                     JOINT ElbowPart2_R
#                       JOINT Wrist_R
#          JOINT Neck_M
#            JOINT NeckPart1_M
#              JOINT Head_M
#                JOINT HeadEnd_M
#                JOINT Head_M_spare
#           JOINT Scapula_L
#             JOINT Shoulder_L
#               JOINT ShoulderPart1_L
#                 JOINT ShoulderPart2_L
#                   JOINT Elbow_L
#                     JOINT ElbowPart1_L
#                       JOINT ElbowPart2_L
#                         JOINT Wrist_L
#   JOINT Hip_L
#     JOINT HipPart1_L
#       JOINT HipPart2_L
#         JOINT Knee_L
#           JOINT KneePart1_L
#             JOINT KneePart2_L
#               JOINT Ankle_L
#                 JOINT Toes_L
#                   JOINT ToesEnd_L
# """
# JOINT_NUM = 44
# SDR_L, SDR_R, HIP_L, HIP_R = 28, 15, 35, 1
# FOOT_L = [41,43]
# FOOT_R = [7,9]
# HEAD = 24
# FILTER_OUT = ["Cup", "Finger", "Head_rivet", "Teeth", "Tongue", "Eye", "Pupil", "Iris", "muscleDriver"]
# JOINT_SCALE = 1
# JOINT_WEIGHTS = [
#     1,
#     1, 1e-10, 1e-10, 1, 1e-10, 1e-10, 1, 1e-10, 1,
#     1, 1, 1, 1,
#     1e-10, 1, 1e-10, 1e-10, 1, 1e-10, 1e-10, 1,
#     1, 1e-10, 1, 1e-10, 1e-10,
#     1e-10, 1, 1e-10, 1e-10, 1, 1e-10, 1e-10, 1,
#     1, 1e-10, 1e-10, 1, 1e-10, 1e-10, 1, 1e-10, 1 ]