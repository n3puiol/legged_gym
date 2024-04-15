import numpy as np

from legged_gym.envs.base.legged_robot_config import LeggedRobotCfg, LeggedRobotCfgPPO

parameterNames = [
    'back_bkz',
    'back_bky',
    'back_bkx',
    'l_arm_shz',
    'l_arm_shx',
    'l_arm_ely',
    'l_arm_elx',
    'l_arm_wry',
    'l_arm_wrx',
    'l_arm_wry2',
    'neck_ry',
    'r_arm_shz',
    'r_arm_shx',
    'r_arm_ely',
    'r_arm_elx',
    'r_arm_wry',
    'r_arm_wrx',
    'r_arm_wry2',
    'l_leg_hpz',
    'l_leg_hpx',
    'l_leg_hpy',
    'l_leg_kny',
    'l_leg_aky',
    'l_leg_akx',
    'r_leg_hpz',
    'r_leg_hpx',
    'r_leg_hpy',
    'r_leg_kny',
    'r_leg_aky',
    'r_leg_akx'
]

PARAMETER_NAMES = parameterNames

controllerGains = {
    "l_leg_aky": 400.0,
    "r_leg_aky": 400.0,
    "l_leg_akx": 400.0,
    "r_leg_akx": 400.0,
    "l_leg_kny": 500.0,
    "r_leg_kny": 500.0,
    "l_leg_hpz": 500.0,
    "r_leg_hpz": 500.0,
    "l_leg_hpx": 500.0,
    "r_leg_hpx": 500.0,
    "l_leg_hpy": 500.0,
    "r_leg_hpy": 500.0,
    "back_bkz": 1000.0,
    "back_bky": 1000.0,
    "back_bkx": 1000.0,
    "l_arm_shz": 400.0,
    "r_arm_shz": 400.0,
    "l_arm_shx": 400.0,
    "r_arm_shx": 400.0,
    "l_arm_ely": 300.0,
    "r_arm_ely": 300.0,
    "l_arm_elx": 300.0,
    "r_arm_elx": 300.0,
    "l_arm_wry": 100.0,
    "r_arm_wry": 100.0,
    "l_arm_wrx": 100.0,
    "r_arm_wrx": 100.0,
    "l_arm_wry2": 100.0,
    "r_arm_wry2": 100.0,
    "neck_ry": 100.0,
}

controllerDamping = {
    "l_leg_aky": 10.0,
    "r_leg_aky": 10.0,
    "l_leg_akx": 10.0,
    "r_leg_akx": 10.0,
    "l_leg_kny": 20.0,
    "r_leg_kny": 20.0,
    "l_leg_hpz": 40.0,
    "r_leg_hpz": 40.0,
    "l_leg_hpx": 40.0,
    "r_leg_hpx": 40.0,
    "l_leg_hpy": 40.0,
    "r_leg_hpy": 40.0,
    "back_bkz": 70.0,
    "back_bky": 70.0,
    "back_bkx": 70.0,
    "l_arm_shz": 30.0,
    "r_arm_shz": 30.0,
    "l_arm_shx": 30.0,
    "r_arm_shx": 30.0,
    "l_arm_ely": 30.0,
    "r_arm_ely": 30.0,
    "l_arm_elx": 30.0,
    "r_arm_elx": 30.0,
    "l_arm_wry": 5.0,
    "r_arm_wry": 5.0,
    "l_arm_wrx": 5.0,
    "r_arm_wrx": 5.0,
    "l_arm_wry2": 5.0,
    "r_arm_wry2": 5.0,
    "neck_ry": 10.0,
}

JOINT_LIMITS = {
    "back_bkz": (-0.663225, 0.663225),
    "back_bky": (-0.219388, 0.538783),
    "back_bkx": (-0.523599, 0.523599),
    "l_arm_shz": (-1.5708, 0.785398),
    "l_arm_shx": (-1.5708, 1.5708),
    "l_arm_ely": (0.0, 3.14159),
    "l_arm_elx": (0.0, 2.35619),
    "l_arm_wry": (0.0, 3.14159),
    "l_arm_wrx": (-1.1781, 1.1781),
    "l_arm_wry2": (-0.001, 0.001),
    "neck_ry": (-0.602139, 1.14319),
    "r_arm_shz": (-0.785398, 1.5708),
    "r_arm_shx": (-1.5708, 1.5708),
    "r_arm_ely": (0.0, 3.14159),
    "r_arm_elx": (-2.35619, 0.0),
    "r_arm_wry": (0.0, 3.14159),
    "r_arm_wrx": (-1.1781, 1.1781),
    "r_arm_wry2": (-0.001, 0.001),
    "l_leg_hpz": (-0.174358, 0.786794),
    "l_leg_hpx": (-0.523599, 0.523599),
    "l_leg_hpy": (-1.61234, 0.65764),
    "l_leg_kny": (0.0, 2.35637),
    "l_leg_aky": (-1.0, 0.7),
    "l_leg_akx": (-0.8, 0.8),
    "r_leg_hpz": (-0.786794, 0.174358),
    "r_leg_hpx": (-0.523599, 0.523599),
    "r_leg_hpy": (-1.61234, 0.65764),
    "r_leg_kny": (0.0, 2.35637),
    "r_leg_aky": (-1.0, 0.7),
    "r_leg_akx": (-0.8, 0.8)
}


def convertAngleToActionSpace(jointName: str, angle: float):
    limits = JOINT_LIMITS[jointName]
    angle = (angle + np.pi) % (2 * np.pi) - np.pi

    if angle >= 0:
        action = angle / (limits[1] + 1e-7)
    else:
        action = -angle / (limits[0] + 1e-7)

    return np.clip(action, -1, 1)


def convertActionSpaceToAngle(jointName: str, action: float):
    limits = JOINT_LIMITS[jointName]
    if action >= 0:
        angle = action * limits[1]
    else:
        angle = -action * limits[0]
    return np.clip(angle, limits[0], limits[1])


def convertActionsToAngle(actions: np.ndarray):
    angles = np.zeros(30)
    for i, k in enumerate(parameterNames):
        angles[i] = convertActionSpaceToAngle(k, actions[i])

    return angles


class AtlasDescriptionRoughCfg(LeggedRobotCfg):
    class env(LeggedRobotCfg.env):
        num_envs = 12
        num_observations = 223  # Assuming a similar setup as Cassie, needs verification
        num_actions = 30  # Adjusted to the actual number of controlled joints

    class terrain(LeggedRobotCfg.terrain):
        measured_points_x = np.linspace(-0.5, 0.5, 11).tolist()
        measured_points_y = np.linspace(-0.5, 0.5, 11).tolist()

    class init_state(LeggedRobotCfg.init_state):
        pos = [0.0, 0.0, 1.0]  # Initial x, y, z position
        default_joint_angles = {
            joint_name: 0.0 for joint_name in PARAMETER_NAMES  # Initialized to neutral positions
        }

    class control(LeggedRobotCfg.control):
        stiffness = controllerGains
        damping = controllerDamping
        action_scale = 0.5
        use_actuator_network = False
        decimation = 4

    class asset(LeggedRobotCfg.asset):
        file = '{LEGGED_GYM_ROOT_DIR}/resources/robots/atlas/atlas_v4_with_multisense.urdf'
        name = "atlas_description"
        foot_name = 'foot'
        terminate_after_contacts_on = ['pelvis']
        flip_visual_attachments = False
        self_collisions = 1

    class rewards(LeggedRobotCfg.rewards):
        soft_dof_pos_limit = 0.95
        soft_dof_vel_limit = 0.9
        soft_torque_limit = 0.9
        max_contact_force = 300.
        only_positive_rewards = False

    class scales(LeggedRobotCfg.rewards.scales):
        termination = -200.
        tracking_ang_vel = 1.0
        torques = -5.e-6
        dof_acc = -2.e-7
        lin_vel_z = -0.5
        feet_air_time = 5.
        dof_pos_limits = -1.
        no_fly = 0.25
        dof_vel = -0.0
        ang_vel_xy = -0.0
        feet_contact_forces = -0.


class AtlasDescriptionRoughCfgPPO(LeggedRobotCfgPPO):
    class runner(LeggedRobotCfgPPO.runner):
        run_name = ''
        experiment_name = 'rough_atlas_description'

    class algorithm(LeggedRobotCfgPPO.algorithm):
        entropy_coef = 0.01
