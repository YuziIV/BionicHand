from isaaclab.envs import DirectRLEnvCfg
from isaaclab.sim import SimulationCfg, PhysxCfg
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils.configclass import configclass
from isaaclab.assets import ArticulationCfg, RigidObjectCfg
from isaaclab.actuators.actuator_cfg import ImplicitActuatorCfg
from isaaclab.sim.spawners.from_files import UsdFileCfg
from isaaclab.utils.assets import ISAAC_NUCLEUS_DIR
from isaaclab.markers import VisualizationMarkersCfg
import isaaclab.sim as sim_utils
from pathlib import Path
import isaaclab.envs.mdp as mdp
from isaaclab.sim.spawners.materials.physics_materials_cfg import RigidBodyMaterialCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab_tasks.direct._bionic_hand.urdf_importer import ImportURDF
import os
from pxr import Usd, UsdPhysics, PhysxSchema, Gf, Sdf
import omni.usd

#PLAY WITH THIS
#./isaaclab.sh -p scripts/reinforcement_learning/skrl/play.py --task Isaac-Repose-Cube-BionicHand-Direct-v0 --num_envs 32 --algorithm PPO --checkpoint /workspace/isaaclab/logs/skrl/bionic_arm/2025-04-13_23-51-58_ppo_torch/checkpoints/best_agent.pt --real-time


urdf_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/bionic_arm/assets/urdf/EZARM_SLDASM/urdf/EZARM_SLDASM.urdf"

#7axis config
#usd_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/_bionic_hand/assets/usd/bionic_arm.usd"

#8axis config
usd_path = "/workspace/isaaclab/source/isaaclab_tasks/isaaclab_tasks/direct/_bionic_hand/assets/usd/AZARM.SLDASM/AZARM.SLDASM.usd"

@configclass
class EventCfg:
    """Configuration for randomization."""

    # -- robot
    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="reset",
        min_step_count_between_reset=720,
        params={
            "asset_cfg": SceneEntityCfg("robot"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    robot_joint_stiffness_and_damping = EventTerm(
        func=mdp.randomize_actuator_gains,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "stiffness_distribution_params": (0.75, 1.5),
            "damping_distribution_params": (0.3, 3.0),
            "operation": "scale",
            "distribution": "log_uniform",
        },
    )
    robot_joint_pos_limits = EventTerm(
        func=mdp.randomize_joint_parameters,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("robot", joint_names=".*"),
            "lower_limit_distribution_params": (0.00, 0.01),
            "upper_limit_distribution_params": (0.00, 0.01),
            "operation": "add",
            "distribution": "gaussian",
        },
    )

    object_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "static_friction_range": (0.7, 1.3),
            "dynamic_friction_range": (1.0, 1.0),
            "restitution_range": (1.0, 1.0),
            "num_buckets": 250,
        },
    )
    object_scale_mass = EventTerm(
        func=mdp.randomize_rigid_body_mass,
        min_step_count_between_reset=720,
        mode="reset",
        params={
            "asset_cfg": SceneEntityCfg("object"),
            "mass_distribution_params": (0.5, 1.5),
            "operation": "scale",
            "distribution": "uniform",
        },
    )

    # -- scene
    reset_gravity = EventTerm(
        func=mdp.randomize_physics_scene_gravity,
        mode="interval",
        is_global_time=True,
        interval_range_s=(36.0, 36.0),  # time_s = num_steps * (decimation * dt)
        params={
            "gravity_distribution_params": ([0.0, 0.0, 0.0], [0.0, 0.0, 0.4]),
            "operation": "add",
            "distribution": "gaussian",
        },
    )
 
@configclass
class BionicHandEnvCfg(DirectRLEnvCfg):
    # Scene and sim
    decimation = 2
    episode_length_s = 10.0
    action_space = 8
    observation_space = 123  # (full)
    state_space = 0
    asymmetric_obs = False
    obs_type = "full"
    
    # simulation
    sim: SimulationCfg = SimulationCfg(
        dt=1 / 120,
        render_interval=decimation,
        physics_material=RigidBodyMaterialCfg(
            static_friction=0.35,
            dynamic_friction=0.25,
        ),
        physx=PhysxCfg(
            bounce_threshold_velocity=0.2,
        ),
    )


    # Robot setup using the converted USD file and actuator config
    if not os.path.exists(usd_path):
        urdf_importer = ImportURDF(urdf_path, usd_path)
        status, prim_path = urdf_importer.import_urdf_to_usd()
        print("URDF import status:", status, "Prim path:", prim_path)
    #BIONIC_ARM_CFG = ArticulationCfg(
    robot_cfg = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=sim_utils.UsdFileCfg(
            usd_path=usd_path,
            activate_contact_sensors=False,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=True,
                retain_accelerations=True,
                max_depenetration_velocity=1000.0,
            ),
            articulation_props=sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0005,
            ),
            joint_drive_props=sim_utils.JointDrivePropertiesCfg(drive_type="force"),
            #fixed_tendons_props=sim_utils.FixedTendonPropertiesCfg(stiffness=1e9, damping=0.0)
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, -0.15, 0.5),
            rot=(0.7071068, 0.7071068, 0.0, 0.0),
            joint_pos={".*": 0.0},
        ),
        actuators={
            "fingers": ImplicitActuatorCfg(
                joint_names_expr=[
                    "tilt_joint",
                    "palm_joint",
                    "pinky_joint",
                    "pinky_mimic_joint",
                    "ring_joint",
                    "ring_mimic_joint",
                    "middle_joint",
                    "middle_mimic_joint",
                    "pointer_joint",
                    "pointer_mimic_joint",
                    "thumb_swivel_joint",
                    "thumb_joint",
                    "thumb_mimic_joint",
                ],
                effort_limit_sim={
                    "tilt_joint": 5.0,
                    "palm_joint": 5.0,
                    "pinky_joint": 2.0,
                    "ring_joint": 2.0,
                    "middle_joint": 2.0,
                    "pointer_joint": 2.0,
                    "thumb_swivel_joint": 2.0,
                    "thumb_joint": 2.0,
                },
                stiffness={
                    "tilt_joint": 1.0,
                    "palm_joint": 1.0,
                    "pinky_joint": 1.0,
                    "ring_joint": 1.0,
                    "middle_joint": 1.0,
                    "pointer_joint": 1.0,
                    "thumb_swivel_joint": 1.0,
                    "thumb_joint": 1.0,
                    "pinky_mimic_joint": 1e9,
                    "ring_mimic_joint": 1e9,
                    "middle_mimic_joint": 1e9,
                    "pointer_mimic_joint": 1e9,
                    "thumb_mimic_joint": 1e9,
                },
                damping={
                    "tilt_joint": 0.1,                    
                    "palm_joint": 0.1,
                    "pinky_joint": 0.1,
                    "ring_joint": 0.1,
                    "middle_joint": 0.1,
                    "pointer_joint": 0.1,
                    "thumb_swivel_joint": 0.1,
                    "thumb_joint": 0.1,
                    "pinky_mimic_joint": 0.0,
                    "ring_mimic_joint": 0.0,
                    "middle_mimic_joint": 0.0,
                    "pointer_mimic_joint": 0.0,
                    "thumb_mimic_joint": 0.0,
                },
                
            ),
        },
        soft_joint_pos_limit_factor=1.0,
    )
    #robot_cfg: ArticulationCfg = BIONIC_ARM_CFG.replace(prim_path="/World/envs/env_.*/Robot").replace(
    #    init_state=ArticulationCfg.InitialStateCfg(
    #        pos=(0.0, -0.15, 0.5),
    #        rot=(0.7071068, 0.7071068, 0.0, 0.0),
    #        joint_pos={".*": 0.0},
    #    )
    #)
    actuated_joint_names = [
        "tilt_joint",
        "palm_joint",
        "pinky_joint",
        "ring_joint",
        "middle_joint",
        "pointer_joint",
        "thumb_swivel_joint",
        "thumb_joint",
    ]
    fingertip_body_names = [
        "pinky_mimic_link",
        "ring_mimic_link",
        "middle_mimic_link",
        "pointer_mimic_link",
        "thumb_mimic_link",
    ]
   
    # in-hand object
    object_cfg: RigidObjectCfg = RigidObjectCfg(
        prim_path="/World/envs/env_.*/object",
        spawn=sim_utils.UsdFileCfg(
            usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
            scale=(0.67, 0.67, 0.67),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                kinematic_enabled=False,
                disable_gravity=False,
                enable_gyroscopic_forces=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=0,
                sleep_threshold=0.005,
                stabilization_threshold=0.0025,
                max_depenetration_velocity=1000.0,
            ),
            mass_props=sim_utils.MassPropertiesCfg(density=567.0),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, -0.39, 0.6), rot=(1.0, 0.0, 0.0, 0.0)),
    )
    # goal object
    goal_object_cfg: VisualizationMarkersCfg = VisualizationMarkersCfg(
        prim_path="/Visuals/goal_marker",
        markers={
            "goal": sim_utils.UsdFileCfg(
                usd_path=f"{ISAAC_NUCLEUS_DIR}/Props/Blocks/DexCube/dex_cube_instanceable.usd",
                scale=(0.67, 0.67, 0.67),
            )
        },
    )
    # scene
    scene: InteractiveSceneCfg = InteractiveSceneCfg(num_envs=8192, env_spacing=0.75, replicate_physics=True)


    # reset
    reset_position_noise = 0.01  # range of position at reset
    reset_dof_pos_noise = 0.2  # range of dof pos at reset
    reset_dof_vel_noise = 0.0  # range of dof vel at reset
    # reward scales
    dist_reward_scale = -10.0
    rot_reward_scale = 1.0
    rot_eps = 0.1
    action_penalty_scale = -0.001
    reach_goal_bonus = 250
    fall_penalty = -50
    fall_dist = 0.24
    vel_obs_scale = 0.2
    success_tolerance = 0.4
    max_consecutive_success = 0
    av_factor = 0.1
    act_moving_average = 0.3
    force_torque_obs_scale = 10.0