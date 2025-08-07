import sys
import rclpy # Import ROS2 Python client library
from rclpy.node import Node # Import Node class from rclpy to create a ROS2 node
from rclpy.qos import QoSProfile, ReliabilityPolicy, HistoryPolicy, DurabilityPolicy # Import ROS2 QoS policy modules

from px4_msgs.msg import OffboardControlMode, VehicleCommand #Import basic PX4 ROS2-API messages for switching to offboard mode
from px4_msgs.msg import TrajectorySetpoint, ActuatorMotors, VehicleThrustSetpoint, VehicleTorqueSetpoint, VehicleRatesSetpoint # Msgs for sending setpoints to the vehicle in various offboard modes
from px4_msgs.msg import VehicleOdometry, VehicleStatus, RcChannels #Import PX4 ROS2-API messages for receiving vehicle state information

import time
import traceback
from typing import Optional

import numpy as np  
import math as m
from scipy.spatial.transform import Rotation as R

from testtt.utilities import test_function
from testtt.Logger import Logger
from testtt.jax_nr import NR_tracker_original#, dynamics, predict_output, get_jac_pred_u, fake_tracker, NR_tracker_flat, NR_tracker_linpred
from testtt.utilities import sim_constants # Import simulation constants
from testtt.jax_mm_rta import *

import jax
import jax.numpy as jnp
import immrax as irx
import control
from functools import partial

# Some configurations
jax.config.update("jax_enable_x64", True)
def jit (*args, **kwargs): # A simple wrapper for JAX's jit function to set the backend device
    device = 'cpu'
    kwargs.setdefault('backend', device)
    return jax.jit(*args, **kwargs)

GP_instantiation_values = jnp.array([[-2, 0.0], #make the second column all zeros
                                    [0, 0.0],
                                    [2, 0.0],
                                    [4, 0.0],
                                    [6, 0.0],
                                    [8, 0.0],
                                    [10, 0.0],
                                    [12, 0.0]]) # at heights of y in the first column, disturbance to the values in the second column
# add a time dimension at t=0 to the GP instantiation values for TVGPR instantiation
actual_disturbance_GP = TVGPR(jnp.hstack((jnp.zeros((GP_instantiation_values.shape[0], 1)), GP_instantiation_values)), 
                                       sigma_f = 5.0, 
                                       l=2.0, 
                                       sigma_n = 0.01,
                                       epsilon=0.1,
                                       discrete=False
                                       )





class OffboardControl(Node):
    def __init__(self, sim: bool) -> None:
        super().__init__('px4_rta_mm_gpr_node')
        test_function()
        # Initialize essential variables
        self.sim: bool = sim
        self.GRAVITY: float = 9.806 # m/s^2, gravitational acceleration

        if self.sim:
            print("Using simulator constants and functions")
            from testtt.utilities import sim_constants # Import simulation constants
            self.MASS = sim_constants.MASS
            self.THRUST_CONSTANT = sim_constants.THRUST_CONSTANT #x500 gazebo simulation motor thrust constant
            self.MOTOR_VELOCITY_ARMED = sim_constants.MOTOR_VELOCITY_ARMED #x500 gazebo motor velocity when armed
            self.MAX_ROTOR_SPEED = sim_constants.MAX_ROTOR_SPEED #x500 gazebo simulation max rotor speed
            self.MOTOR_INPUT_SCALING = sim_constants.MOTOR_INPUT_SCALING #x500 gazebo simulation motor input scaling

        elif not self.sim:
            print("Using hardware constants and functions")
            #TODO: do the hardware version of the above here
            try:
                from testtt.utilities import hardware_constants
                self.MASS = hardware_constants.MASS
            except ImportError:
                raise ImportError("Hardware not implemented yet.")


        # Logging related variables
        self.time_log = []
        self.x_log, self.y_log, self.z_log, self.yaw_log = [], [], [], []
        self.ctrl_comp_time_log = []
        # self.m0_log, self.m1_log, self.m2_log, self.m3_log = [], [], [], [] # direct actuator control logs
        # self.f_log, self.M_log = [], [] # force and moment logs
        self.throttle_log, self.roll_rate_log, self.pitch_rate_log, self.yaw_rate_log = [], [], [], [] # throttle and rate logs
        self.metadata = np.array(['Sim' if self.sim else 'Hardware',
                                ])

##########################################################################################
        # Configure QoS profile for publishing and subscribing
        qos_profile = QoSProfile(
            reliability=ReliabilityPolicy.BEST_EFFORT,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            history=HistoryPolicy.KEEP_LAST,
            depth=1
        )

        # Create publishers
        self.offboard_control_mode_publisher = self.create_publisher(
            OffboardControlMode, '/fmu/in/offboard_control_mode', qos_profile)
        self.vehicle_command_publisher = self.create_publisher(
            VehicleCommand, '/fmu/in/vehicle_command', qos_profile)
        self.trajectory_setpoint_publisher = self.create_publisher(
            TrajectorySetpoint, '/fmu/in/trajectory_setpoint', qos_profile)
        self.actuator_motors_publisher = self.create_publisher(
            ActuatorMotors, '/fmu/in/actuator_motors', qos_profile)
        self.vehicle_thrust_setpoint_publisher = self.create_publisher(
            VehicleThrustSetpoint, '/fmu/in/vehicle_thrust_setpoint', qos_profile)
        self.vehicle_torque_setpoint_publisher = self.create_publisher(
            VehicleTorqueSetpoint, '/fmu/in/vehicle_torque_setpoint', qos_profile)
        self.vehicle_rates_setpoint_publisher = self.create_publisher(
            VehicleRatesSetpoint, '/fmu/in/vehicle_rates_setpoint', qos_profile)

        # Create subscribers
        self.vehicle_odometry_subscriber = self.create_subscription( #subscribes to odometry data (position, velocity, attitude)
            VehicleOdometry, '/fmu/out/vehicle_odometry', self.vehicle_odometry_subscriber_callback, qos_profile)
        self.vehicle_status_subscriber = self.create_subscription(
            VehicleStatus, '/fmu/out/vehicle_status', self.vehicle_status_subscriber_callback, qos_profile)
            
        self.offboard_mode_rc_switch_on: bool = True if self.sim else False   # RC switch related variables and subscriber
        self.MODE_CHANNEL: int = 5 # Channel for RC switch to control offboard mode (-1: position, 0: offboard, 1: land)
        self.rc_channels_subscriber = self.create_subscription( #subscribes to rc_channels topic for software "killswitch" for position v offboard v land mode
            RcChannels, '/fmu/out/rc_channels', self.rc_channel_subscriber_callback, qos_profile
        )
        
        # MoCap related variables
        self.mocap_initialized: bool = False
        self.full_rotations: int = 0
        self.max_yaw_stray = 15 * np.pi / 180

        # PX4 variables
        self.offboard_heartbeat_counter: int = 0
        self.vehicle_status = VehicleStatus()
        # self.takeoff_height = -5.0

        # Callback function time constants
        self.heartbeat_period: float = 0.1 # (s) We want 10Hz for offboard heartbeat signal
        self.control_period: float = 0.01 # (s) We want 1000Hz for direct control algorithm
        self.traj_idx = 0 # Index for trajectory setpoint

        # Timers for my callback functions
        self.offboard_timer = self.create_timer(self.heartbeat_period,
                                                self.offboard_heartbeat_signal_callback) #Offboard 'heartbeat' signal should be sent at 10Hz
        self.control_timer = self.create_timer(self.control_period,
                                               self.control_algorithm_callback) #My control algorithm needs to execute at >= 100Hz
        self.rollout_timer = self.create_timer(self.control_period,
                                               self.rollout_callback) #My rollout function needs to execute at >= 100Hz

        # Initialize newton-raphson algorithm parameters
        self.last_input: jnp.ndarray = jnp.array([self.MASS * self.GRAVITY, 0.01, 0.01, 0.01]) # last input to the controller
        self.hover_input_planar: jnp.ndarray = jnp.array([self.MASS * self.GRAVITY, 0.]) # hover input to the controller
        self.odom_counter = 0
        self.T_LOOKAHEAD: float = 0.8 # (s) lookahead time for the controller in seconds
        self.T_LOOKAHEAD_PRED_STEP: float = 0.1 # (s) we do state prediction for T_LOOKAHEAD seconds ahead in intervals of T_LOOKAHEAD_PRED_STEP seconds
        self.INTEGRATION_TIME: float = self.control_period # integration time constant for the controller in seconds

        # NR tracker JIT-compile
        init_state = jnp.array([0.1, 0.1, 0.1, 0.02, 0.03, 0.02, 0.01, 0.01, 0.03]) # Initial state vector for testing
        init_input = self.last_input  # Initial input vector for testing
        init_noise = jnp.array([0.01]) # [w1= unkown horizontal wind disturbance]
        init_ref = jnp.array([0.0, 0.0, -3.0, 0.0])  # Initial reference vector for testing
        NR_tracker_original(init_state, init_input, init_ref, self.T_LOOKAHEAD, self.T_LOOKAHEAD_PRED_STEP, self.INTEGRATION_TIME, self.MASS) # JIT-compile the NR tracker function


        # Initialize rta_mm_gpr variables
        n_obs = 9
        x0 = jnp.array(init_state[0:5])  # Initial state vector for testing
        self.obs = jnp.tile(jnp.array([[0, x0[1], get_gp_mean(actual_disturbance_GP, 0.0, x0)[0]]]),(n_obs,1))

        self.x_pert = 1e-4 * jnp.array([1., 1., 1., 1., 1.])
        ix0 = irx.icentpert(x0, self.x_pert)
        u0 = jnp.array(init_input[0:2])  # Initial input vector for testing
        w0 = jnp.array(init_noise)  # Initial noise vector for testing
        print(f"{x0=}, {ix0=}, {ix0.shape=}")

        # JIT-compile the linearization system function and do LQR control for ref and feedback
        time0 = time.time()
        A, B = jitted_linearize_system(quad_sys_planar, x0, u0, w0)
        print(f"{A=},{B=}")
        print(f"Time taken for linearization {time.time() - time0}\n")

        time0 = time.time()
        A, B = jitted_linearize_system(quad_sys_planar, x0, u0, w0)
        print(f"{A=},{B=}")
        print(f"Time taken for linearization after jit {time.time() - time0}")

        # Do LQR
        time0 = time.time()
        K_reference, P, _ = control.lqr(A, B, Q_ref_planar, R_ref_planar)
        K_feedback, P, _ = control.lqr(A, B, Q_planar, R_planar)
        print(f"{K_feedback= }, {K_reference=}, Time taken for LQR: {time.time() - time0}")

        time0 = time.time()
        K_reference, P, _ = control.lqr(A, B, Q_ref_planar, R_ref_planar)
        K_feedback, P, _ = control.lqr(A, B, Q_planar, R_planar)
        print(f"{K_feedback= }, {K_reference=}, Time taken for LQR part 2: {time.time() - time0}")

        # Set up rollout parameters
        t0 = 0.0  # Initial time
        self.tube_timestep = 0.01  # Time step
        self.tube_horizon = 30.0   # Reachable tube horizon
        self.sys_mjacM = irx.mjacM(quad_sys_planar.f) # create a mixed Jacobian inclusion matrix for the system dynamics function
        self.perm = irx.Permutation((0, 1, 2, 3, 4, 5, 6, 7, 8)) # create a permutation for the inclusion system calculation

        # JIT-compile rollout and collection_id_jax functions
        time0 = time.time()
        reachable_tube, rollout_ref, rollout_feedfwd_input = jitted_rollout(0.0, ix0, x0, K_feedback, K_reference, self.obs, self.tube_horizon, self.tube_timestep, self.perm, self.sys_mjacM) # JIT compile the rollout function for performance
        reachable_tube.block_until_ready()
        rollout_ref.block_until_ready()
        rollout_feedfwd_input.block_until_ready()
        print(f"Time taken for rollout: {time.time() - time0} seconds")

        time0 = time.time()
        reachable_tube, rollout_ref, rollout_feedfwd_input = jitted_rollout(0.0, ix0, x0, K_feedback, K_reference, self.obs, self.tube_horizon, self.tube_timestep, self.perm, self.sys_mjacM)
        reachable_tube.block_until_ready()
        rollout_ref.block_until_ready()
        rollout_feedfwd_input.block_until_ready()
        print(f"Time taken for rollout after jit : {time.time() - time0} seconds")

        time0 = time.time()
        violation_safety_time_idx = collection_id_jax(rollout_ref, reachable_tube)
        print(f"Collection ID: {violation_safety_time_idx}, Time taken for collection_id: {time.time() - time0}")

        time0 = time.time()
        violation_safety_time_idx = collection_id_jax(rollout_ref, reachable_tube)
        print(f"Collection ID: {violation_safety_time_idx}, Time taken for collection_id after jit: {time.time()-time0}")

        time0 = time.time()
        applied_u = u_applied(x0, x0, u0, K_feedback)
        print(f"Applied u: {applied_u} Time taken for u_applied: {time.time() - time0} seconds")

        time0 = time.time()
        applied_u = u_applied(x0, x0, u0, K_feedback)
        print(f"Applied u: {applied_u} Time taken for u_applied after jit: {time.time() - time0} seconds")

        self.last_lqr_update_time: float = 0.0  # Initialize last LQR update time
        self.first_LQR: bool = True  # Flag to indicate if this is the first LQR update
        self.collection_time: float = 0.0  # Time at which the collection starts

        # Time variables
        self.T0 = time.time() # (s) initial time of program
        self.time_from_start = time.time() - self.T0 # (s) time from start of program 
        self.begin_actuator_control = 15 # (s) time after which we start sending actuator control commands
        self.land_time = self.begin_actuator_control + 20 # (s) time after which we start sending landing commands
        if self.sim:
            self.max_height = -12.5
        else:
            raise NotImplementedError("Hardware not implemented yet.")
        # self.reachable_tube, self.rollout_ref, self.rollout_feedfwd_input = reachable_tube, rollout_ref, rollout_feedfwd_input
        # exit(0)

    

    def rc_channel_subscriber_callback(self, rc_channels):
        """Callback function for RC Channels to create a software 'killswitch' depending on our flight mode channel (position vs offboard vs land mode)"""
        print('In RC Channel Callback')
        flight_mode = rc_channels.channels[self.MODE_CHANNEL-1] # +1 is offboard everything else is not offboard
        self.offboard_mode_rc_switch_on: bool = True if flight_mode >= 0.75 else False


    def adjust_yaw(self, yaw: float) -> float:
        """Adjust yaw angle to account for full rotations and return the adjusted yaw.

        This function keeps track of the number of full rotations both clockwise and counterclockwise, and adjusts the yaw angle accordingly so that it reflects the absolute angle in radians. It ensures that the yaw angle is not wrapped around to the range of -pi to pi, but instead accumulates the full rotations.
        This is particularly useful for applications where the absolute orientation of the vehicle is important, such as in control algorithms or navigation systems.
        The function also initializes the first yaw value and keeps track of the previous yaw value to determine if a full rotation has occurred.

        Args:
            yaw (float): The yaw angle in radians from the motion capture system after being converted from quaternion to euler angles.

        Returns:
            psi (float): The adjusted yaw angle in radians, accounting for full rotations.
        """        
        mocap_psi = yaw
        psi = None

        if not self.mocap_initialized:
            self.mocap_initialized = True
            self.prev_mocap_psi = mocap_psi
            psi = mocap_psi
            return psi

        # MoCap angles are from -pi to pi, whereas the angle state variable should be an absolute angle (i.e. no modulus wrt 2*pi)
        #   so we correct for this discrepancy here by keeping track of the number of full rotations.
        if self.prev_mocap_psi > np.pi*0.9 and mocap_psi < -np.pi*0.9: 
            self.full_rotations += 1  # Crossed 180deg in the CCW direction from +ve to -ve rad value so we add 2pi to keep it the equivalent positive value
        elif self.prev_mocap_psi < -np.pi*0.9 and mocap_psi > np.pi*0.9:
            self.full_rotations -= 1 # Crossed 180deg in the CW direction from -ve to +ve rad value so we subtract 2pi to keep it the equivalent negative value

        psi = mocap_psi + 2*np.pi * self.full_rotations
        self.prev_mocap_psi = mocap_psi
        
        return psi


    def vehicle_odometry_subscriber_callback(self, msg) -> None:
        """Callback function for vehicle odometry topic subscriber."""
        print("==" * 30)
        self.x = msg.position[0]
        self.y = msg.position[1]
        self.z = msg.position[2] #+ (2.25 * self.sim) # Adjust z for simulation, new gazebo model has ground level at around -1.39m 

        self.vx = msg.velocity[0]
        self.vy = msg.velocity[1]
        self.vz = msg.velocity[2]


        self.roll, self.pitch, yaw = R.from_quat(msg.q, scalar_first=True).as_euler('xyz', degrees=False)
        self.yaw = self.adjust_yaw(yaw)  # Adjust yaw to account for full rotations
        r_final = R.from_euler('xyz', [self.roll, self.pitch, self.yaw], degrees=False)         # Final rotation object
        self.rotation_object = r_final  # Store the final rotation object for further use

        self.p = msg.angular_velocity[0]
        self.q = msg.angular_velocity[1]
        self.r = msg.angular_velocity[2]

        self.full_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw, self.p, self.q, self.r])
        self.nr_state_vector = np.array([self.x, self.y, self.z, self.vx, self.vy, self.vz, self.roll, self.pitch, self.yaw])
        self.flat_state_vector = np.array([self.x, self.y, self.z, self.yaw, self.vx, self.vy, self.vz, 0., 0., 0., 0., 0.])
        self.rta_mm_gpr_state_vector_planar = np.array([self.y, self.z, self.vy, self.vz, self.roll])# px, py, h, v, theta = x
        self.output_vector = np.array([self.x, self.y, self.z, self.yaw])
        self.position = np.array([self.x, self.y, self.z])
        self.velocity = np.array([self.vx, self.vy, self.vz])
        self.quat = self.rotation_object.as_quat()  # Quaternion representation (xyzw)
        self.ROT = self.rotation_object.as_matrix()
        self.omega = np.array([self.p, self.q, self.r])

        print(f"in odom, flat output: {self.output_vector}")
        if self.first_LQR:
            self.odom_counter += 1
            t00 = time.time()
            noise = jnp.array([0.0])  # Small noise to avoid singularity in linearization
            # t0 = time.time()
            A, B = jitted_linearize_system(quad_sys_planar, self.rta_mm_gpr_state_vector_planar, self.hover_input_planar, noise)
            A, B = np.array(A), np.array(B)
            # print(f"Time to linearize system: {time.time() - t0} seconds")

            # t0 = time.time()
            K, P, _ = control.lqr(A, B, Q_planar, R_planar)
            self.feedback_K = 1 * K
            # print(f"Time taken for LQR synthesis for K_feedback: {time.time() - t0} seconds")

            # t0 = time.time()
            K, P, _ = control.lqr(A, B, Q_ref_planar, R_ref_planar)  # Compute the LQR gain matrix
            self.reference_K = 1 * K  # Store the reference gain matrix
            # print(f"Time taken for LQR synthesis for K_reference: {time.time() - t0} seconds")

            
            self.last_lqr_update_time = time.time() - self.T0  # Set the last LQR update time to the current time
            print(f"Odom: time taken for entire LQR update: {time.time() - t00} seconds")

            # if self.odom_counter > 5:
            #     exit(0)
        ODOMETRY_DEBUG_PRINT = False
        if ODOMETRY_DEBUG_PRINT:
            # print(f"{self.full_state_vector=}")
            print(f"{self.nr_state_vector=}")
            # print(f"{self.flat_state_vector=}")
            print(f"{self.output_vector=}")
            print(f"{self.roll = }, {self.pitch = }, {self.yaw = }(rads)")
            # print(f"{self.rotation_object.as_euler('xyz', degrees=True) = } (degrees)")
            # print(f"{self.ROT = }")

    def rollout_callback(self):
        """Callback function for the rollout timer."""
        print(f"\nIn rollout callback at time: ", time.time() - self.T0)
        try:
            self.time_from_start = time.time() - self.T0
            t00 = time.time()  # Start time for rollout computation
            thresh = 1.0
            current_time = self.time_from_start
            current_state = self.rta_mm_gpr_state_vector_planar
            current_state_interval = irx.icentpert(current_state, self.x_pert)
            print(f"{current_time= }, {self.collection_time= }")

            if current_time >= self.collection_time:
                print("Unsafe region begins now. Recomputing reachable tube and reference trajectory.")
                t0 = time.time()  # Reset time for rollout computation
                self.reachable_tube, self.rollout_ref, self.rollout_feedfwd_input = jitted_rollout(
                    current_time, current_state_interval, current_state, self.feedback_K, self.reference_K, self.obs, self.tube_horizon, self.tube_timestep, self.perm, self.sys_mjacM
                )
                self.reachable_tube.block_until_ready()
                self.rollout_ref.block_until_ready()
                self.rollout_feedfwd_input.block_until_ready()
                # print(f"Time taken by rollout: {time.time() - t0:.4f} seconds")

                # t0 = time.time()  # Reset time for collection index computation
                t_index = collection_id_jax(self.rollout_ref, self.reachable_tube, thresh)
                t_index = int(t_index)
                # print(f"Time taken for collection index computation: {time.time() - t0:.4f} seconds")

                safety_horizon = t_index * self.tube_timestep
                self.collection_time = current_time + safety_horizon  # Update the collection time based on the current time and index
                print(f"{self.collection_time=}\n{safety_horizon=}")

                self.traj_idx = 0
            else:
                print("You're safe!")
            print(f"Time taken for whole rollout process: {time.time() - t00:.4f} seconds")


        except AttributeError as e:
            print("Ignoring missing attribute:", e)
            return
        except Exception as e:
            raise  # Re-raise all other types of exceptions            


    def vehicle_status_subscriber_callback(self, vehicle_status) -> None:
        """Callback function for vehicle_status topic subscriber."""
        self.vehicle_status = vehicle_status

    def arm(self) -> None:
        """Send an arm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=1.0)
        self.get_logger().info('Arm command sent')

    def disarm(self) -> None:
        """Send a disarm command to the vehicle."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_COMPONENT_ARM_DISARM, param1=0.0)
        self.get_logger().info('Disarm command sent')

    def engage_offboard_mode(self) -> None:
        """Switch to offboard mode."""
        self.publish_vehicle_command(
            VehicleCommand.VEHICLE_CMD_DO_SET_MODE, param1=1.0, param2=6.0)
        self.get_logger().info("Switching to offboard mode")

    def land(self) -> None:
        """Switch to land mode."""
        self.publish_vehicle_command(VehicleCommand.VEHICLE_CMD_NAV_LAND)
        self.get_logger().info("Switching to land mode")

    def publish_offboard_control_heartbeat_signal_position(self) -> None:
        """Publish the offboard control mode heartbeat for position-only setpoints."""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = True
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self.offboard_control_mode_publisher.publish(msg)
        # self.get_logger().info("Switching to position control mode")

    def publish_offboard_control_heartbeat_signal_actuators(self) -> None:
        """Publish the offboard control mode heartbeat for actuator-only setpoints."""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = False
        msg.direct_actuator = True
        self.offboard_control_mode_publisher.publish(msg)
        # self.get_logger().info("Switching to actuator control mode")

    def publish_offboard_control_heartbeat_signal_thrust_moment(self) -> None:
        """Publish the offboard control mode heartbeat for actuator-only setpoints."""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = False
        msg.thrust_and_torque = True
        msg.direct_actuator = False
        self.offboard_control_mode_publisher.publish(msg)
        # self.get_logger().info("Switching to force and moment control mode")

    def publish_offboard_control_heartbeat_signal_body_rate(self) -> None:
        """Publish the offboard control mode heartbeat for body rate setpoints."""
        msg = OffboardControlMode()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.position = False
        msg.velocity = False
        msg.acceleration = False
        msg.attitude = False
        msg.body_rate = True
        msg.thrust_and_torque = False
        msg.direct_actuator = False
        self.offboard_control_mode_publisher.publish(msg)
        # self.get_logger().info("Switching to body rate control mode")
    

    def publish_position_setpoint(self, x: float = 0.0, y: float = 0.0, z: float = -3.0, yaw: float = 0.0) -> None:
        """Publish the trajectory setpoint.

        Args:
            x (float, optional): Desired x position in meters. Defaults to 0.0.
            y (float, optional): Desired y position in meters_. Defaults to 0.0.
            z (float, optional): Desired z position in meters. Defaults to -3.0.
            yaw (float, optional): Desired yaw position in radians. Defaults to 0.0.

        Returns:
            None

        Raises:
            TypeError: If x, y, z, or yaw are not of type float.
        Raises:
            ValueError: If x, y, z are not within the expected range.
        """
        for name, val in zip(("x","y","z","yaw"), (x,y,z,yaw)):
            if not isinstance(val, float):
                raise TypeError(
                                f"\n{'=' * 60}"
                                f"\nInvalid input type for {name}\n"
                                f"Expected float\n"
                                f"Received {type(val).__name__}\n"
                                f"{'=' * 60}"
                                )
               
        # if not (-2.0 <= x <= 2.0) or not (-2.0 <= y <= 2.0) or not (-3.0 <= z <= -0.2):
        #     raise ValueError("x must be between -2.0 and 2.0, y must be between -2.0 and 2.0, z must be between -0.2 and -3.0")
        
        msg = TrajectorySetpoint()
        msg.position = [x, y, z] # position in meters
        msg.yaw = yaw # yaw in radians
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.trajectory_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing position setpoints {[x, y, z, yaw]}")


    def publish_actuator_setpoint(self, m0: float = 0.0, m1: float = 0.0, m2: float = 0.0, m3: float = 0.0) -> None:

        """Publish the actuator setpoint.

        Args:
            m0 (float): Desired throttle for motor 0.
            m1 (float): Desired throttle for motor 1.
            m2 (float): Desired throttle for motor 2.
            m3 (float): Desired throttle for motor 3.

        Returns:
            None

        Raises:
            ValueError: If m0, m1, m2, or m3 are not within 0-1.
        """
        for name, val in zip(("m0", "m1", "m2", "m3"), (m0, m1, m2, m3)):
            if not (0 <= val <= 1):
                raise ValueError(
                                f"\n{'=' * 60}"
                                f"\nInvalid input for {name}\n"
                                f"Expected value between 0 and 1\n"
                                f"Received {val}\n"
                                f"{'=' * 60}"
                                )        

        msg = ActuatorMotors()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        # 0th motor seems to be back left motor when we're aligned with quad's 0 yaw position
        # 1st motor seems to be front right motor when we're aligned with quad's 0 yaw position
        # 2nd motor seems to be back right motor when we're aligned with quad's 0 yaw position
        # 3rd motor seems to be front left motor when we're aligned with quad's 0 yaw position
        msg.control = [m0, m1, m2, m3] + 8 * [0.0]
        self.actuator_motors_publisher.publish(msg)
        self.get_logger().info(f"Publishing actuator setpoints: {msg.control}")

    def publish_force_moment_setpoint(self, f: float = 0.0, M: list[float] = [0.0, 0.0, 0.0]) -> None:
        """Publish the force and moment setpoint.
        
        Args:
            f (float): Desired force in Newtons.
            M (float): Desired moment in Newtons.
        
        Returns:
            None
                    
        Raises:
            ValueError: If f is not within 0-1 or M is not within -1 to 1.
        """
        if not (0 <= f <= 1):
            raise ValueError(
                            f"\n{'=' * 60}"
                            f"\nInvalid input for force\n"
                            f"Expected value between 0 and 1\n"
                            f"Received {f}\n"
                            f"{'=' * 60}"
                            )
        if not all(-1 <= m <= 1 for m in M):
            raise ValueError(
                            f"\n{'=' * 60}"
                            f"\nInvalid input for moment\n"
                            f"Expected values between -1 and 1 for each component\n"
                            f"Received {M}\n"
                            f"{'=' * 60}"
                            )
        
        msg1 = VehicleThrustSetpoint()
        msg1.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg1.xyz = [0.,0.,-1.0]  # Thrust in Newtons
        self.vehicle_thrust_setpoint_publisher.publish(msg1)
        self.get_logger().info(f"Publishing thrust setpoint: {msg1.xyz}")

        msg2 = VehicleTorqueSetpoint()
        msg2.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg2.xyz = [0.0, 0.0, 0.0]  # Torque in Newton-meters
        self.vehicle_torque_setpoint_publisher.publish(msg2)
        self.get_logger().info(f"Publishing torque setpoint: {msg2.xyz}")

    def publish_body_rate_setpoint(self, throttle: float = 0.0, p: float = 0.0, q: float = 0.0, r: float = 0.0) -> None:
        """Publish the body rate setpoint.
        
        Args:
            p (float): Desired roll rate in radians per second.
            q (float): Desired pitch rate in radians per second.
            r (float): Desired yaw rate in radians per second.
            throttle (float): Desired throttle in normalized from [-1,1] in NED body frame

        Returns:
            None
        
        Raises:
            ValueError: If p, q, r, or throttle are not within expected ranges.
        """

        
        # print(f"Publishing body rate setpoint: roll={p}, pitch={q}, yaw={r}, throttle={throttle}")
        msg = VehicleRatesSetpoint()
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        msg.roll = p
        msg.pitch = q
        msg.yaw = r
        msg.thrust_body[0] = 0.0
        msg.thrust_body[1] = 0.0
        msg.thrust_body[2] = -1 * float(throttle)
        self.vehicle_rates_setpoint_publisher.publish(msg)
        self.get_logger().info(f"Publishing body rate setpoint: roll={p}, pitch={q}, yaw={r}, thrust_body={throttle}")

        # exit(0)

    def publish_vehicle_command(self, command, **params) -> None:
        """Publish a vehicle command."""
        msg = VehicleCommand()
        msg.command = command
        msg.param1 = params.get("param1", 0.0)
        msg.param2 = params.get("param2", 0.0)
        msg.param3 = params.get("param3", 0.0)
        msg.param4 = params.get("param4", 0.0)
        msg.param5 = params.get("param5", 0.0)
        msg.param6 = params.get("param6", 0.0)
        msg.param7 = params.get("param7", 0.0)
        msg.target_system = 1
        msg.target_component = 1
        msg.source_system = 1
        msg.source_component = 1
        msg.from_external = True
        msg.timestamp = int(self.get_clock().now().nanoseconds / 1000)
        self.vehicle_command_publisher.publish(msg)

    def offboard_heartbeat_signal_callback(self) -> None:
        """Callback function for the heartbeat signals that maintains flight controller in offboard mode and switches between offboard flight modes."""
        self.time_from_start = time.time() - self.T0
        print(f"In offboard callback at {self.time_from_start:.2f} seconds")

        if not self.offboard_mode_rc_switch_on: #integration of RC 'killswitch' for offboard to send heartbeat signal, engage offboard, and arm
            print(f"Offboard Callback: RC Flight Mode Channel {self.MODE_CHANNEL} Switch Not Set to Offboard (-1: position, 0: offboard, 1: land) ")
            self.offboard_heartbeat_counter = 0
            return

        if self.time_from_start <= self.begin_actuator_control:
            self.publish_offboard_control_heartbeat_signal_position()
        elif self.time_from_start <= self.land_time:  
            self.publish_offboard_control_heartbeat_signal_body_rate()
        elif self.time_from_start > self.land_time:
            self.publish_offboard_control_heartbeat_signal_position()
        else:
            raise ValueError("Unexpected time_from_start value")

        if self.offboard_heartbeat_counter <= 10:
            if self.offboard_heartbeat_counter == 10:
                self.engage_offboard_mode()
                self.arm()
            self.offboard_heartbeat_counter += 1

        
        
    def control_algorithm_callback(self) -> None:
        """Callback function to handle control algorithm once in offboard mode."""
        self.time_from_start = time.time() - self.T0
        if not (self.offboard_mode_rc_switch_on and (self.vehicle_status.nav_state == VehicleStatus.NAVIGATION_STATE_OFFBOARD) ):
            print(f"Not in offboard mode.\n"
                  f"Current state: {self.vehicle_status.nav_state}\n"
                  f"Expected offboard state: {VehicleStatus.NAVIGATION_STATE_OFFBOARD}\n"
                  f"Offboard RC switch status: {self.offboard_mode_rc_switch_on}")
            return

        if self.time_from_start <= self.begin_actuator_control:
            self.publish_position_setpoint(0., 4.0, self.max_height, 0.0)

        elif self.time_from_start <= self.land_time:
            # f, M = self.control_administrator()
            # self.publish_force_moment_setpoint(f, M)
            self.control_administrator()

        elif self.time_from_start > self.land_time or (abs(self.z) <= 1.5 and self.time_from_start > 20):
            print("Landing...")
            self.publish_position_setpoint(0.0, 0.0, -0.83, 0.0)
            if abs(self.x) < 0.2 and abs(self.y) < 0.2 and abs(self.z) <= 0.85:
                print("Vehicle is close to the ground, preparing to land.")
                self.land()                    
                exit(0)

        else:
            raise ValueError("Unexpected time_from_start value")

    def get_ref(self, time_from_start: float) -> tuple[jnp.ndarray, jnp.ndarray]:
        """Get the reference trajectory for the LQR and NR tracker.

        Args:
            time_from_start (float): Time from the start of the program in seconds.

        Returns:
            tuple: Reference trajectory for LQR and NR tracker.
        """
        # Define your reference trajectories here based on time_from_start
        x_des = 0.0
        y_des = 0.0
        z_des = np.clip(self.max_height + 0.1 * time_from_start, self.max_height, -0.55)  # Clip z_des to be between self.max_height and -0.55

        vx_des = 0.0
        vy_des = 0.0
        vz_des = 0.0

        roll_des = 0.0
        pitch_des = 0.0
        yaw_des = 0.0

        ref_lqr_planar = jnp.array([y_des, z_des, vy_des, vz_des, roll_des])  # Reference position setpoint for planar LQR tracker (y, z, vy, vz, roll)
        ref_lqr_3D = jnp.array([x_des, y_des, z_des, vx_des, vy_des, vz_des, roll_des, pitch_des, yaw_des])
        ref_nr = jnp.array([x_des, y_des, z_des, yaw_des])  # Reference position setpoint for NR tracker (x, y, z, yaw)
        return ref_lqr_planar, ref_lqr_3D, ref_nr  

    def control_administrator(self) -> None:
        self.time_from_start = time.time() - self.T0
        print(f"\nIn control administrator at {self.time_from_start:.2f} seconds")
        ref_lqr_planar, ref_lqr_3D, ref_nr = self.get_ref(self.time_from_start)
        ctrl_T0 = time.time()

        NR_new_u, _ = NR_tracker_original(self.nr_state_vector, self.last_input, ref_nr, self.T_LOOKAHEAD, self.T_LOOKAHEAD_PRED_STEP, self.INTEGRATION_TIME, self.MASS)
        print(f"Time taken for NR tracker: {time.time() - ctrl_T0:.4f} seconds")
        # LQR_new_u_planar = self.lqr_administrator_planar(ref_lqr_planar, self.rta_mm_gpr_state_vector_planar, self.last_input[0:2], self.output_vector)  # Compute LQR control input for planar system
        # LQR_new_u_3D = self.lqr_administrator_3D(ref_lqr_3D, self.nr_state_vector, self.last_input, self.output_vector)  # Compute LQR control input
        rta_new_u_planar = self.rta_mm_gpr_administrator(ref_lqr_planar, self.rta_mm_gpr_state_vector_planar, self.last_input[0:2], self.output_vector)  # Compute RTA-MM GPR control input for planar system

        control_comp_time = time.time() - ctrl_T0 # Time taken for control computation
        print(f"\nEntire control Computation Time: {control_comp_time:.4f} seconds, Good for {1/control_comp_time:.2f}Hz control loop")


        print(f"{NR_new_u =}")
        # print(f"{LQR_new_u_planar =}")
        # print(f"{LQR_new_u_3D =}")  
        print(f"{rta_new_u_planar =}")

        # new_u = np.hstack([LQR_new_u_planar, NR_new_u[2:]])  # New control input from the LQR tracker
        new_u = np.hstack([rta_new_u_planar, NR_new_u[2:]])  # New control input from the RTA-MM GPR tracker
        # new_u = np.hstack([LQR_new_u, NR_new_u[2:]])  # New control input from the NR tracker
        # new_u = LQR_new_u  # Use the LQR control input directly
        print(f"{new_u = }")
        # exit(0)
        # if self.traj_idx > 11:
        #     print(f"Trajectory index {self.traj_idx} exceeded limit, stopping control.")
        #     exit(0)

        self.last_input = new_u  # Update the last input for the next iteration
        new_force = new_u[0]
        new_throttle = float(self.get_throttle_command_from_force(new_force))
        new_roll_rate = float(new_u[1])  # Convert jax.numpy array to float
        new_pitch_rate = float(new_u[2])  # Convert jax.numpy array to float
        new_yaw_rate = float(new_u[3])    # Convert jax.numpy array to float
        self.publish_body_rate_setpoint(new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate)
        # exit(0)

        # Log the states, inputs, and reference trajectories for data analysis
        state_input_ref_log_info = [self.time_from_start,
                                    float(self.x), float(self.y), float(self.z), float(self.yaw),
                                    control_comp_time,
                                    new_throttle, new_roll_rate, new_pitch_rate, new_yaw_rate,
                                    ]
        self.update_logged_data(state_input_ref_log_info)
        print("==" * 30)

    def update_lqr_feedback(self, sys, state, input, noise):
            print(f"\nUPDATING LQR")
            t0 = time.time()
            A, B = jitted_linearize_system(sys, state, input, noise)  # Linearize the system dynamics
            K, P, _ = control.lqr(A, B, Q_planar, R_planar)
            self.feedback_K = 1 * K

            K, P, _ = control.lqr(A, B, Q_ref_planar, R_ref_planar)  # Compute the LQR gain matrix
            self.reference_K = 1 * K  # Store the reference gain matrix
            print(f"LQR Update time: {time.time()-t0}")

            self.last_lqr_update_time = self.time_from_start  # Update the last LQR update time
            if self.first_LQR:
                self.first_LQR = False  # Set first_LQR to False after the first update
            PRINT_LQR_DEBUG = False  # Set to True to print debug information for LQR
            if PRINT_LQR_DEBUG:
                print(f"\n\n{'=' * 60}")
                print(f"Linearized System Matrices:\n{A=}\n{B=}")
                print(f"LQR Gain Matrix:\n{K=}")
                print(f"Feedback Gain Matrix:\n{self.feedback_K}")
                print(f"{A.shape=}, {B.shape=}, {self.feedback_K.shape=}")
                print(f"{'=' * 60}\n\n")

    def lqr_administrator_planar(self, ref, state, input, output):
        self.time_from_start = time.time() - self.T0 # Update time from start of the program
        print(f"In LQR Administrator: {self.time_from_start=:.2f} and {self.last_lqr_update_time=:.2f}, difference: {self.time_from_start - self.last_lqr_update_time:.2f}")
        t0 = time.time()  # Start time for LQR computation

        if (self.time_from_start - self.last_lqr_update_time) >= 1.0 or self.first_LQR:  # Re-linearize and re-compute the LQR gain X seconds
            noise = jnp.array([0.0])  # Small noise to avoid singularity in linearization
            self.update_lqr_feedback(quad_sys_planar, state, input, noise)
            print(f"Time taken to update LQR {time.time - t0}")

        error = ref - state  # Compute the error between the reference and the current state
        nominal = self.feedback_K @ error  # Compute the nominal control input
        nominalG = nominal + jnp.array([sim_constants.MASS * sim_constants.GRAVITY, 0.0])  # Add gravity compensation
        clipped = jnp.clip(nominalG, ulim_planar.lower, ulim_planar.upper)  # Clip the control input to the limits

        PRINT_LQR_DEBUG = True  # Set to True to print debug information for LQR
        if PRINT_LQR_DEBUG:
            print(f"\n\n{'=' * 60}")
            print(f"Current State:\n{state=}")
            print(f"Reference:\n{ref=}")
            print(f"Error:\n{error=}")  
            print(f"Nominal Control Input: {nominal}")
            print(f"Input with Gravity Compensation: {nominalG}")
            print(f"{ulim_planar.lower=}, {ulim_planar.upper=}")
            print(f"Clipped Control Input: {clipped}")
            print(f"{'=' * 60}\n\n")

        print(f"Time taken for LQR computation: {time.time() - t0:.4f} seconds")
        return clipped

    def rta_mm_gpr_administrator(self, ref, state, input, output):
        """Run the RTA-MM administrator to compute the control inputs."""
        self.time_from_start = time.time() - self.T0 # Update time from start of the program
        print(f"\nIn RTA-MM GPR Administrator at {self.time_from_start=:.2f}")
        t0 = time.time()  # Start time for RTA-MM GPR computation

        if (self.time_from_start - self.last_lqr_update_time) >= 5.0 or self.first_LQR or abs(self.yaw) > self.max_yaw_stray:  # Re-linearize and re-compute the LQR gain X seconds
            noise = jnp.array([0.0])  # Small noise to avoid singularity in linearization
            self.update_lqr_feedback(quad_sys_planar, state, input, noise)

        current_state = self.rta_mm_gpr_state_vector_planar
        current_state_interval = irx.icentpert(current_state, self.x_pert)
        applied_input = u_applied(current_state, self.rollout_ref[self.traj_idx, :], self.rollout_feedfwd_input[self.traj_idx, :], self.feedback_K)
        self.traj_idx += 1
        print(f"{self.traj_idx=}")

        PRINT_RTA_DEBUG = True  # Set to True to print debug information for RTA-MM GPR
        if PRINT_RTA_DEBUG:
            print(f"{current_state=}")
            print(f"{current_state_interval=}")
            print(f"{self.rollout_ref[self.traj_idx, :] =}")
            print(f"{self.rollout_feedfwd_input[self.traj_idx, :] =}")
            print(f"{applied_input=}")

        print(f"Time taken for RTA-MM GPR computation: {time.time() - t0:.4f} seconds")
        return applied_input

    def get_throttle_command_from_force(self, collective_thrust): #Converts force to throttle command
        """ Convert the positive collective thrust force to a positive throttle command. """
        print(f"Conv2Throttle: collective_thrust: {collective_thrust}")
        if self.sim:
            try:
                motor_speed = m.sqrt(collective_thrust / (4.0 * self.THRUST_CONSTANT))
                throttle_command = (motor_speed - self.MOTOR_VELOCITY_ARMED) / self.MOTOR_INPUT_SCALING
                return throttle_command
            except Exception as e:
                print(f"Error in throttle conversion: {e}")
                exit(1)

        if not self.sim: # I got these parameters from a curve fit of the throttle command vs collective thrust from the hardware spec sheet
            a = 0.00705385408507030
            b = 0.0807474474438391
            c = 0.0252575818743285
            throttle_command = a*collective_thrust + b*m.sqrt(collective_thrust) + c  # equation form is a*x + b*sqrt(x) + c = y
            return throttle_command


    # def lqr_administrator_3D(self, ref, state, input, output):
    #     if self.time_from_start % 1 == 0:  # Re-linearize and re-compute the LQR gain every X seconds
    #         A, B = jitted_linearize_system(quad_sys_3D, state, input, jnp.array([0.0]))  # Linearize the system dynamics
    #         K, P, _ = control.lqr(A, B, Q_3D, R_3D)
    #         self.feedback_K = 1 * K

    #     error = ref - state  # Compute the error between the reference and the current state
    #     nominal = self.feedback_K @ error
    #     nominalG = nominal + jnp.array([sim_constants.MASS * sim_constants.GRAVITY, 0.0, 0., 0.,])  # Add gravity compensation
    #     clipped = jnp.clip(nominalG, ulim_3D.lower, ulim_3D.upper)

    #     PRINT_LQR_DEBUG = False  # Set to True to print debug information for LQR
    #     if PRINT_LQR_DEBUG:
    #         print(f"\n\n{'=' * 60}")
    #         print(f"Linearized System Matrices:\n{A=}\n{B=}")
    #         print(f"LQR Gain Matrix:\n{K=}")
    #         print(f"Feedback Gain Matrix:\n{self.feedback_K}")
    #         print(f"{A.shape=}, {B.shape=}, {self.feedback_K.shape=}")
    #         print(f"Current State:\n{state=}")
    #         print(f"Reference:\n{ref=}")
    #         print(f"Error:\n{error}")

    #         print(f"Nominal Control Input (before clipping): {nominal}")
    #         print(f"Nominal Control Input with Gravity Compensation: {nominalG}")
    #         print(f"{ulim_3D.lower=}, {ulim_3D.upper=}")
    #         print(f"Clipped Control Input: {clipped}")
    #         print(f"{'=' * 60}\n\n")

    #     return clipped

# ~~ The following functions handle the log update and data retrieval for analysis ~~
    def update_logged_data(self, data):
        print("Updating Logged Data")
        self.time_log.append(data[0])
        self.x_log.append(data[1])
        self.y_log.append(data[2])
        self.z_log.append(data[3])
        self.yaw_log.append(data[4])
        self.ctrl_comp_time_log.append(data[5])
        self.throttle_log.append(data[6])
        self.roll_rate_log.append(data[7])
        self.pitch_rate_log.append(data[8])
        self.yaw_rate_log.append(data[9])


    def get_time_log(self): return np.array(self.time_log).reshape(-1, 1)
    def get_x_log(self): return np.array(self.x_log).reshape(-1, 1)
    def get_y_log(self): return np.array(self.y_log).reshape(-1, 1)
    def get_z_log(self): return np.array(self.z_log).reshape(-1, 1)
    def get_yaw_log(self): return np.array(self.yaw_log).reshape(-1, 1)
    def get_ctrl_comp_time_log(self): return np.array(self.ctrl_comp_time_log).reshape(-1, 1)
    # def get_m0_log(self): return np.array(self.m0_log).reshape(-1, 1)
    # def get_m1_log(self): return np.array(self.m1_log).reshape(-1, 1)
    # def get_m2_log(self): return np.array(self.m2_log).reshape(-1, 1)
    # def get_m3_log(self): return np.array(self.m3_log).reshape(-1, 1)
    # def get_f_log(self): return np.array(self.f_log).reshape(-1, 1)
    # def get_M_log(self): return np.array(self.M_log).reshape(-1, 1)
    def get_throttle_log(self): return np.array(self.throttle_log).reshape(-1, 1)
    def get_roll_rate_log(self): return np.array(self.roll_rate_log).reshape(-1, 1)
    def get_pitch_rate_log(self): return np.array(self.pitch_rate_log).reshape(-1, 1)
    def get_yaw_rate_log(self): return np.array(self.yaw_rate_log).reshape(-1, 1)

    def get_metadata(self): return self.metadata.reshape(-1, 1)




# ~~ Entry point of the code -> Initializes the node and spins it. Also handles exceptions and logging ~~
def main(args=None):
    sim: Optional[bool] = None
    logger = None 
    offboard_control: Optional[OffboardControl] = None

    def shutdown_logging():
        print(
              f"Interrupt/Error/Termination Detected, Triggering Logging Process and Shutting Down Node...\n"
              f"{'=' * 65}"
              )
        if logger:
            logger.log(offboard_control)
        if offboard_control:
            offboard_control.destroy_node()
        rclpy.shutdown()

    try:

        print(              
            f"{65 * '='}\n"
            f"Initializing ROS 2 node: '{__name__}' for offboard control\n"
            f"{65 * '='}\n"
        )

        # Figure out if in simulation or hardware mode to set important variables to the appropriate values
        sim_val = int(input("Are you using the simulator? Write 1 for Sim and 0 for Hardware: "))
        if sim_val not in (0, 1):
            raise ValueError(
                            f"\n{65 * '='}\n"
                            f"Invalid input for sim: {sim_val}, expected 0 or 1\n"
                            f"{65 * '='}\n")
        sim = bool(sim_val)
        print(f"{'SIMULATION' if sim else 'HARDWARE'}")

        rclpy.init(args=args)
        offboard_control = OffboardControl(sim)

        logger = Logger([sys.argv[1]])  # Create logger with passed filename
        rclpy.spin(offboard_control)    # Spin the ROS 2 node

    except KeyboardInterrupt:
        print(
              f"\n{65 * '='}\n"
              f"Keyboard interrupt detected (Ctrl+C), exiting...\n"
              )
    except Exception as e:
        # print(f"\nError in main: {e}")
        traceback.print_exc()
    finally:
        shutdown_logging()
        print("\nNode has shut down.")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"\nError in __main__: {e}")
        traceback.print_exc()