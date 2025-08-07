import jax.numpy as jnp
from jax import jit, jacfwd, lax, jacrev, hessian

GRAVITY = 9.806
USING_CBFS = True
C = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1]])



# @jit
def dynamics(state, input, mass):
    """Quadrotor dynamics. xdot = f(x, u)."""
    x, y, z, vx, vy, vz, roll, pitch, yaw = state
    curr_thrust = input[0]
    body_rates = input[1:]
    T = jnp.array([[1, jnp.sin(roll) * jnp.tan(pitch), jnp.cos(roll) * jnp.tan(pitch)],
                    [0, jnp.cos(roll), -jnp.sin(roll)],
                    [0, jnp.sin(roll) / jnp.cos(pitch), jnp.cos(roll) / jnp.cos(pitch)]])
    curr_rolldot, curr_pitchdot, curr_yawdot = T @ body_rates

    sr = jnp.sin(roll)
    sy = jnp.sin(yaw)
    sp = jnp.sin(pitch)
    cr = jnp.cos(roll)
    cp = jnp.cos(pitch)
    cy = jnp.cos(yaw)

    vxdot = -(curr_thrust / mass) * (sr * sy + cr * cy * sp)
    vydot = -(curr_thrust / mass) * (cr * sy * sp - cy * sr)
    vzdot = GRAVITY - (curr_thrust / mass) * (cr * cp)

    return jnp.hstack([vx, vy, vz, vxdot, vydot, vzdot, curr_rolldot, curr_pitchdot, curr_yawdot])

# @jit
def fwd_euler(state, input, lookahead_step, integrations_int, mass):
    """Forward Euler integration."""
    def for_function(i, current_state):
        return current_state + dynamics(current_state, input, mass) * lookahead_step

    pred_state = lax.fori_loop(0, integrations_int, for_function, state)
    return pred_state

# @jit
def predict_state(state, u, T_lookahead, lookahead_step, mass):
    """Predict the next state at time t+T via fwd euler integration of nonlinear dynamics."""
    integrations_int = (T_lookahead / lookahead_step).astype(int)
    # print(f"{T_lookahead = }, {lookahead_step = }, {mass = }")
    # integrations_int = int(T_lookahead / lookahead_step)
    # print(f"{integrations_int = }")
    pred_state = fwd_euler(state, u, lookahead_step, integrations_int, mass)
    return pred_state

# @jit
def predict_output(state, u, T_lookahead, lookahead_step, mass):
    """Take output from the predicted states."""
    pred_state = predict_state(state, u, T_lookahead, lookahead_step, mass)
    return C @ pred_state

# @jit
def get_jac_pred_u(state, last_input, T_lookahead, lookahead_step, mass):
    """Get the jacobian of the predicted output with respect to the control input."""
    raw_val = jacfwd(predict_output, 1)(state, last_input, T_lookahead, lookahead_step, mass)
    return raw_val.reshape((4,4))

# @jit
def get_inv_jac_pred_u(state, last_input, T_lookahead, lookahead_step, mass):
    """Get the inverse of the jacobian of the predicted output with respect to the control input."""
    return jnp.linalg.pinv(get_jac_pred_u(state, last_input, T_lookahead, lookahead_step, mass).reshape((4,4)))

# @jit
def execute_cbf(current, phi, max_value, min_value, gamma):
    """Execute the control barrier function."""
    zeta_max = gamma * (max_value - current) - phi
    zeta_min = gamma * (min_value - current) - phi
    v = jnp.where(current >= 0, jnp.minimum(0, zeta_max), jnp.maximum(0, zeta_min))
    return v

# @jit
def integral_cbf(last_input, phi):
    """Integral control barrier function set-up for all inputs."""
    # Extract values from input
    curr_thrust, curr_roll_rate, curr_pitch_rate, curr_yaw_rate = last_input
    phi_thrust, phi_roll_rate, phi_pitch_rate, phi_yaw_rate = phi

    # CBF parameters
    thrust_gamma = 1.0  # CBF parameter
    thrust_max = 27.0  # max thrust (force) value
    thrust_min = 0.5  # min thrust (force) value
    v_thrust = execute_cbf(curr_thrust, phi_thrust, thrust_max, thrust_min, thrust_gamma)

    # CBF for rates
    rates_max_abs = 0.8  # max absolute value of roll, pitch, and yaw rates
    rates_max = rates_max_abs
    rates_min = -rates_max_abs
    gamma_rates = 1.0  # CBF parameter
    
    v_roll = execute_cbf(curr_roll_rate, phi_roll_rate, rates_max, rates_min, gamma_rates)
    v_pitch = execute_cbf(curr_pitch_rate, phi_pitch_rate, rates_max, rates_min, gamma_rates)
    v_yaw = execute_cbf(curr_yaw_rate, phi_yaw_rate, rates_max, rates_min, gamma_rates)

    v = jnp.array([v_thrust, v_roll, v_pitch, v_yaw])
    return v


@jit
def fake_tracker(currstate, currinput, ref, T_lookahead, lookahead_step, integration_step, mass):
    """Fake tracker for testing purposes."""
    print("Fake tracker")
    print(f"{currstate = }")
    print(f"{currinput = }")
    print(f"{ref = }")
    print(f"{T_lookahead = }, {lookahead_step = }, {integration_step = }, {mass = }")
    alpha = jnp.array([20, 30, 30, 30])
    print(f"{alpha = }")

    pred = predict_output(currstate, currinput, T_lookahead, lookahead_step, mass)
    print(f"{pred = }")

    error = get_tracking_error(ref, pred)
    print(f"{error=}")

    dgdu = get_jac_pred_u(currstate, currinput, T_lookahead, lookahead_step, mass)
    dgdu_inv = jnp.linalg.inv(dgdu)
    dgdu_inv2 = get_inv_jac_pred_u(currstate, currinput, T_lookahead, lookahead_step, mass)
    print(f"{dgdu=}\n{dgdu_inv=}\n{dgdu_inv2=}\n")


    NR = dgdu_inv @ error  # calculates newton-raphson control input without speed-up parameter
    print(f"{NR=}")

    v = integral_cbf(currinput, NR)
    print(f"{v= }")
    udot = NR + v
    print(f"{udot=}")

    change_u = udot * integration_step  # crude integration of udot to get u (maybe just use 0.02 as period)
    print(f"{change_u=}")
    u = currinput + alpha * change_u  # u_new = u_old + alpha * change_u
    print(f"{u=}")

    return u, v

   
@jit
def NR_tracker_original(currstate, currinput, ref, T_lookahead, lookahead_step, integration_step, mass):
    """Standard Newton-Raphson method to track the reference trajectory with forward euler integration of dynamics for prediction."""
    alpha = jnp.array([20, 30, 30, 30])
    pred = predict_output(currstate, currinput, T_lookahead, lookahead_step, mass)
    error = get_tracking_error(ref, pred) # calculates tracking error
    # dgdu = get_jac_pred_u(currstate, currinput, T_lookahead, lookahead_step, mass)
    # dgdu_inv = jnp.linalg.inv(dgdu)
    dgdu_inv2 = get_inv_jac_pred_u(currstate, currinput, T_lookahead, lookahead_step, mass)
    NR = dgdu_inv2 @ error # calculates newton-raphson control input without speed-up parameter
    v = integral_cbf(currinput, NR)
    udot = NR + v # udot = { inv(dg/du) * (yref - ypred) } + v = NR + v = newton-raphson control input + CBF adjustment
    change_u = udot * integration_step #crude integration of udot to get u (maybe just use 0.02 as period)
    u = currinput + alpha * change_u # u_new = u_old + alpha * change_u
    
    return u, v

# @jit
def error_func(state, u, T_lookahead, lookahead_step, ref, mass):
    """Norm-squared error for my optim-based NR tracker."""
    pred = predict_output(state, u, T_lookahead, lookahead_step, mass)
    # error = get_tracking_error(ref, pred)
    error = ref-pred
    return jnp.sum( error**2 ) 

# @jit
def get_grad_error_u(state, u, T_lookahead, lookahead_step, ref, mass):
    """Get the gradient of the error function with respect to the control input."""
    return jacfwd(error_func, 1)(state, u, T_lookahead, lookahead_step, ref, mass)

# @jit
def get_hess_error_u(state, u, T_lookahead, lookahead_step, ref, mass):
    """Get the hessian of the error function with respect to the control input."""
    return jacfwd(get_grad_error_u, 1)(state, u, T_lookahead, lookahead_step, ref, mass).reshape(4,4)

# @jit
def NR_tracker_optim(currstate, currinput, ref, T_lookahead, lookahead_step, integration_step, mass):
    """Optimization Newton-Raphson method testing for tracking control."""
    alpha = jnp.array([20, 30, 30, 30])
    dgdu = get_grad_error_u(currstate, currinput, T_lookahead, lookahead_step, ref, mass)
    hess = get_hess_error_u(currstate, currinput, T_lookahead, lookahead_step, ref, mass)
    d2gdu2_inv = jnp.linalg.inv(hess)
    scaling = d2gdu2_inv @ dgdu
    NR = scaling
    udot = -alpha * NR
    change_u = udot * integration_step
    new_u = currinput + change_u
    return new_u

# @jit
def quaternion_from_yaw(yaw):
    """Converts a yaw angle to a quaternion."""
    half_yaw = yaw / 2.0
    return jnp.array([jnp.cos(half_yaw), 0, 0, jnp.sin(half_yaw)])

# @jit
def quaternion_conjugate(q):
    """Returns the conjugate of a quaternion."""
    return jnp.array([q[0], -q[1], -q[2], -q[3]])

# @jit
def quaternion_multiply(q1, q2):
    """Multiplies two quaternions."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return jnp.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2
    ])

# @jit
def yaw_error_from_quaternion(q):
    """Returns the yaw error from the quaternion of angular error."""
    return 2 * jnp.arctan2(q[3], q[0])

# @jit
def quaternion_normalize(q):
    """Normalizes a quaternion."""
    return q / jnp.linalg.norm(q)

# @jit
def shortest_path_yaw_quaternion(current_yaw, desired_yaw):
    """Returns the shortest path between two yaw angles with quaternions."""
    q_current = quaternion_normalize(quaternion_from_yaw(current_yaw))
    q_desired = quaternion_normalize(quaternion_from_yaw(desired_yaw))
    q_error = quaternion_multiply(q_desired, quaternion_conjugate(q_current))
    q_error_normalized = quaternion_normalize(q_error)
    return yaw_error_from_quaternion(q_error_normalized)

# @jit
def get_tracking_error(ref, pred):
    """Calculates the tracking error between the reference and predicted outputs with yaw error handled by quaternions."""
    err = ref - pred
    err = err.at[3].set(shortest_path_yaw_quaternion(pred[3], ref[3]))
    return err

# @jit
def linear_predictor(STATE, INPUT, MASS, eAT, int_eATB):
    gravity = jnp.array([MASS * GRAVITY, 0, 0, 0])#gravity vector that counteracts input vector: [mg, 0, 0, 0]
    return C @ (eAT@STATE + int_eATB @ (INPUT - gravity)) # y(t+T) = C * (eAT * x(t) + int_eATB * (u(t) - gravity))

@jit
def NR_tracker_linpred(currstate, currinput, ref, integration_step, mass, eAT, int_eATB, jac_inv):
    """Newton-Raphson method with linearized prediction for tracking"""
    # print("NR_tracker_linpred called with parameters:")
    alpha = jnp.array([20, 30, 30, 30])
    pred = linear_predictor(currstate, currinput, mass, eAT, int_eATB)
    error = get_tracking_error(ref, pred) # calculates tracking error

    NR = jac_inv @ error # calculates newton-raphson control input without speed-up parameter
    v = integral_cbf(currinput, NR)
    udot = NR + v # udot = { inv(dg/du) * (yref - ypred) } + v = NR + v = newton-raphson control input + CBF adjustment
    change_u = udot * integration_step #crude integration of udot to get u (maybe just use 0.02 as period)
    u = currinput + alpha * change_u # u_new = u_old + alpha * change_u

    # print(f"{v.shape}, udot.shape = {udot.shape}, change_u.shape = {change_u.shape}, u.shape = {u.shape}, {NR.shape=}")
    return u, v


@jit
def NR_tracker_flat(STATE, INPUT, x_df, ref, T_LOOKAHEAD, step, MASS, ROT, yaw_dot):
    curr_x, curr_y, curr_z, curr_yaw = STATE[0], STATE[1], STATE[2], STATE[-1]
    curr_vx, curr_vy, curr_vz = STATE[3], STATE[4], STATE[5]
    curr_yawdot = INPUT[3]

    z1 = jnp.array([curr_x, curr_y, curr_z, curr_yaw])
    z2 = jnp.array([curr_vx, curr_vy, curr_vz, curr_yawdot])
    z3 = x_df[8:12]

    dgdu_inv = (2 / T_LOOKAHEAD**2) * jnp.eye(4)
    alpha = jnp.array([20, 30, 30, 30])
    pred = z1 + z2 * T_LOOKAHEAD + (1 / 2) * z3 * T_LOOKAHEAD**2
    
    error = ref - pred
    NR = dgdu_inv @ error
    u_df = alpha * NR
    
    A_df = jnp.block([
                [jnp.zeros((4, 4)), jnp.eye(4), jnp.zeros((4, 4))],
                [jnp.zeros((4, 4)), jnp.zeros((4, 4)), jnp.eye(4)],
                [jnp.zeros((4, 4)), jnp.zeros((4, 4)), jnp.zeros((4, 4))]
            ])
    
    B_df = jnp.block([
                [jnp.zeros((4, 4))],
                [jnp.zeros((4, 4))],
                [jnp.eye(4)]
            ])
    
    x_dot_df = A_df @ x_df + B_df @ u_df
    x_df = x_df + x_dot_df * step
    # print(f"{x_dot_df = }, {x_dot_df.shape = }")
    # print(f"{x_df = }, {x_df.shape = }")

    sigma = x_df[0:4]
    sigmad1 = x_df[4:8]
    sigmad2 = x_df[8:12]
    sigmad3 = u_df

    a1 = jnp.array([1,0,0])
    a2 = jnp.array([0,1,0])
    a3 = jnp.array([0, 0, -1])

    accels = sigmad2[0:3] + GRAVITY * a3
    thrust = MASS * jnp.linalg.norm(accels)
    
    thrust_max = 27.0
    thrust_min = 0.5
    clipped_thrust = jnp.clip(thrust, thrust_min, thrust_max)
    
    b3 = accels / jnp.linalg.norm(accels)


    e1 = jnp.cos(curr_yaw) * a1 + jnp.sin(curr_yaw) * a2
    val2 = jnp.cross(b3, e1, axis=0)
    b2 = val2 / jnp.linalg.norm(val2)
    b1 = jnp.cross(b2, b3, axis=0)

    # print(f"{a1=}, {a2=}, {a3=}")
    # print(f"{e1=}, {b2=}, {b3=}, {b1=}")


    j = sigmad3[0:3]
    val3 = j - (j.T @ b3) * b3

    p = (b2.T @ ((MASS / -clipped_thrust) * val3))#.item()
    q = (b1.T @ ((MASS / -clipped_thrust) * val3))#.item()
    # print(f"{p = }, {q = }")
    A = jnp.vstack((e1, b2, a3))
    B = ROT

    # Define known values in x and y
    # x_known = jnp.array([ref[3][0]])
    x_known = jnp.array([yaw_dot])
    y_known = jnp.hstack([p, q])  # y1 = 4, y2 = 2

    # print(f"{x_known = }")
    # print(f"{y_known = }")
    # print(f"{x_known.shape = }")
    # print(f"{y_known.shape = }")
    # exit(0)

    # print(f"{A = }, {A.shape = }")
    # print(f"{B = }, {B.shape = }")
    # Split A into parts affecting unknowns (A1) and knowns (A2)
    A1 = A[:, :2]  # First two columns (unknown x1, x2)
    A2 = A[:, 2:]  # Last column (known x3)

    # Split B into parts affecting unknowns (B1) and knowns (B2)
    B1 = B[:, 2:]  # Last column (unknown y3)
    B2 = B[:, :2]  # First two columns (known y1, y2)

    # Compute right-hand side
    rhs = (B2 @ y_known) - (A2 @ x_known)  # Move known values to RHS

    # Solve system for [x1, x2, y3]
    M = jnp.hstack((A1, -B1))  # Construct coefficient matrix
    unknowns = jnp.linalg.solve(M, rhs)  # Solve for unknowns

    # Extract solutions
    x1, x2, r = unknowns
    r = -r
    # print(f"{r = }")



    max_rate = 0.8
    p = jnp.clip(p, -max_rate, max_rate)
    q = jnp.clip(q, -max_rate, max_rate)
    r = jnp.clip(r, -max_rate, max_rate)
    
    u = jnp.array([clipped_thrust.reshape((1,1)), p.reshape((1,1)), q.reshape((1,1)), r.reshape((1,1))]).reshape((4,1))
    return u, x_df
    # return clipped_thrust, p, q, r, x_df