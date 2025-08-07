import jax
import jax.numpy as jnp
import immrax as irx
from functools import partial
from testtt.utilities import sim_constants # Import simulation constants
from .TVGPR import TVGPR # Import the TVGPR class for Gaussian Process Regression
from .GPR import GPR # Import the GPR class for Gaussian Process Regression
import control
import numpy as np

## Some configurations
jax.config.update("jax_enable_x64", True)
def jit (*args, **kwargs): # A simple wrapper for JAX's jit function to set the backend device
    device = 'cpu'
    kwargs.setdefault('backend', device)
    return jax.jit(*args, **kwargs)


## Get input from rollout feedforward input and error between rollout reference with LQR feedback
@jit
def u_applied(x, xref, uref, K_feedback):
    # u_nom = (K_feedback @ (xref - x)) + uref
    # u_applied_clipped = jnp.clip(u_nom, ulim_planar.lower, ulim_planar.upper) # clip the applied input to the input saturation limits

    error = x - xref  # Compute the error between the reference and the current state
    u_fb = -K_feedback @ error
    u_total = u_fb + uref 
    u_clipped = jnp.clip(u_total, ulim_planar.lower, ulim_planar.upper)  # Clip the control input to the limits

    return u_clipped
 

## Get mean of GP at a time based on height
def get_gp_mean(GP, t, x) :
    return GP.mean(jnp.array([t, x[1]])).reshape(-1)


## Collection idx function
@jit
def collection_id_jax(xref, xemb, threshold=0.3):
    diff1 = jnp.abs(xref - xemb[:, :xref.shape[1]]) > threshold
    diff2 = jnp.abs(xref - xemb[:, xref.shape[1]:]) > threshold
    nan_mask = jnp.isnan(xref).any(axis=1) | jnp.isnan(xemb).any(axis=1)
    fail_mask = diff1.any(axis=1) | diff2.any(axis=1) | nan_mask

    # Safe handling using lax.cond
    return jax.lax.cond(
        jnp.any(fail_mask),
        lambda _: jnp.argmax(fail_mask),  # return first failing index
        lambda _: -1,                     # otherwise -1
        operand=None
    )


## Rollout function
# @partial(jax.jit, static_argnames=['T', 'dt', 'perm', 'sys_mjacM'])
@partial(jax.jit, static_argnames=['T', 'dt', 'perm', 'sys_mjacM'])
def rollout(t_init, ix, xc, K_feed, K_reference, obs, T, dt, perm, sys_mjacM):
    def mean_disturbance (t, x) :
            return GP.mean(jnp.hstack((t, x[1]))).reshape(-1)    

    def sigma(t, x):
        return jnp.sqrt(GP.variance(jnp.hstack((t, x[1]))).reshape(-1))

    def sigma_bruteforce_if(t, ix):
        div = 100
        x_list = [ix.lower + ((ix.upper - ix.lower)/div)*i for i in range(div)]
        sigma_list = 3.0*jnp.array([sigma(t, x) for x in x_list]) # # TODO: get someone to explain: why 3.0?
        return irx.interval(jnp.array([jnp.min(sigma_list)]), jnp.array([jnp.max(sigma_list)]))
        
    def step (carry, t) :
        xt_emb, xt_ref, MS = carry
        error = xt_ref - jnp.array([0., -1.95, 0., 0., 0.])  # Compute the error between the reference and the current state
        nominal = -K_reference @ error  # Compute the nominal input based on the reference state and feedback gain
        uG = nominal + jnp.array([sim_constants.MASS * sim_constants.GRAVITY, 0.0])  # Add the gravitational force to the nominal input
        u_ref_clipped = jnp.clip(uG, ulim_planar.lower, ulim_planar.upper)  # Clip the reference input to the input saturation limits
        # u_reference = (K_reference @ xt_ref) + jnp.array([sim_constants.MASS * sim_constants.GRAVITY, 0.0]) # get the reference input from the linearized planar quad LQR feedback K and current state
        # u_ref_clipped = jnp.clip(u_reference, ulim_planar.lower, ulim_planar.upper) # clip the reference input to the input saturation limits
        # jax.debug.print("xt_ref: {xt}, u_reference: {ur}, u_ref_clipped: {uc}", xt=xt_ref, ur=u_reference, uc=u_ref_clipped)


        # GP Interval Work
        GP_mean_t = GP.mean(jnp.array([t, xt_ref[1]])).reshape(-1) # get the mean of the disturbance at the current time and height
        xint = irx.ut2i(xt_emb) # buffer sampled sigma bound with lipschitz constant to recover guarantee
        x_div = (xint.upper - xint.lower)/(100*2) # x_div is 
        sigma_lip = MS @ x_div.T # Lipschitz constant for sigma function above
        w_diff = sigma_bruteforce_if(t, irx.ut2i(xt_emb)) # TODO: Explain
        w_diffint = irx.icentpert(0.0, w_diff.upper + sigma_lip.upper[1]) # TODO: Explain
        wint = irx.interval(GP_mean_t) + w_diffint

        
        # Compute the mixed Jacobian inclusion matrix for the system dynamics function and the disturbance function
        Mt, Mx, Mu, Mw = sys_mjacM( irx.interval(t), irx.ut2i(xt_emb), ulim_planar, wint,
                                    centers=((jnp.array([t]), xt_ref, u_ref_clipped, GP_mean_t),), 
                                    permutations=(perm,))[0]
        
        _, MG = G_mjacM(irx.interval(jnp.array([t])), irx.ut2i(xt_emb), 
                        centers=((jnp.array([t]), xt_ref,),), 
                        permutations=(G_perm,))[0]
        Mt = irx.interval(Mt)
        Mx = irx.interval(Mx)
        Mu = irx.interval(Mu)
        Mw = irx.interval(Mw)

        

        # Embedding system for reachable tube overapproximation due to state/input/disturbance uncertainty around the quad_sys_planar.f reference system under K_ref
        F = lambda t, x, u, w: (Mx + Mu@K_feed + Mw@MG)@(x - xt_ref) + Mw@w_diffint + quad_sys_planar.f(0., xt_ref, u_ref_clipped, GP_mean_t) # with GP Jac
        embsys = irx.ifemb(quad_sys_planar, F)
        xt_emb_p1 = xt_emb + dt*embsys.E(irx.interval(jnp.array([t])), xt_emb, u_ref_clipped, wint)

        # Move the reference forward in time as well
        # jax.debug.print("GPmean: {GP_mean}", GP_mean=GP_mean_t)
        xt_ref_p1 = xt_ref + dt*quad_sys_planar.f(t, xt_ref, u_ref_clipped, GP_mean_t)

        
        return ((xt_emb_p1, xt_ref_p1, MS), (xt_emb_p1, xt_ref_p1, u_ref_clipped))
    


    GP = TVGPR(obs, sigma_f = 5.0, l=2.0, sigma_n = 0.01, epsilon = 0.25) # define the GP model for the disturbance
    tt = jnp.arange(0, T, dt) + t_init # define the time horizon for the rollout
    MS0 = jax.jacfwd(sigma, argnums=(1,))(t_init, xc)[0] #TODO: Explain

    G_mjacM = irx.mjacM(mean_disturbance) # TODO: Explain
    G_perm = irx.Permutation((0, 1, 2, 4, 5, 3))

    _, xx = jax.lax.scan(step, (irx.i2ut(ix), xc, irx.interval(MS0)), tt) #TODO: change variable names to be more descriptive
    return jnp.vstack((irx.i2ut(ix), xx[0])), jnp.vstack((xc, xx[1])), jnp.vstack(xx[2]) #TODO: change variable names to be more descriptive

# jitted_rollout = jax.jit(rollout, static_argnames=['T', 'dt', 'perm', 'sys_mjacM']) # JIT compile the rollout function for performance
jitted_rollout = rollout

## Planar Case
class PlanarMultirotorTransformed(irx.System) :
    def __init__ (self) :
        self.xlen = 5
        self.evolution = 'continuous'
        self.G = sim_constants.GRAVITY # gravitational acceleration in m/s^2
        self.M = sim_constants.MASS # mass of the multirotor in kg

    def f(self, t, x, u, w): #jax version of Eq.30 in "Trajectory Tracking Runtime Assurance for Systems with Partially Unknown Dynamics"
        py, pz, h, v, theta = x
        u1, u2 = u
        wz = w # horizontal wind disturbance as a function of height
        G = self.G
        M = self.M

        return jnp.hstack([
            h*jnp.cos(theta) - v*jnp.sin(theta), #py_dot = hcos(theta) - vsin(theta)
            h*jnp.sin(theta) + v*jnp.cos(theta), #pz_dot = hsin(theta) + vcos(theta)
            (wz/M) * jnp.cos(theta) + G*jnp.sin(theta), #hdot = (wz/m)*cos(theta) + G*sin(theta)
            -(u1/M) * jnp.cos(theta) + G*jnp.cos(theta) - (wz/M) * jnp.sin(theta), #vdot = -(u1/m)*cos(theta) + G*cos(theta) - (wz/m)*sin(theta)
            u2
        ])

quad_sys_planar = PlanarMultirotorTransformed()
ulim_planar = irx.interval([0, -1],[21, 1]) # Input saturation interval -> -5 <= u1 <= 15, -5 <= u2 <= 5
Q_planar = jnp.array([1, 1, 1, 1, 1]) * jnp.eye(quad_sys_planar.xlen) # weights that prioritize overall tracking of the reference (defined below)
R_planar = jnp.array([1, 1]) * jnp.eye(2)
Q_ref_planar =jnp.array([20, 50, 500, 500, 1]) * jnp.eye(quad_sys_planar.xlen) # Different weights that prioritize reference reaching origin
R_ref_planar = jnp.array([20, 20]) * jnp.eye(2)

## 3D Case
class ThreeDMultirotorTransformed(irx.System):
    def __init__(self):
        self.xlen = 9
        self.evolution = 'continuous'
        self.G = sim_constants.GRAVITY  # gravitational acceleration in m/s^2
        self.M = sim_constants.MASS  # mass of the multirotor in kg
        self.C = jnp.array([[1, 0, 0, 0, 0, 0, 0, 0, 0],
               [0, 1, 0, 0, 0, 0, 0, 0, 0],
               [0, 0, 1, 0, 0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0, 0, 0, 0, 1]])

    def f(self, t, state, input, w):
        """Quadrotor dynamics. xdot = f(x, u, w)."""
        x, y, z, vx, vy, vz, roll, pitch, yaw = state
        curr_thrust = input[0]
        body_rates = input[1:]
        GRAVITY = self.G
        MASS = self.M
        wz = w # horizontal wind disturbance as a function of height

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

        vxdot = -(curr_thrust / MASS) * (sr * sy + cr * cy * sp)
        vydot = -(curr_thrust / MASS) * (cr * sy * sp - cy * sr)
        vzdot = GRAVITY - (curr_thrust / MASS) * (cr * cp)

        return jnp.hstack([vx, vy, vz, vxdot, vydot, vzdot, curr_rolldot, curr_pitchdot, curr_yawdot])

quad_sys_3D = ThreeDMultirotorTransformed()
ulim_3D = irx.interval([0, -1, -1, -1], [21, 1, 1, 1])  # Input saturation interval -> 0 <= u1 <= 20, -1 <= u2 <= 1, -1 <= u3 <= 1
Q_3D = jnp.array([1, 1, 1, 1, 1, 1, 1, 1, 1]) * jnp.eye(quad_sys_3D.xlen)  # weights that prioritize overall tracking of the reference (defined below)
R_3D = jnp.array([1, 1, 1, 1]) * jnp.eye(4)  # weights for the control input (thrust and body rates)


## JAX Linearization Function
@partial(jit, static_argnums=0)
def jax_linearize_system(sys, x0, u0, w0):
    """Compute the Jacobian of the system dynamics function with respect to state and input at the initial conditions."""
    A, B = jax.jacfwd(sys.f, argnums=(1, 2))(0, x0, u0, w0) # Compute the Jacobian of the system dynamics function with respect to state and input at the initial conditions
    return A, B
# jitted_linearize_system = jax.jit(jax_linearize_system, static_argnums=0)
jitted_linearize_system = jax_linearize_system



if __name__ == "__main__":

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
    
    # Initial conditions
    x0 = jnp.array([-1.5, -2., 0., 0., 0.1]) # [x1=py, x2=pz, x3=h, x4=v, x5=theta]
    u0 = jnp.array([sim_constants.MASS*sim_constants.GRAVITY, 0.0]) # [u1=thrust, u2=roll angular rate]
    w0 = jnp.array([0.01]) # [w1= unkown horizontal wind disturbance]
    x0_pert = jnp.array([0.01, 0.01, 0.01, 0.01, 0.01])
    ix0 = irx.icentpert(x0, x0_pert)

    n_obs = 9
    obs = jnp.tile(jnp.array([[0, x0[1], get_gp_mean(actual_disturbance_GP, 0.0, x0)[0]]]),(n_obs,1))

    A,B = jitted_linearize_system(quad_sys_planar, x0, u0, w0)
    K_reference, P, _ = control.lqr(A, B, Q_ref_planar, R_ref_planar)
    K_feedback, P, _ = control.lqr(A, B, Q_planar, R_planar)

    # t0 = 0.     # Initial time
    # dt = 0.01  # Time step
    # T = 30.0   # Reachable tube horizon
    # tt = jnp.arange(t0, T+dt, dt)

    # sys_mjacM = irx.mjacM(quad_sys_planar.f) # create a mixed Jacobian inclusion matrix for the system dynamics function
    # perm = irx.Permutation((0, 1, 2, 3, 4, 5, 6, 7, 8)) # create a permutation for the inclusion system calculation



    # , T, dt, perm, sys_mjacM):
    reachable_tube, rollout_ref, rollout_feedfwd_input = rollout(0.0, ix0, x0, K_feedback, K_reference, obs, 30., 0.01, irx.Permutation((0, 1, 2, 3, 4, 5, 6, 7, 8)), irx.mjacM(quad_sys_planar.f))


    n_obs = 9
    obs = jnp.tile(jnp.array([[0, x0[1], get_gp_mean(actual_disturbance_GP, 0.0, x0)[0]]]),(n_obs,1))
    print(f"{get_gp_mean(actual_disturbance_GP, 0.0, jnp.array([0, 0])) = }")
    print(f"{get_gp_mean(actual_disturbance_GP, 0.0, jnp.array([0, -0.55])) = }")
    print(f"{get_gp_mean(actual_disturbance_GP, 0.0, jnp.array([0, -5])) = }")
    print(f"{get_gp_mean(actual_disturbance_GP, 0.0, jnp.array([0, -10])) = }")
    print(f"{obs = }")

    print("This module is not intended to be run directly. Please import it in your main script.")



# @partial(jax.jit, static_argnames=['T', 'dt', 'perm', 'sys_mjacM'])
# def rollout(t_init, ix, xc, K_feed, K_reference, obs, T, dt, perm, sys_mjacM):
#     def mean_disturbance (t, x) :
#             return GP.mean(jnp.hstack((t, x[1]))).reshape(-1)    

#     def sigma(t, x):
#         return jnp.sqrt(GP.variance(jnp.hstack((t, x[1]))).reshape(-1))

#     def sigma_bruteforce_if(t, ix):
#         div = 100
#         x_list = [ix.lower + ((ix.upper - ix.lower)/div)*i for i in range(div)]
#         sigma_list = 3.0*jnp.array([sigma(t, x) for x in x_list]) # # TODO: get someone to explain: why 3.0?
#         return irx.interval(jnp.array([jnp.min(sigma_list)]), jnp.array([jnp.max(sigma_list)]))
        
#     def step (carry, t) :
#         xt_emb, xt_ref, MS = carry

#         u_reference = (K_reference @ xt_ref) + jnp.array([sim_constants.GRAVITY, 0.0]) # get the reference input from the linearized planar quad LQR feedback K and current state
#         u_ref_clipped = jnp.clip(u_reference, ulim_planar.lower, ulim_planar.upper) # clip the reference input to the input saturation limits


#         # GP Interval Work
#         GP_mean_t = GP.mean(jnp.array([t, xt_ref[1]])).reshape(-1) # get the mean of the disturbance at the current time and height
#         xint = irx.ut2i(xt_emb) # buffer sampled sigma bound with lipschitz constant to recover guarantee
#         x_div = (xint.upper - xint.lower)/(100*2) # x_div is 
#         sigma_lip = MS @ x_div.T # Lipschitz constant for sigma function above
#         w_diff = sigma_bruteforce_if(t, irx.ut2i(xt_emb)) # TODO: Explain
#         w_diffint = irx.icentpert(0.0, w_diff.upper + sigma_lip.upper[1]) # TODO: Explain
#         wint = irx.interval(GP_mean_t) + w_diffint

        
#         # Compute the mixed Jacobian inclusion matrix for the system dynamics function and the disturbance function
#         Mt, Mx, Mu, Mw = sys_mjacM( irx.interval(t), irx.ut2i(xt_emb), ulim_planar, wint,
#                                     centers=((jnp.array([t]), xt_ref, u_ref_clipped, GP_mean_t),), 
#                                     permutations=(perm,))[0]
        
#         _, MG = G_mjacM(irx.interval(jnp.array([t])), irx.ut2i(xt_emb), 
#                         centers=((jnp.array([t]), xt_ref,),), 
#                         permutations=(G_perm,))[0]
#         Mt = irx.interval(Mt)
#         Mx = irx.interval(Mx)
#         Mu = irx.interval(Mu)
#         Mw = irx.interval(Mw)

        

#         # Embedding system for reachable tube overapproximation due to state/input/disturbance uncertainty around the quad_sys_planar.f reference system under K_ref
#         F = lambda t, x, u, w: (Mx + Mu@K_feed + Mw@MG)@(x - xt_ref) + Mw@w_diffint + quad_sys_planar.f(0., xt_ref, u_ref_clipped, GP_mean_t) # with GP Jac
#         embsys = irx.ifemb(quad_sys_planar, F)
#         xt_emb_p1 = xt_emb + dt*embsys.E(irx.interval(jnp.array([t])), xt_emb, u_ref_clipped, wint)

#         # Move the reference forward in time as well
#         xt_ref_p1 = xt_ref + dt*quad_sys_planar.f(t, xt_ref, u_ref_clipped, GP_mean_t)

        
#         return ((xt_emb_p1, xt_ref_p1, MS), (xt_emb_p1, xt_ref_p1, u_ref_clipped))
    


#     GP = TVGPR(obs, sigma_f = 5.0, l=2.0, sigma_n = 0.01, epsilon = 0.25) # define the GP model for the disturbance
#     tt = jnp.arange(0, T, dt) + t_init # define the time horizon for the rollout
#     MS0 = jax.jacfwd(sigma, argnums=(1,))(t_init, xc)[0] #TODO: Explain

#     G_mjacM = irx.mjacM(mean_disturbance) # TODO: Explain
#     G_perm = irx.Permutation((0, 1, 2, 4, 5, 3))

#     _, xx = jax.lax.scan(step, (irx.i2ut(ix), xc, irx.interval(MS0)), tt) #TODO: change variable names to be more descriptive
#     return jnp.vstack((irx.i2ut(ix), xx[0])), jnp.vstack((xc, xx[1])), jnp.vstack(xx[2]) #TODO: change variable names to be more descriptive

# @jit
# def collection_id_jax(xref, xemb, threshold=0.3):
#     diff1 = jnp.abs(xref - xemb[:, :xref.shape[1]]) > threshold
#     diff2 = jnp.abs(xref - xemb[:, xref.shape[1]:]) > threshold
#     nan_mask = jnp.isnan(xref).any(axis=1) | jnp.isnan(xemb).any(axis=1)
#     fail_mask = diff1.any(axis=1) | diff2.any(axis=1) | nan_mask

#     # Safe handling using lax.cond
#     return jax.lax.cond(
#         jnp.any(fail_mask),
#         lambda _: jnp.argmax(fail_mask),  # return first failing index
#         lambda _: -1,                     # otherwise -1
#         operand=None
#     )
