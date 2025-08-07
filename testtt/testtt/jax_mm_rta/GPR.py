import jax
import jax.numpy as jnp
from functools import partial

class GPR():
    def __init__(self, obs : jax.Array, obs_dim : int = 0, sigma_f : float = 3.0, l : float = 1.0, sigma_n : float = 0.0001) -> None:

        if obs_dim == 0:
            self.obs_dim = obs.shape[1] - 1 
        else:
            self.obs_dim = obs_dim

        self.x = obs[:, :self.obs_dim]
        self.y = obs[:, self.obs_dim:]
        # print(self.x)
        # print(self.y)
        self.sigma_f = sigma_f
        self.l = l
        self.sigma_n = sigma_n
        
        self.set_KL()

    def mean(self, x_star : jax.Array):
        f_bar_star, _ = self.fit(x_star)
        return f_bar_star
    
    def variance(self, x_star : jax.Array):
        # ORIGINAL VERSION
        _, cov_f_star = self.fit(x_star)
        return jnp.diag(cov_f_star)

        # IMMRAX-FRENDLY VERSION (CURRENTLY 1D ONLY)
        # cov = jnp.zeros((x_star.shape[0], 1))

        # for s, xs in enumerate(x_star):
        #     ca = 0
        #     for i in range(self.L.shape[0]):
        #         for j in range(self.L.shape[1]):
        #             S = jnp.dot(self.x[i, :],self.x[i, :]) + jnp.dot(self.x[j, :],self.x[j, :]) - jnp.dot(2*(self.x[i, :]+self.x[j, :]-xs),xs)
        #             ca += self.L[i, j] * jnp.exp(-S/(2*(self.l**2)))
        #             # * jnp.exp(-(jnp.dot(self.x[i, :],self.x[i, :]))/(2*(self.l**2))) * \
        #             #         jnp.exp(-jnp.dot(self.x[j, :],self.x[j, :])/(2*(self.l**2))) * \
        #             #         jnp.exp(-jnp.dot((self.x[i, :]+self.x[j, :]),xs)/((self.l**2)))
        #             # print(ca[0])
        #     # cov = cov.at[s].set(jnp.exp(-jnp.dot(xs, xs)/(self.l**2)) * ca)
        #     cov = cov.at[s].set(ca)
            
        # cov_f_star = self.sigma_f - (self.sigma_f**2 * jnp.diag(cov))
        # return jnp.diag(cov_f_star)
    
    
    def std_dev(self, x_star : jax.Array): 
        # return jnp.sqrt(jnp.maximum(self.variance(x_star), 0.0))
        return jnp.sqrt(self.variance(x_star))
    
    def add_obs(self, obs : jax.Array):
        self.x = jnp.vstack((self.x, obs[:, :self.obs_dim]))
        self.y = jnp.vstack((self.y, obs[:, self.obs_dim:]))
        self.set_KL()

    def set_obs(self, obs : jax.Array):
        self.x = obs[:, :self.obs_dim]
        self.y = obs[:, self.obs_dim:]
        self.set_KL()

    def set_KL(self):
        self.K = self.kernel(self.x, self.x)
        self.L = jnp.linalg.inv(self.K + (self.sigma_n**2 * jnp.eye(self.K.shape[0])))

    def kernel(self, x, y):
        # check dimension of x
        x_dim = x.ndim
        # print(x_dim)
        if x_dim == 1: # if observation points are 1D
            return self.sigma_f * jnp.exp(-0.5 * (x.reshape(-1, 1) - y)**2 / self.l**2)
        elif x_dim > 1: # if observation points are 2D or more
            # result = jnp.zeros((x.shape[0], y.shape[0]))
            # for i in range(x.shape[0]):
            #     result = result.at[i, :].set(sigma_f * jnp.exp(-0.5 * jnp.sum((x[i, :] - y)**2, axis=1) / l**2)) 
            # return result
            # return sigma_f * jnp.exp(-0.5 * jnp.sum((x.T - y)**2) / l**2) 
            return jax.vmap(partial(self.multidim_kernel, y=y), in_axes=0, out_axes=0)(x)
        else:
            raise ValueError("Points must be 1D or more")


    def fit(self, x_star):
        K_star2, K_star = self.cov_matrices(x_star)
        f_bar_star, cov_f_star = self.gpr_params(K_star2, K_star)
        
        return f_bar_star, cov_f_star

    def multidim_kernel(self, x, y):
        return self.sigma_f * jnp.exp(-0.5 * jnp.sum((x - y)**2, axis=1) / self.l**2)

    def cov_matrices(self, x_star):
        # print("COV DEBUG")
        # print(x_star)
        # print(self.x)
        K_star2 = self.kernel(x_star, x_star)
        K_star = self.kernel(x_star, self.x)
        # print(K)
        # print(K_star2)
        # print(K_star)
        # print(K_star2)


        # K = jnp.zeros((x.shape[0], x.shape[0]))
        # K_star2 = jnp.zeros((x_star.shape[0], x_star.shape[0]))
        # K_star = jnp.zeros((x_star.shape[0], x.shape[0]))
        # for i in range(x.shape[0]):
        #     print(kernel(x[i, :], x, sigma_f, l))
        #     K = K.at[i, :].set(kernel(x[i, :], x, sigma_f, l))
        # for i in range(x_star.shape[0]):
        #     # K_star2 = K_star2.at[i, :].set(kernel(x_star[i, :], x_star, sigma_f, l))
        #     K_star = K_star.at[i, :].set(kernel(x_star[i, :], x, sigma_f, l).T)

        return K_star2, K_star

    def gpr_params(self, K_star2, K_star):
        # f_bar_star = jnp.dot(jnp.dot(K_star.T, jnp.linalg.inv(K + sigma_n**2 * jnp.eye(K.shape[0]))), y)
        # cov_f_star = K_star2 - jnp.dot(jnp.dot(K_star.T, jnp.linalg.inv(K + sigma_n**2 * jnp.eye(K.shape[0]))), K_star)
        # STANDARD VERSION
        
        # print("GPR DEBUG")
        # print("L", self.L.shape)
        # print("K_star", K_star.shape)
        # print("y", self.y.shape)
        # print(jnp.dot(K_star.T, self.L))
    
        # f_bar_star = jnp.dot(K_star.T, self.L).dot(self.y)
        # print('l1')
        l1 = self.L@self.y
        # print('l2')
        f_bar_star = K_star.T@l1
        # print(f_bar_star.shape)
        cov_f_star = K_star2 - jnp.dot(K_star.T, self.L).dot(K_star)
        # print(cov_f_star.shape)

        
        # CHOLESKY DECOMP VERSION
        # L = jnp.linalg.cholesky(K + (sigma_n**2 * jnp.eye(K.shape[0])))
        # print(L)
        # alpha = jnp.linalg.solve(L.T, jnp.linalg.solve(L, y))
        # print(alpha)
        # f_bar_star = jnp.dot(K_star, alpha)
        

        # v = jnp.linalg.solve(L.T, K_star.T).T
        # cov_f_star = K_star2 - jnp.dot(v, v.T)

        return f_bar_star, cov_f_star
    

