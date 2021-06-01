import numpy as np
from gpflowopt.domain import ContinuousParameter
import tensorflow as tf
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import gpflow
from pyDOE import lhs
dim=2
num_ini=3
priori_known_mode=True
low_bdd_prior=-1.0
eps = 1e-4
Max_iter = 30
lr = 1e-1
sig_e = 1e-4
if priori_known_mode==False:
    low_bdd=0.0
else:
    low_bdd=low_bdd_prior

def neg_branin(x):
    x = np.atleast_2d(x)
    x1 = x[:, 0]
    x2 = x[:, 1]
    x11 = 15.0 * x1 - 5.0
    x22 = 15.0 * x2
    a = 1.
    b = 5.1 / (4. * np.pi ** 2)
    c = 5. / np.pi
    r = 6.
    s = 10.
    t = 1. / (8. * np.pi)
    ret = -(a * (x22 - b * x11 ** 2 + c * x11 - r) ** 2 + s * (1 - t) * np.cos(x11) + s)
    return ret[:, None]

def K_np(x_data, phi, sigma):
    x_data = np.multiply(x_data, np.reshape(np.exp(phi), [1, -1]))
    dist = np.sum(np.square(x_data), axis=1)
    dist = np.reshape(dist, [-1, 1])
    sq_dists = np.add(np.subtract(dist, 2 * np.matmul(x_data, np.transpose(x_data))), np.transpose(dist))
    my_kernel = np.exp(sigma) * (np.exp(-sq_dists)) + sig_e * np.diag(np.ones(np.shape(x_data)[0]))
    return my_kernel
def K_star_np(x_data, X_M, phi, sigma):
    x_data = np.multiply(x_data, np.reshape(np.exp(phi), [1, -1]))
    X_M = np.multiply(X_M, np.reshape(np.exp(phi), [1, -1]))

    rA = np.reshape(np.sum(np.square(x_data), 1), [-1, 1])
    rB = np.reshape(np.sum(np.square(X_M), 1), [-1, 1])
    pred_sq_dist = np.add(np.subtract(rA, np.multiply(2., np.matmul(x_data, np.transpose(X_M)))), np.transpose(rB))
    pred_kernel = np.exp(sigma) * np.exp(-pred_sq_dist)

    return pred_kernel
def Posterior(x_data, X_M, phi, sigma, y_data, low_bdd, q_mu):
    K_M_0 = np.transpose(K_star_np(x_data, X_M, phi, 0.0))
    K_0 = K_np(x_data, phi, 0.0)
    K_0_inv = np.linalg.inv(K_0)
    GP_mu_M = np.matmul(np.matmul(K_M_0, K_0_inv), y_data)
    ratio_down = (1.0 - np.matmul(np.matmul(K_M_0, K_0_inv), np.transpose(K_M_0)))

    x = np.arange(0, 1, 1.0 / 100.0)
    y = np.arange(0, 1, 1.0 / 100.0)

    X, Y = np.meshgrid(x, y)

    for k in range(X.shape[0]):
        for j in range(Y.shape[0]):
            xx = np.reshape(np.array([X[k, j], Y[k, j]]), [1, 2])
            K_star_0 = np.transpose(K_star_np(x_data, xx, phi, 0.0))
            K_star_M_0 = K_star_np(xx, X_M, phi, 0.0)
            GP_mu_star = np.matmul(np.matmul(K_star_0, K_0_inv), y_data)
            GP_var_star = np.exp(sigma) * (1.0 - np.matmul(np.matmul(K_star_0, K_0_inv), np.transpose(K_star_0)))
            ratio_up = (K_star_M_0 - np.matmul(np.matmul(K_star_0, K_0_inv), np.transpose(K_M_0)))
            GPIO_mean = GP_mu_star + (ratio_up / (ratio_down+eps)) * (q_mu+low_bdd-GP_mu_M)
            GPIO_var = np.maximum(GP_var_star - np.square(ratio_up / (ratio_down+eps)) * (np.exp(sigma) * ratio_down),0.0)
            if k == 0 and j == 0:
                max_coor = xx
                max_value = GPIO_mean + 2.0 * np.sqrt(GPIO_var)
                max_mean=GPIO_mean
                max_var=GPIO_var
            else:
                cur_value = GPIO_mean + 2.0 * np.sqrt(GPIO_var)
                if max_value < cur_value:
                    max_coor = xx
                    max_value = cur_value
                    max_mean=GPIO_mean
                    max_var=GPIO_var
    if np.sum(np.sum(np.abs(x_data-max_coor),axis=1)==0.0) !=0.0:
        max_coor=X_M

    return max_coor,max_mean,max_var
def K(x_data, phi, sigma):
    x_data = tf.multiply(x_data, tf.reshape(tf.exp(phi), [1, -1]))
    dist = tf.reduce_sum(tf.square(x_data), 1)
    dist = tf.reshape(dist, [-1, 1])
    sq_dists = tf.add(tf.subtract(dist, 2 * tf.matmul(x_data, tf.transpose(x_data))), tf.transpose(dist))
    my_kernel = tf.exp(sigma) * (tf.exp(-sq_dists)) + sig_e * tf.reshape(tf.diag(tf.ones([1, tf.shape(x_data)[0]])),
                                                                         [tf.shape(x_data)[0], tf.shape(x_data)[0]])
    return my_kernel
def K_star(x_data, X_M, phi, sigma):
    x_data = tf.multiply(x_data, tf.reshape(tf.exp(phi), [1, -1]))
    X_M = tf.multiply(X_M, tf.reshape(tf.exp(phi), [1, -1]))

    rA = tf.reshape(tf.reduce_sum(tf.square(x_data), 1), [-1, 1])
    rB = tf.reshape(tf.reduce_sum(tf.square(X_M), 1), [-1, 1])
    pred_sq_dist = tf.add(tf.subtract(rA, tf.multiply(2., tf.matmul(x_data, tf.transpose(X_M)))), tf.transpose(rB))
    pred_kernel = tf.exp(sigma) * tf.exp(-pred_sq_dist)
    return pred_kernel
def ELBO(x_data, y_data, low_bdd, X_M, phi, sigma, alpha_q, beta_q, lamda_p):
    eps = 1e-8
    k_vec = K_star(x_data, X_M, phi, 0.0)
    k_vecT = tf.transpose(k_vec)

    R = y_data - (low_bdd) * k_vec
    RT = tf.transpose(R)

    Sig = tf.exp(sigma) * (K(x_data, phi, 0.0) - tf.matmul(k_vec, k_vecT)) + sig_e * tf.reshape(
        tf.diag(tf.ones([1, tf.shape(x_data)[0]])), [tf.shape(x_data)[0], tf.shape(x_data)[0]])
    Sig_inv = tf.matrix_inverse(Sig)

    alpha_pos = tf.exp(alpha_q)
    beta_pos = tf.exp(beta_q)
    lamda_pos = tf.exp(lamda_p)
    KL_div = (alpha_pos - 1) * tf.digamma(alpha_pos) - tf.lgamma(alpha_pos) + tf.log(beta_pos) + lamda_pos * tf.divide(
        alpha_pos, beta_pos) - alpha_pos

    Fac1 = -0.5 * tf.reduce_sum(tf.log(tf.abs(tf.diag_part(tf.cholesky(Sig))) + eps)) - 0.5 * tf.matmul(
        tf.matmul(RT, Sig_inv), R)
    Fac2 = tf.divide(alpha_pos, beta_pos) * tf.matmul(tf.matmul(k_vecT, Sig_inv), R) - 0.5 * tf.divide(
        alpha_pos + tf.square(alpha_pos), tf.square(beta_pos)) * tf.matmul(tf.matmul(k_vecT, Sig_inv), k_vec)

    result = Fac1 + Fac2 + KL_div
    return -result
def acq_fcn_EI(xx,x_data, y_data, low_bdd, X_M, phi, sigma, q_mu,q_var):
    K_M_0=tf.transpose(K_star(x_data,X_M,phi,0.0))
    K_0 = K(x_data, phi, 0.0)
    K_0_inv = tf.matrix_inverse(K_0)
    GP_mu_M = tf.matmul(tf.matmul(K_M_0, K_0_inv), y_data)
    ratio_down = (1.0 - tf.matmul(tf.matmul(K_M_0, K_0_inv), tf.transpose(K_M_0)))

    K_star_0 = tf.transpose(K_star(x_data, xx, phi, 0.0))
    K_star_M_0 = K_star(xx, X_M, phi, 0.0)
    GP_mu_star = tf.matmul(tf.matmul(K_star_0, K_0_inv), y_data)
    GP_var_star = tf.exp(sigma) * (1.0 - tf.matmul(tf.matmul(K_star_0, K_0_inv), tf.transpose(K_star_0)))
    ratio_up = (K_star_M_0 - tf.matmul(tf.matmul(K_star_0, K_0_inv), tf.transpose(K_M_0)))
    OBCGP_mean = GP_mu_star + (ratio_up / (ratio_down + eps)) * (q_mu + low_bdd - GP_mu_M)
    OBCGP_var = tf.nn.relu(GP_var_star - tf.square(ratio_up / (ratio_down + eps)) * (tf.exp(sigma) * ratio_down))+q_var

    y_max=tf.reduce_max(y_data)
    tau=tf.divide(y_max-OBCGP_mean,tf.sqrt(OBCGP_var)+eps)

    dist = tf.distributions.Normal(loc=0.0,scale=1.0)
    fcn_val= (dist.prob(tau)-tau*dist.cdf(tau))*tf.sqrt(OBCGP_var)

    return fcn_val
def par_update(X, Y):
    model = gpflow.gpr.GPR(X, Y, gpflow.kernels.RBF(dim, ARD=True), name='GPReg')
    model.likelihood.variance = sig_e
    model.likelihood.variance.fixed = True
    model.kern.lengthscales = 0.1
    model.kern.variance=1.0
    model.kern.lengthscales.prior = gpflow.priors.Gamma(1, 1)
    model.kern.variance.prior = gpflow.priors.Gamma(1., 1.)
    #model.kern.lengthscales.transform = gpflow.transforms.Exp(1e-1)
    model.optimize(maxiter=3)
    sigma_np = np.float32(np.log(model.kern.variance.value))
    phi_np = np.reshape(np.float32(0.5 * (-np.log(2.0) - 2.0 * np.log(model.kern.lengthscales.value))), [-1, 1])

    return sigma_np, phi_np




X=lhs(dim,samples=num_ini)
Y=neg_branin(X)
Y_best=np.max(Y)
Y_rescale=Y-Y_best

if priori_known_mode==False:
    low_bdd_feed=0.0
else:
    low_bdd_feed=np.maximum(low_bdd_prior-Y_best,0.0)

lamda_feed = np.float32(np.log(1.0))
alpha_feed = np.float32(0.0)
beta_feed = lamda_feed
X_M_t = np.reshape(X[np.argmax(Y)], (1, dim))
X_M_t_feed = np.log(np.divide(X_M_t, 1.0 - X_M_t + eps)) + np.random.normal(0.0, 0.01, [1, 2])
for iter in range(50):
    sigma_feed, phi_feed = par_update(X, Y_rescale)
    tf.reset_default_graph()
    alpha_v = tf.Variable(tf.convert_to_tensor(alpha_feed, dtype=tf.float32))
    beta_v = tf.Variable(tf.convert_to_tensor(beta_feed, dtype=tf.float32))
    X_M_t_v = tf.Variable(tf.convert_to_tensor(X_M_t_feed, tf.float32))
    X_M = (tf.sigmoid(X_M_t_v))
    x_data_p = tf.convert_to_tensor(X, dtype=tf.float32)
    y_data_p = tf.convert_to_tensor(Y_rescale, dtype=tf.float32)
    low_bdd_p = tf.convert_to_tensor(low_bdd_feed, tf.float32)
    lamda_p = tf.convert_to_tensor(lamda_feed, dtype=tf.float32)
    phi_p = (tf.convert_to_tensor(phi_feed, dtype=tf.float32))
    sigma_p = (tf.convert_to_tensor(sigma_feed, dtype=tf.float32))

    pt_sel = tf.Variable(tf.convert_to_tensor(X_M_t_feed, tf.float32))
    costF = ELBO(x_data_p, y_data_p, low_bdd_p, X_M, phi_p, sigma_p, alpha_v, beta_v, lamda_p)
    acq_fcn_val = acq_fcn_EI(tf.sigmoid(pt_sel), x_data_p, y_data_p, low_bdd_p, X_M, phi_p, sigma_p,
                              tf.exp(alpha_v - beta_v),tf.exp(alpha_v-2*beta_v))

    optimizer = tf.train.AdamOptimizer(lr)
    optimizer_acq = tf.train.AdamOptimizer(1e-3)

    train_par = optimizer.minimize(costF)
    train_acq = optimizer.minimize(-acq_fcn_val, var_list=pt_sel)

    sess = tf.Session()
    init = tf.global_variables_initializer()
    sess.run(init)

    for opt_iter in range(Max_iter):
        sess.run(train_par)
    for opt_iter_acq in range(300):
        sess.run(train_acq)
    X_M_np, phi_np, sigma_np, alpha_np, beta_np = sess.run((X_M, phi_p, sigma_p, alpha_v, beta_v))
    q_mu = np.exp(alpha_np - beta_np)
    q_var = np.exp(alpha_np - 2.0 * beta_np)

    pt_next=sess.run(tf.sigmoid(pt_sel))
    val_next = np.reshape(neg_branin(pt_next), [])

    X = np.concatenate((X, pt_next), axis=0)
    Y = np.concatenate((Y, np.reshape(val_next, [1, 1])), axis=0)

    if val_next>Y_best:
        if val_next-Y_best>low_bdd_feed:
            X_M_t_feed = np.log(np.divide(pt_next + eps, 1.0 - pt_next + eps)) + np.random.normal(0.0, 0.01, [1, 2])
            alpha_feed = np.float32(0.0)
            beta_feed = lamda_feed
            X_M_t = pt_next
            Y_best = val_next
            low_bdd_feed = np.maximum(low_bdd_prior - Y_best, 0.0)
            Y_rescale=Y-Y_best
        else:
            X_M_t_feed = np.log(np.divide(pt_next + eps, 1.0 - pt_next + eps)) + np.random.normal(0.0, 0.01, [1, 2])
            alpha_feed = np.float32(0.0)
            beta_feed = lamda_feed
            X_M_t = pt_next
            Y_best = val_next
            low_bdd_feed = np.maximum(low_bdd_prior-Y_best,0.0)
            Y_rescale = Y - Y_best
    else:
        X_M_t_feed = np.log(np.divide(X_M_t + eps, 1.0 - X_M_t + eps)) + np.random.normal(0.0, 0.01, [1, 2])
        alpha_feed = np.float32(0.0)
        beta_feed = lamda_feed
        Y_rescale = Y - Y_best

    sess.close()
    tf.reset_default_graph()
    X_M_best = np.reshape(X[np.argmax(Y)], (1, dim))
    print('iteration #:%d\t' % (iter + 1) + 'current Minimum value: %f\t' % (-np.max(Y)) + '\tcurrent Minimum point: (%f,%f)' % (X_M_best[0,0],X_M_best[0,1]))
