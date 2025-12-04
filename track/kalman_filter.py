import numpy as np

class KalmanFilterSimple:
    def __init__(self):
        ndim, dt = 4, 1.0
        self._dim_x = 2 * ndim
        self._dim_z = ndim
        self.f_mat = np.eye(self._dim_x)
        for i in range(ndim):
            self.f_mat[i, ndim + i] = dt
        self.h_mat = np.zeros((self._dim_z, self._dim_x))
        self.h_mat[:, :ndim] = np.eye(ndim)
        self._std_pos = 1.0 / 20.0
        self._std_vel = 1.0 / 160.0 

    def initiate(self, z_vector):
        x_vector_pos = z_vector.reshape(-1)
        x_vector_vel = np.zeros_like(x_vector_pos)
        x_vector = np.r_[x_vector_pos, x_vector_vel]
        p_mat = np.eye(self._dim_x)
        p_mat[:4, :4] *= (self._std_pos ** 2)
        p_mat[4:, 4:] *= (self._std_vel ** 2)
        return x_vector, p_mat
    
    def predict(self, x_vector, p_mat):
        q_mat = np.eye(self._dim_x)
        q_mat[:4, :4] *= (self._std_pos ** 2)
        q_mat[4:, 4:] *= (self._std_vel ** 2)
        x_vector = np.dot(self.f_mat, x_vector)
        p_mat = np.linalg.multi_dot((self.f_mat, p_mat, self.f_mat.T)) + q_mat
        return x_vector, p_mat

    def update(self, x_vector, p_mat, z_vector):
        z_vector_hat = np.dot(self.h_mat, x_vector)
        s_mat = np.linalg.multi_dot((self.h_mat, p_mat, self.h_mat.T))
        r_mat = np.eye(self._dim_z) * (self._std_pos ** 2)
        s_mat += r_mat
        K = np.linalg.multi_dot((p_mat, self.h_mat.T, np.linalg.inv(s_mat)))
        y_vector = z_vector - z_vector_hat
        x_vector = x_vector + np.dot(K, y_vector)
        I = np.eye(self._dim_x)
        p_mat = np.dot(I - np.dot(K, self.h_mat), p_mat)
        return x_vector, p_mat
