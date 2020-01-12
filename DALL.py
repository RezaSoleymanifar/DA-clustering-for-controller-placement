# %load DALL.py
import numpy as np
from math import *
from matplotlib import pyplot as plt
import matplotlib.cm as cm


class DA:
    def __init__(self, data, max_m, gamma, t_start = None, t_min = 1e-5, alpha = 0.9, interactive = None):
        self.N, self.d = np.shape(data)
        self.x = self.pre_process(data)
        self.p_x = 1 / self.N
        self.max_m = max_m
        self.t_start = t_start
        self.T_min = t_min
        self.gamma = gamma
        self.alpha = alpha
        self.interactive = interactive

        self.T = inf
        self.hist_soft = []
        self.hist_hard = []
        self.hist_best_hard = []
        self.beta_hist = []
        self.m_hist = []

        self.y = []
        self.y.append(sum(self.x[i] * self.p_x for i in range(self.N)))
        self.y_p = None
        self.y_best = self.y.copy()
        self.m = len(self.y)

        self.eta = None
        self.d_xy = None
        self.d_yy = None
        self.D_xy = None

        self.p_yx = None
        self.Z = None
        self.p_y = None
        self.p_xy = None
        self.C = None

        self.ctr = 0

    def pre_process(self, data):
        data -= np.mean(data, axis = 0)
        data /= np.max(data)
        return data

    def init_crtl_T(self):
        self.update_da()
        self.calc_associations()
        C_xy = sum(self.p_xy[(i,0)] * (np.reshape(self.x[i], (-1,1)) - np.reshape(self.y[0], (-1,1))) @
                   (np.reshape(self.x[i], (-1, 1)) - np.reshape(self.y[0], (-1, 1))).T for i in range(self.N))
        w, v= np.linalg.eig(C_xy)
        result = 2 * np.max(v)
        return result

    def sed(self, input1, input2):
        return np.linalg.norm(input1 - input2) ** 2

    def update_da(self):
        self.m = len(self.y)
        self.eta = self.gamma * (self.m - 1) + 1
        self.d_xy = {(i, j): self.sed(self.x[i], self.y[j]) for i in range(self.N) for j in range(self.m)}
        self.d_yy = {(j, j_): self.sed(self.y[j], self.y[j_]) for j in range(self.m) for j_ in range(self.m)}
        self.D_xy = {(i,j):self.d_xy[(i,j)] + self.gamma * sum(self.d_yy[(j, j_)] for j_ in range(self.m))
                     for i in range(self.N) for j in range(self.m)}
        return None

    def calc_associations(self):
        reg = self.D_xy[max(self.D_xy, key=self.D_xy.get)] / self.T
        self.p_yx = {(i, j): exp(reg - self.D_xy[(i, j)] / self.T) for i in range(self.N) for j in range(self.m)}
        self.normalize_p()
        self.p_y = [sum(self.p_yx[(i, j)] * self.p_x for i in range(self.N)) for j in range(self.m)]
        self.p_xy = {(i, j): (self.p_yx[(i, j)] * self.p_x) / self.p_y[j] for i in range(self.N) for j in range(self.m)}
        self.C = [sum(self.p_xy[(i, j)] * self.x[i] for i in range(self.N)) for j in range(self.m)]

    def normalize_p(self):
        self.Z = [sum(self.p_yx[(i, j)] for j in range(self.m)) for i in range(self.N)]
        self.p_yx = {(i, j): self.p_yx[(i, j)] / self.Z[i] for i in range(self.N) for j in range(self.m)}
        return None

    def calc_theta(self):
        I = np.eye(self.d)
        size = self.m * self.d
        self.theta = np.zeros((size, size))
        if self.m == 1:
            self.theta = self.eta * I
        else:
            for i in range(0, size, self.d):
                for j in range(0, size, self.d):
                    self.theta[i:i+self.d, j:j + self.d] = self.eta * I if i == j else - self.gamma * I
        return None

    def chk_exist(self, list, item):
        for item_ in list:
            if np.allclose(item, item_):
                return True
        return False

    def brk_list(self, list):
        if len(list) == self.d:
            temp = [list]
        else:
            temp = [list[i:i + self.d] for i in range(0, len(list), self.d)]
        return temp

    def merge_ctrds(self):
        temp = []
        for ctrd in self.y:
            if len(temp) == 0:
                temp.append(ctrd)
            elif not self.chk_exist(temp, ctrd) and len(temp) < self.max_m:
                temp.append(ctrd)
        self.y = temp.copy()
        return None

    def calc_ctrds(self):
        b = np.hstack(self.C)
        self.calc_theta()
        sol = np.linalg.solve(self.theta, b)
        self.y = self.brk_list(sol)
        self.merge_ctrds()
        self.m = len(self.y)
        self.beta_hist.append(1/self.T)
        self.m_hist.append(self.m)
        return None

    def purturb_ctrds(self):
        eps = np.random.random(self.y[0].shape)
        self.y = [self.y[j] - eps for j in range(len(self.y))]
        self.y_p = [self.y[j] + eps for j in range(len(self.y))]
        self.y = self.y + self.y_p
        return None

    def chk(self, i, j):
        if i == j:
            return True
        else:
            return False

    def calc_obj(self, idx, ctrds):
        d_xy = {(i, j): self.sed(self.x[i], ctrds[j]) for i in range(self.N) for j in range(self.m)}
        d_yy = {(j, j_): self.sed(ctrds[j], ctrds[j_]) for j in range(self.m) for j_ in range(self.m)}
        D_xy = {(i, j): d_xy[(i, j)] + self.gamma * sum(d_yy[(j, j_)] for j_ in range(self.m))
                     for i in range(self.N) for j in range(self.m)}
        D_hard = sum(self.chk(idx[i], j) * D_xy[(i, j)] for i in range(self.N) for j in range(self.m))
        return D_hard

    def proj(self, pnt):
        temp = np.zeros(self.N)
        for i in range(self.N):
            if i not in self.proj_idx:
                temp[i] = np.linalg.norm(self.x[i] - pnt)
            else:
                temp[i] = inf
        new_idx = np.argmin(temp)
        self.proj_idx.append(new_idx)
        result = self.x[new_idx]
        return result

    def update_hist(self):
        temp = np.zeros((self.N, self.m))
        for i in range(self.N):
            for j in range(self.m):
                temp[i,j] = self.p_yx[(i,j)]
        hard_idx_x = np.argmax(temp, axis=1)

        self.clusters = [[self.x[i] for i in range(self.N) if hard_idx_x[i] == j] for j in range(self.m)]
        self.proj_idx = []
        self.ctrds = [self.proj(self.y[j]) for j in range(self.m)]

        self.D_soft = sum(self.p_yx[(i, j)] * self.D_xy[(i, j)] for i in range(self.N) for j in range(self.m))
        self.D_hard = self.calc_obj(hard_idx_x, self.ctrds)

        if (len(self.hist_hard) != 0 and self.D_hard < min(self.hist_hard)) or self.ctr == 1:
            self.y_best = self.y.copy()
            self.ctrds_best = self.ctrds.copy()
            self.clusters_best = self.clusters.copy()

        self.hist_soft.append(self.D_soft)
        self.hist_hard.append(self.D_hard)
        self.hist_best_hard.append(min(self.hist_hard))
        return None

    def main(self):
        self.update_da()
        self.calc_associations()
        self.calc_ctrds()
        self.update_hist()
        return None

    def plt_train(self, dur):
        if self.interactive == 1:
            plt.scatter(range(len(self.hist_soft)), self.hist_soft, marker='s', c='c', edgecolors='black')
            plt.plot(self.hist_soft, label = 'Non-projected', c='c', linestyle = '--')
            plt.plot(self.hist_best_hard, label = 'Projected (best)', c = 'm', linestyle ='-')
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':', alpha=0.4)
            plt.grid(which='major', linestyle='--', alpha=0.9)
            plt.title('ECP-LL, gamma={0}, N={1}'.format(self.gamma, self.N))
            plt.xlabel('Iteration #')
            plt.ylabel('Objective function value')
            plt.legend()
            plt.pause(dur)
            plt.savefig("ProjectedVsNonProjected")
            plt.clf()

        elif self.interactive == 2:
            colors = cm.rainbow(np.linspace(0, 1, self.m))
            ctr = 0
            for ctrd, ctrd_soft, cluster, color in zip(self.ctrds, self.y, self.clusters, colors):
                ctr += 1
                if len(cluster) != 0:
                    cluster = np.array(cluster)
                    x_cord, y_cord = cluster[:, 0], cluster[:, 1]
                    plt.scatter(x_cord, y_cord, c=[color], edgecolors=None, s = 30, alpha= 0.15, label = 'Cluster {0}'.format(ctr))
                ctrds_soft_x_cord, ctrds_soft_y_cord = ctrd_soft[0], ctrd_soft[1]
                plt.scatter(ctrds_soft_x_cord, ctrds_soft_y_cord, marker="s", edgecolors='black', c=[color], alpha= 1)
                ctrds_x_cord, ctrds_y_cord = ctrd[0], ctrd[1]
                plt.scatter(ctrds_x_cord, ctrds_y_cord, marker="v", edgecolors='black', c=[color], alpha= 1)
            plt.legend()
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':', alpha=0.4)
            plt.grid(which='major', linestyle='--', alpha=0.9)
            plt.title('ECP-LL Clustering, gamma={0}, N={1}'.format(self.gamma, self.N))
            plt.xlabel('x coordinate')
            plt.ylabel('y coordinate')
            plt.pause(dur)
            plt.clf()

        elif self.interactive == 3 and self.ctr != 1:
            colors = cm.rainbow(np.linspace(0, 1, self.m))
            ctr = 0
            for ctrd, cluster, color in zip(self.ctrds_best, self.clusters_best, colors):
                ctr += 1
                if len(cluster) != 0:
                    cluster = np.array(cluster)
                    x_cord, y_cord = cluster[:, 0], cluster[:, 1]
                    plt.scatter(x_cord, y_cord, c=[color], edgecolors=None, s = 30, alpha= 0.15, label = 'Cluster {0}'.format(ctr))
                ctrd_x_cord, ctrd_y_cord = ctrd[0], ctrd[1]
                plt.scatter(ctrd_x_cord, ctrd_y_cord, marker="o", edgecolors='black', c=[color], alpha= 1)
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':', alpha=0.4)
            plt.grid(which='major', linestyle='--', alpha=0.9)
            plt.title('ECP-LL Clustering, gamma={0}, N={1}'.format(self.gamma, self.N))
            plt.xlabel('x coordinate')
            plt.ylabel('y coordinate')
            plt.legend()
            plt.pause(dur)
            plt.clf()
        elif self.interactive == 4:
            plt.step(self.beta_hist, self.m_hist, where = 'post')
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':', alpha=0.4)
            plt.grid(which='major', linestyle='--', alpha=0.9)
            plt.title('ECP-LL Clustering Phase Transition, gamma={0}, N={1}'.format(self.gamma, self.N))
            plt.xlabel(r'$\beta = \frac{1}{T}$')
            plt.ylabel('# of Centroids')
            plt.xticks(self.beta_hist)
            for xc in self.beta_hist:
                plt.axvline(x = xc, ls = ':', c = 'r')
            plt.pause(dur)
            plt.clf()
        return None

    def train(self):
        plt.ion()
        while(self.T > self.T_min):
            #initialize starting critical temperature
            self.ctr += 1
            if self.ctr == 1:
                self.T = self.init_crtl_T() if self.t_start == None else self.t_start
            try:
                self.main()
            except OverflowError:
                break;

            if self.interactive in (1,2,3,4):
                self.plt_train(0.01)
                print('''
                ****************************
                Iteration {0} in progress...
                Temperature: {1}
                Cost (Soft): {2}
                Cost best (hard): {3}
                ****************************
                '''.format(self.ctr, self.T, self.D_soft, self.hist_best_hard[-1]))

            self.T *= self.alpha
            if self.m < self.max_m:
                self.purturb_ctrds()
#         self.plt_train(inf)
        np.save('results.npy', {'controller_locs':self.ctrds_best,
                                'clusters':self.clusters_best})
        print('Results saved to current directory.')
        return None
