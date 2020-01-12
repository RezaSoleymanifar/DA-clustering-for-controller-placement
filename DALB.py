# %load DALB.py
import numpy as np
from math import *
from matplotlib import pyplot as plt
import matplotlib.cm as cm
# from scipy.interpolate import spline


class DA:
    def __init__(self, data, max_m, gamma, t_start = None, t_min = 1e-5, alpha = 0.9, interactive = None):
        """__init__

        :param data:matrix with data as rows
        :param max_m:max number of clusters
        :param t_start:start temperature
        :param t_min:final temperature
        :param gamma:controller synch importance coeff.
        :param alpha:temperature damp ratio
        :param interactive:1 plots obj. val., 2 plots current sol., 3: best sol.
        """
        self.N, self.d = np.shape(data)
        self.x = self.__pre_process(data)
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

        self.d_xy = None
        self.d_yy = None

        self.p_yx = None
        self.Z = None
        self.p_y = None
        self.p_xy = None

        self.ctr = 0

    def __pre_process(self, data):
        """__pre_process

        :param data: matrix
        """
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
        self.d_xy = {(i, j): self.sed(self.x[i], self.y[j]) for i in range(self.N) for j in range(self.m)}
        self.d_yy = {(j, j_): self.sed(self.y[j], self.y[j_]) for j in range(self.m) for j_ in range(self.m)}
        return None

    def calc_associations(self):
        reg = self.d_xy[max(self.d_xy, key=self.d_xy.get)] / self.T
        self.p_yx = {(i, j): exp(reg - self.d_xy[(i, j)] / self.T) for i in range(self.N) for j in range(self.m)}
        self.normalize_p()
        self.p_y = [sum(self.p_yx[(i, j)] * self.p_x for i in range(self.N)) for j in range(self.m)]
        self.p_xy = {(i, j): (self.p_yx[(i, j)] * self.p_x) / self.p_y[j] for i in range(self.N) for j in range(self.m)}
        # self.C = [sum(self.p_xy[(i, j)] * self.x[i] for i in range(self.N)) for j in range(self.m)]

    def normalize_p(self):
        self.Z = [sum(self.p_yx[(i, j)] for j in range(self.m)) for i in range(self.N)]
        self.p_yx = {(i, j): self.p_yx[(i, j)] / self.Z[i] for i in range(self.N) for j in range(self.m)}
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
        temp = [sum(self.d_yy[(j, j_)] for j_ in range(self.m)) for j in range(self.m)]
        self.j_s = np.argmin(temp)
        if self.m == 1:
            self.y[0] = (self.gamma * self.N * sum(self.p_yx[(i, self.j_s)] * self.x[i] for i in range(self.N)))\
                        / ( (self.m -1) * self.gamma * self.N + sum(self.p_yx[(i, self.j_s)] for i in range(self.N)))
        else:
            temp_y = self.y.copy()
            self.y = [(self.gamma * self.N * self.y[self.j_s] + sum(self.p_yx[(i,j)] * self.x[i] for i in range(self.N)))
                      / (self.gamma * self.N + sum(self.p_yx[(i, j)] for i in range(self.N))) for j in range(self.m)
                      if j != self.j_s]
            y_j_s = (self.gamma * self.N * sum(temp_y[j_] for j_ in range(self.m) if j_ != self.j_s)\
                    + sum(self.p_yx[(i, self.j_s)] * self.x[i] for i in range(self.N)))\
                    / ((self.m - 1) * self.gamma * self.N + sum(self.p_yx[(i, self.j_s)] for i in range(self.N)))
            self.y.insert(self.j_s, y_j_s)
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

        D_hard = sum(self.chk(idx[i], j) * self.sed(self.x[i], ctrds[j]) for i in range(self.N) for j in range(self.m)) \
                 + self.gamma * sum(self.N * self.d_yy[(self.j_s, j_)] for j_ in range(self.m))
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

        self.D_soft = sum(self.p_yx[(i,j)] * self.sed(self.x[i], self.y[j]) for i in range(self.N) for j in range(self.m)) \
                 + self.gamma * sum(self.N * self.d_yy[(self.j_s, j_)] for j_ in range(self.m))
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
            plt.plot(self.hist_best_hard, label = 'Projected (Best)', c = 'm', linestyle ='-')
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':', alpha=0.4)
            plt.grid(which='major', linestyle='--', alpha=0.9)
            plt.title('ECP-LB, gamma={0}, N={1}'.format(self.gamma, self.N))
            plt.xlabel('Iteration #')
            plt.ylabel('Objective function value')
            plt.legend()
            plt.pause(dur)
#             plt.savefig("ProjectedVsNonProjected.pdf")
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

                if ctr - 1 == self.j_s:
                    plt.scatter(ctrds_soft_x_cord, ctrds_soft_y_cord, marker="s", edgecolors='black', c=[color],
                                alpha=1, s= 100)
                    ctrds_x_cord, ctrds_y_cord = ctrd[0], ctrd[1]
                    plt.scatter(ctrds_x_cord, ctrds_y_cord, marker="v", edgecolors='black', c=[color], alpha=1, s= 100)
                else:
                    plt.scatter(ctrds_soft_x_cord, ctrds_soft_y_cord, marker="s", edgecolors='black', c=[color], alpha= 1)
                    ctrds_x_cord, ctrds_y_cord = ctrd[0], ctrd[1]
                    plt.scatter(ctrds_x_cord, ctrds_y_cord, marker="v", edgecolors='black', c=[color], alpha= 1)
            plt.legend()
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':', alpha=0.4)
            plt.grid(which='major', linestyle='--', alpha=0.9)
            plt.title('ECP-LB Clustering, gamma={0}, N={1}'.format(self.gamma, self.N))
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
                if ctr - 1 == self.j_s:
                    ctrd_x_cord, ctrd_y_cord = ctrd[0], ctrd[1]
                    plt.scatter(ctrd_x_cord, ctrd_y_cord, marker="o", edgecolors='black', c=[color], alpha= 1, s= 100)
                else: ctrd_x_cord, ctrd_y_cord = ctrd[0], ctrd[1]
                plt.scatter(ctrd_x_cord, ctrd_y_cord, marker="o",
                            edgecolors='black', c=[color], alpha=1)
            plt.minorticks_on()
            plt.grid(which='minor', linestyle=':', alpha=0.4)
            plt.grid(which='major', linestyle='--', alpha=0.9)
            plt.title('ECP-LB Clustering, gamma={0}, N={1}'.format(self.gamma, self.N))
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
        # self.plt_train(inf)
        np.save('results.npy', {'controller_locs':self.ctrds_best,
                                'clusters':self.clusters_best, 'leader_index':self.j_s})
        print('Results saved to current directory.')
        return None
