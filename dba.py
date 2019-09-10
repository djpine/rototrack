# 12062018
# FLAT_ASPECT_RATIO = 4.0 (previous 4.3)

# 03132019
# analyze the rotation in the particle frame
# rotationGenerator when theta == 0, return identical matrix

#05132019
# after caompare SEM sizing and the optic tracking  measured length,
# the DIAM is average determined as 0.65 um (1.30 /2 by SEM)
# BLUR is determined as 0.29 (1.59 - 1.30)


import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import glob
from scipy import stats
from scipy import linalg
import pandas as pd
import sys
import pickle
sys.path.append('/Users/wenhaizheng/Dropbox/ExpData/tools/')

BLUR = 0.29  # box_size = int(3 * diam/mpp) change to 0.29 12162018
BLUR_TOP = 0.36
DIAM = 0.65
FLAT_ASPECT_RATIO = 4.0


def rotationGenerator(axis, theta):
    """Generate a rotation matrix which rotate a vector theta about axis"""
    """reference:https://en.wikipedia.org/wiki/Rotation_matrix, rotaion matrix
    from axis and angle"""
    "axis is a 3d vector"
    if theta != 0:
        axis = axis / linalg.norm(axis)
        u = np.array([[0, -axis[2], axis[1]],
                      [axis[2], 0, -axis[0]],
                      [-axis[1], axis[0], 0]])
        return linalg.expm(u * theta)
    else:
        return np.eye(3)


def _plot_projection_PDF(center_dist, diam=DIAM):
    plt.close()
    fig, ax = plt.subplots(figsize=(8, 6))
    edge_width = 0.01
    count, binEdge = np.histogram(
        center_dist/diam, bins=int((1.1)/edge_width),
        range=(0, 1.2))
    xfit = np.linspace(0, diam - 0.5 * edge_width, 100)/diam
    # The upper bound of xfit is slightly smaller than 1
    yfit = np.tan(np.arcsin(xfit))
    ax.bar(binEdge[1:] - edge_width / 2.0,
           count * 1.0 / np.sum(count) / edge_width,
           width=edge_width)
    ax.plot(xfit, yfit, 'r-')  # The fit,
    ax.text(0.2, 0.3, r"$d$ is {0:.2g}".format(diam) + r"$\mathrm{\mu m}$",
            fontsize=16, transform=ax.transAxes)
    ax.vlines(1.0, 0.0, 7.0, linestyles='dashed')
    ax.set_xlabel(r"$l_{sub} / d$", fontsize=16)
    ax.set_ylabel(r"Probability Density", fontsize=16)
    return fig


def _msd_direct_array(x, lag_max=None):
    """
    x is ndarray"""
    if not lag_max:
        lag_max = len(x) // 10
    msds = np.zeros(lag_max + 1)
    for i in range(lag_max + 1):
        diffs = x[i:] - x[:len(x) - i]
        msds[i] = (diffs ** 2).mean()
    return msds


def msd_direct(x, lag_max=None):
    """
    x is pd.Series"""
    if not lag_max:
        lag_max = len(x) // 10
    if isinstance(x, pd.Series):
        msds = np.zeros(lag_max + 1)
        for i in range(lag_max + 1):
            msds[i] = ((x.shift(i) - x) ** 2).mean()
        return msds
    else:
        return _msd_direct_array(x, lag_max)


def single_file_analysis(data_folder):
    df, fps, name = read_one_folder(data_folder)


def read_one_folder(data_folder):
    """
    Example of info:
    ['001', '8001frames', 'particle1', '40fps', 'v7', 'data']
    """
    df = pd.read_csv(data_folder + "/" + "tracking_data.csv", index_col=0)
    metadata = pd.read_csv(
        data_folder + "/" + "metadata.csv", index_col=0, squeeze=True)
    info = data_folder.split("_")
    fps = int(info[3][0:-3])
    name = info[0] if info[2][8:] == "1" else info[0] + "_" + info[2][8:]
    return df, fps, name, metadata


def _plot_orientation_on_sphere(orientations, view_angles, fig_name="foo"):
    axes_num = len(view_angles) + 1
    fig = plt.figure(figsize=(8 * axes_num, 8))
    axes = [fig.add_axes(
        [i / axes_num, 0.0, 1.0 / axes_num, 1.0], projection='3d')
        for i in range(axes_num)]
    phi = np.linspace(0, np.pi, 100)
    theta = np.linspace(0, 2*np.pi, 100)
    phi, theta = np.meshgrid(phi, theta)

    x = np.sin(phi) * np.cos(theta)
    y = np.sin(phi) * np.sin(theta)
    z = np.cos(phi)
    n = len(orientations)
    t = 255 * np.linspace(1, n, n) / n

    def _single_axes(ax, set_viewpoint=False, view_angle=None):
        """
        view_angle : (elev, azim)
        """
        ax.set_aspect('equal')
        ax.plot_wireframe(x, y, z, rstride=3, cstride=3, alpha=0.3)
        ax.plot_surface(x, y, z,  rstride=3, cstride=3, alpha=0.3)
        ax.set_axis_off()
        ax.scatter(orientations[:, 0], orientations[:, 1], orientations[:, 2],
                   c=t, cmap='jet')
        if set_viewpoint:
            ax.view_init(view_angle[0], view_angle[1])
        ax.set_title(r"azim = {0:d}$\degree$ , elev = {1:d}$\degree$".format(
            ax.azim, ax.elev), fontsize=14)
        return

    _single_axes(axes[0])
    for i, ax in enumerate(axes[1:]):
        _single_axes(axes[i + 1], set_viewpoint=True,
                     view_angle=view_angles[i])
    plt.savefig(fig_name + '.jpg')
    plt.close()
    return


def _plot_msad(data, plot_max=None, fit_short_time=False):
    fig, ax = plt.subplots()
    axes = ["t", "x", "y", "z"]
    if plot_max:
        plot_max = min(plot_max, len(data))
    else:
        plot_max = len(data)
    x = data[1:plot_max, 0]
    for i in range(1, 4):
        ax.plot(x, data[1:plot_max, i], 'o', label=axes[i])
    y = data[1:plot_max, 1:].sum(axis=1)
    ax.plot(x, y, 'o', label="total")
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.legend()
    ax.set_xlabel(r"$\tau \ \mathrm{[s]}$", fontsize=16)
    ax.set_ylabel(r"$MSAD \ \mathrm{[rad^2]}$", fontsize=16)
    plt.tight_layout()
    if fit_short_time:
        k, b, fit_result = short_time_fit(x, y)
        yfit = k * x + b
        d_r = k / 4
        ax.plot(x, yfit, 'k--',
                label="$D_r$ = {0:.2g}".format(d_r) + r' $\mathrm{s^{-1}}$')
        ax.legend()
        return fig, d_r, b, fit_result
    else:
        return fig


def short_time_fit(x, y, fit_max=10):
    fit_result = pd.DataFrame(
        [], columns=["slope", "intercept", "rvalue", "pvalue", "stderr"])
    fit_max = min(fit_max, len(x))
    for lag in range(2, fit_max):
        fit_result.loc[lag] = stats.linregress(x[:lag], y[:lag])
    idx = fit_result.slope.idxmax()
    k, b = fit_result.slope[idx], fit_result.intercept[idx]
    return k, b, fit_result


def _short_time_fit_and_plot(ax, x, y, scale_coef=1,
                             data_label='', fit_label='', fit_units=''):
    k, b, fit_result = short_time_fit(x, y)
    yfit = k * x + b
    line = ax.plot(x, y, 'o', markersize=4, label=data_label)
    scaled_k = k * scale_coef
    ax.plot(x, yfit, '--', markersize=4, color=line[0]._color,
            label=fit_label + "= {0:.2g}".format(scaled_k) + fit_units)
    return scaled_k, b, fit_result


def _msd_axis_setting(ax):
    ax.legend(fontsize=12)
    ax.set_xlim(0)
    ax.set_ylim(0)
    ax.set_xlabel(r"$\tau \ \mathrm{[s]}$", fontsize=16)
    ax.set_ylabel(r"$MSD \ \mathrm{[\mu m ^2]}$", fontsize=16)
    plt.tight_layout()


def _puu_axis_setting(ax, xlim=0, ylim=0):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    plt.tight_layout()


def _calculate_diam(data_df):
    if (data_df["inertia_ratio"] > FLAT_ASPECT_RATIO).any():
        diam = (data_df[data_df["inertia_ratio"] > FLAT_ASPECT_RATIO
                        ]["length[um]"].mean() - BLUR) / 2
    elif (data_df["inertia_ratio"] < 1.1).any():
        diam = data_df[data_df["inertia_ratio"] < 1.1
                       ]["length[um]"].min() - BLUR_TOP
    else:
        diam = DIAM
    if abs(diam - DIAM) > (0.05 * DIAM):
        diam = DIAM
    print("diameter is {0:.3f}".format(diam))
    return diam


class TrackingResult:
    def __init__(self, data_folder, lag_max=None):

        self.blur = BLUR
        self.data_folder = data_folder
        self.stats = {}
        self.stats_detail = {}
        self.data, self.fps, self.name, self.metadata = read_one_folder(
            self.data_folder)
        self.mpp = round(float(self.metadata["calibration_um"]), 3)
        self.orientation_axis = ["uzx", "uzy", "uzz"]
        self.view_angles = ((0, 0), (90, 0))
        if not lag_max:
            if len(self.data) > 8001:
                self.lag_max = len(self.data) // 10
            else:
                self.lag_max = len(self.data) // 2
        else:
            self.lag_max = lag_max
        self.lag_t_stats = pd.DataFrame(
            [], columns=['lag_time[s]', 'x_msd[um^2]', 'y_msd[um^2]',
                         'x_msd[um^2]_err', 'y_msd[um^2]_err'])
        self.lag_t_stats['lag_time[s]'] = np.arange(
            self.lag_max + 1) / self.fps
        print(self.name + ":")
        self.calculate_diam()
        self.calulate_center_distance()
        self.plot_projection_PDF()
        self.calculate_msd()
        self.calculate_theta()
        self.head_tail_correction()
        self.angular_displacement()
        self.calculate_msad()
        self.orientation_correlation()
        self.lag_t_stats.to_csv(self.data_folder + "/" +
                                "lag_t_stats.csv", index=False)
        self.plot_msad()
        self.plot_short_time_msd()
        self.plot_long_time_msd()
        self.plot_orientation_on_sphere()
        self.plot_autocorrelations()
        self.plot_long_time_autocorrelations()
        self.save_stats()
        return

    def calculate_diam(self):
        self.diam = _calculate_diam(self.data)
        return

    def calulate_center_distance(self):
        self.data["center_distance[um]"] = np.clip(self.data["length[um]"] -
                                                   self.blur - self.diam,
                                                   0, self.diam)
        return

    def plot_projection_PDF(self):
        fig = _plot_projection_PDF(
            self.data["center_distance[um]"].values, self.diam)
        fig.savefig(self.data_folder + "/projection_pdf.jpg")
        plt.close()

    def calculate_theta(self):
        sin_theta = self.data["center_distance[um]"] / self.diam
        self.data["theta[rad]"] = np.arcsin(sin_theta)
        self.data["uzx"] = sin_theta * np.cos(self.data["phi[rad]"])  # x
        self.data["uzy"] = sin_theta * np.sin(self.data["phi[rad]"])  # y
        self.data["uzz"] = np.sqrt(1 - sin_theta ** 2)  # np.abs(z)
        return

    def calculate_msd(self):
        """
        calculate the translational mean square dispacement
        """
        self.lag_t_stats["x_msd[um^2]"] = msd_direct(
            self.data["x[pix]"].values * self.mpp, self.lag_max)
        self.lag_t_stats["y_msd[um^2]"] = msd_direct(
            self.data["y[pix]"].values * self.mpp, self.lag_max)
        return

    def calculate_msad(self):
        """
        calculate angular mean square  dispacement
        """
        for comp in self.ad_components:
            self.lag_t_stats[comp[0:-5] + "_msd[rad^2]"] = msd_direct(
                self.data[comp].values, self.lag_max)
        self.lag_t_stats["dphi_x_p_msd[rad^2]"] = msd_direct(
            self.data["phi_x_particle[rad]"].values, self.lag_max)
        self.lag_t_stats["dphi_y_p_msd[rad^2]"] = msd_direct(
            self.data["phi_y_particle[rad]"].values, self.lag_max)
        return

    def head_tail_correction(self):
        """
        choose the closer jump
        """
        for i in self.data.index[1:]:
            if np.inner(self.data.loc[i, self.orientation_axis],
                        self.data.loc[i - 1, self.orientation_axis]) < 0:
                self.data.loc[i, self.orientation_axis
                              ] = -self.data.loc[i, self.orientation_axis]
                self.data.loc[i, "theta[rad]"
                              ] = np.pi - self.data.loc[i, "theta[rad]"]
                self.data.loc[i, "phi[rad]"
                              ] = np.mod(np.pi + self.data.loc[i, "phi[rad]"],
                                         2 * np.pi)

        return

    def angular_displacement(self):
        dphi_lab = np.cross(self.data[self.orientation_axis].iloc[:-1].values,
                            self.data[self.orientation_axis].iloc[1:].values)
        dphi_norm = np.linalg.norm(dphi_lab, axis=1)
        self.ad_components = ["phi_x[rad]", "phi_y[rad]", "phi_z[rad]"]
        self.data = self.data.join(pd.DataFrame(np.concatenate(
            (np.array([[0, 0, 0]]),
             np.cumsum(dphi_lab, axis=0))),
            columns=self.ad_components)
        )
        divisor = dphi_norm.reshape((-1, 1))
        rotation_axis = np.divide(dphi_lab, divisor, where=(divisor != 0))
        rotation_matrices = (rotationGenerator(rotation_axis[i], dphi_norm[i])
                             for i in range(len(dphi_norm)))
        uz0 = self.data[self.orientation_axis].iloc[0].values
        dphi0 = np.cross(np.array([0, 0, 1]), uz0)
        m0 = rotationGenerator(dphi0 / np.linalg.norm(dphi0),
                               np.linalg.norm(dphi0))
        ux = np.dot(m0, np.array([1, 0, 0])).reshape((1, 3))
        uy = np.dot(m0, np.array([0, 1, 0])).reshape((1, 3))
        for m in rotation_matrices:
            ux = np.vstack((ux, np.dot(m, ux[-1])))
            uy = np.vstack((uy, np.dot(m, uy[-1])))
        dphi_x_particle = np.sum(ux[:-1] * dphi_lab, axis=1).cumsum()
        dphi_y_particle = np.sum(uy[:-1] * dphi_lab, axis=1).cumsum()
        self.data["phi_x_particle[rad]"] = np.concatenate(
            ([0], dphi_x_particle))
        self.data["phi_y_particle[rad]"] = np.concatenate(
            ([0], dphi_y_particle))
        return

    def orientation_correlation(self):
        orientation = self.data[self.orientation_axis].values
        p_uu = np.zeros((self.lag_max + 1, 2))

        def p2_Legendre(x):
            return 1/2.0 * (3 * x ** 2 - 1)

        for i in range(self.lag_max + 1):
            uu = np.sum(orientation[i:] * orientation[:len(orientation) - i],
                        axis=1)
            p_uu[i, 0] = uu.mean()
            p_uu[i, 1] = p2_Legendre(uu).mean()
        self.lag_t_stats = self.lag_t_stats.join(
            pd.DataFrame(p_uu, columns=["P1_uu", "P2_uu"]))
        return

    def plot_orientation_on_sphere(self):
        _plot_orientation_on_sphere(
            self.data[self.orientation_axis].values, self.view_angles,
            fig_name=self.data_folder + "/" +
            "orientation_distribution_on_sphere")

    def plot_short_time_msd(self, plot_max=10):
        fig, ax = plt.subplots()
        x = self.lag_t_stats["lag_time[s]"][1: plot_max].values
        for axis in ["x_msd[um^2]", "y_msd[um^2]"]:
            y = self.lag_t_stats[axis][1: plot_max].values
            k, b, fit_result = short_time_fit(x, y, fit_max=plot_max)
            yfit = k * x + b
            dt = k / 2
            line = ax.plot(x, y, 'o', markersize=4, label=axis[: -6])
            ax.plot(x, yfit, '--', markersize=4, color=line[0]._color,
                    label="$D_T$ = {0:.2g}".format(
                        dt) + r" $\mathrm{\mu m ^2 / s}$"
                    )
            self.stats[axis[:-6]+"_D_T[um^2/s]"] = dt
            self.stats[axis[:-6]+"_error[um^2]"] = b
            self.stats_detail[axis[:-6]+"_short"] = fit_result
        _msd_axis_setting(ax)
        plt.savefig(self.data_folder + "/msd_short_time.jpg")
        plt.close()

    def plot_long_time_msd(self):
        x = self.lag_t_stats["lag_time[s]"][1: self.lag_max].values
        for axis in ["x_msd[um^2]", "y_msd[um^2]"]:
            fig, ax = plt.subplots()
            y = self.lag_t_stats[axis][1: self.lag_max].values
            ax.plot(x, y, 'o', markersize=4, label=axis[: -6])
            self.stats[axis[:-6]+"_long_max"] = y.max()
            _msd_axis_setting(ax)
            plt.savefig(
                self.data_folder + "/" + axis[:2] + "msd_long_time.jpg")
            plt.close()

    def plot_msad(self):
        data = self.lag_t_stats[
            ["lag_time[s]", "phi_x_msd[rad^2]", "phi_y_msd[rad^2]",
             "phi_z_msd[rad^2]"]].values
        fig, *tmp = _plot_msad(data, plot_max=10, fit_short_time=True)
        self._fit_parameters_assignment(
            tmp, "D_R(msad)[s^-1]", "u_error(msad)", "msad")
        fig.savefig(self.data_folder + "/" + "short_time_msad.jpg")
        fig = _plot_msad(data)
        fig.savefig(self.data_folder + "/" + "long_time_msad.jpg")
        plt.close()
        return

    def plot_autocorrelations(self):
        fig, ax = plt.subplots()
        x = self.lag_t_stats["lag_time[s]"][1: self.lag_max].values
        y1 = -np.log(self.lag_t_stats["P1_uu"][1: self.lag_max].values) / 2
        y2 = -np.log(self.lag_t_stats["P2_uu"][1: self.lag_max].values) / 6
        tmp = _short_time_fit_and_plot(
            ax, x, y1, data_label=r"$-\ \frac{1}{2}\ \ln P_1$",
            fit_label="$D^{P_1}_R$", fit_units=r'$ \mathrm{s^{-1}}$')
        self._fit_parameters_assignment(
            tmp, "D_R(P1)[s^-1]", "u_error(P1)", "P1")
        tmp = _short_time_fit_and_plot(
            ax, x, y2, data_label=r"$-\ \frac{1}{6}\ \ln P_2 $",
            fit_label="$D^{P_2}_R$", fit_units=r' $\mathrm{s^{-1}}$')
        self._fit_parameters_assignment(
            tmp, "D_R(P2)[s^-1]", "u_error(P2)", "P2")
        ax.legend(fontsize=14)
        ax.set_xlabel(r"$\tau \ \mathrm{[s]}$", fontsize=16)
        _puu_axis_setting(ax, ylim=(0, np.nanmax((2, 1.2 * y2[-1]))))
        plt.savefig(self.data_folder + "/puu_long_time.jpg")
        _puu_axis_setting(ax, xlim=(0, x[10]), ylim=(0, 1.2 * y2[10]))
        plt.savefig(self.data_folder + "/puu_short_time.jpg")
        plt.close()

    def plot_long_time_autocorrelations(self):
        fig, ax = plt.subplots()
        x = self.lag_t_stats["lag_time[s]"][1: self.lag_max].values
        y1 = self.lag_t_stats["P1_uu"][1: self.lag_max].values
        y2 = self.lag_t_stats["P2_uu"][1: self.lag_max].values
        self.stats["P1_min"], self.stats["P2_min"] = y1.min(), y2.min()
        ax.plot(x, y1, 'o', markersize=4, label="$P_1$")
        ax.plot(x, y2, 'd', markersize=4, label="$P_2 $")
        ax.legend(fontsize=14)
        _puu_axis_setting(ax)
        plt.savefig(self.data_folder + "/puu_long_time_linear.jpg")
        plt.close()

    def _fit_parameters_assignment(self, tmp, scaled_k_name="", b_name="",
                                   fit_result_name=""):
        self.stats[scaled_k_name] = tmp[0]
        self.stats[b_name] = tmp[1]
        self.stats_detail[fit_result_name] = tmp[2]

    def save_stats(self):
        pd.Series(self.stats).to_csv(self.data_folder + "/" + "stats.csv")
        for item in self.stats_detail:
            self.stats_detail[item].to_csv(
                self.data_folder + "/" + item + ".csv")


def analysis_all_files(lag_max=None):
    df_stats = pd.DataFrame(
        [], columns=[
            'D_R(msad)[s^-1]', 'D_R(P2)[s^-1]', 'D_R(P1)[s^-1]',
            'u_error(msad)', 'u_error(P2)', 'u_error(P1)',
            'x_msd_D_T[um^2/s]', 'x_msd_error[um^2]', 'y_msd_D_T[um^2/s]',
            'y_msd_error[um^2]', 'x_msd_long_max', 'y_msd_long_max',
            "P1_min", "P2_min"])
    folders = glob.glob("*frames*particle*fps*_data")
    caches = {}
    for folder in folders:
        t = TrackingResult(folder, lag_max)
        caches[t.name], df_stats.loc[t.name] = t, t.stats
    return caches, df_stats


if __name__ == "__main__":
    caches, df_stats = analysis_all_files()
    with open('caches.pickle', 'wb') as f:
        pickle.dump(caches, f, protocol=pickle.HIGHEST_PROTOCOL)
    df_stats.to_csv("stats.csv")
