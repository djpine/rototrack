import numpy as np
import pandas as pd
import glob
import os
from scipy import ndimage
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import pims
from skimage.morphology import convex_hull_image
from skimage.measure import regionprops, profile_line

"""
# 04212017
# add moving window feature to track the probes of drift

# 11022018
# improve the moving window function, when the bbox touch the roi
# boundary, re-center the particle
# v2: correct an error on: "frame_num" is referenced before assignment
# v3: reduce redundant print

# 11112018v8
# identify phi or phi + Pi by comparing the center intensity of the
# two parts

# 11122018v9
# For the binary feature c, perform binary_closing before fill holes.
# In this way, it's less likely to have hole

# 11122018v0
# In the pre_process, make larger box_size
# 2 * int(DIAM / SIZE_PER_PIXEL)  -> int(2.5 * DIAM / SIZE_PER_PIXEL)


# 01122019v3
# use the mean center intensity as the criterir to determine up or down direction
#

#01272919v4
# change threshold_8bit=5 (originally  threshold_8bit=2)
# change mass_list.sort(key=lambda x: x[3], reverse=True)
#   from mass_list.sort(key=lambda x: x[2], reverse=True)
"""


BIT_NUM = 12
DIAM = 0.63
SIZE_PER_PIXEL = 0.043
MINIMAL_MASS = int(np.pi * (DIAM / 2 / SIZE_PER_PIXEL) ** 2)
VERSION_NUMBER = 4
HALF_WINDOW_SIZE = 50


def show_and_select(pic, file_name="",
                    half_window_size=HALF_WINDOW_SIZE, color_list=None):
    """
    Display the picture(usually the 1st frame in a movie). Let the user
    choose which particles to track.

    Parameters
    ----------
    pic : ndarray

    file_name : str
        the name of the file where pic is from

    half_window_size : int
        half size of the window which contains a single particle

    color_list : list of str
        when there are multiple color particles(which represent
        different aspect ratios)
    """
    window_centers_xy = pd.DataFrame([], columns=["x", "y", "color"])
    if half_window_size >= min(pic.shape) // 2:
        window_centers_xy.loc[1] = {"x": pic.shape[1] // 2,
                                    "y": pic.shape[0] // 2,
                                    "color": "g"}
    else:
        fig = plt.figure(figsize=(10, 10))
        ax = fig.add_subplot(111)
        ax.set_title(
            "Select the particles you want to track(Press 'd' when finished)")
        ax.imshow(pic, cmap='gray')
        fig.canvas.mpl_connect('button_press_event',
                               lambda event: onclick(
                                   event, window_centers_xy, pic.shape, fig,
                                   ax, half_window_size, color_list)
                               )
        fig.canvas.mpl_connect(
            'key_press_event', lambda event: on_key(event, file_name))
        plt.show()
    window_centers_xy.to_csv(file_name + "_window_centers.csv")
    return window_centers_xy


def boundary_correction1d(left, window_size, fullsize, boundary_tolerance=0):
    if left < boundary_tolerance:
        left = boundary_tolerance
    elif left + window_size > fullsize - boundary_tolerance:
        left = fullsize - window_size - boundary_tolerance
    return left


def boundary_correction(left_top, window_size, fullsize, boundary_tolerance=0):
    """
    window_size[i] should be smaller than fullsize[i]
    """
    if not hasattr(window_size, '__iter__'):
        window_size = (window_size, ) * len(left_top)
    return tuple(boundary_correction1d(left_top[i], window_size[i],
                                       fullsize[i], boundary_tolerance)
                 for i in range(len(left_top))
                 )


def on_key(event, file_name):
    if event.key == "d":
        print("Done")
        plt.savefig(file_name + "_selected_particles" + ".jpg")
        plt.close()
    else:
        print("You pressed ", event.key)
        print("Finish seletion?Press d")


def onclick(event, window_centers_xy, framesize, fig, ax,
            half_window_size=HALF_WINDOW_SIZE, color_list=None):
    left_top_rc = boundary_correction((int(event.ydata) - half_window_size,
                                       int(event.xdata) - half_window_size),
                                      2 * half_window_size,
                                      framesize)
    iy, ix = (left_top_rc[0] + half_window_size,
              left_top_rc[1] + half_window_size)
    ith_choice = len(window_centers_xy) + 1
    print("{0:d} : x = {1:d}, y = {2:d}".format(ith_choice, ix, iy))
    if not color_list:
        color = 'g'  # Given green color
    else:
        color = input(
            "Input the color of the particle:("+'/'.join(color_list)+")")
    ax.add_patch(patches.Rectangle((ix - half_window_size,
                                    iy - half_window_size),
                                   2 * half_window_size, 2 * half_window_size,
                                   fill=False, color=color)
                 )

    ax.text(ix - half_window_size, iy - half_window_size // 2,
            "{0:0d}".format(ith_choice), fontsize=half_window_size // 4)
    fig.canvas.draw()
    window_centers_xy.loc[ith_choice] = {"x": ix, "y": iy, "color": color}
    return


def pre_process(image, box_size, sigma=1, bit_num=BIT_NUM,
                inverse_image=True):
    """
    remove short wavelength noise and long wavelength background
    """
    if inverse_image:
        image = (2 ** bit_num - 1) - image
    background = ndimage.uniform_filter(np.array(image, dtype=np.float),
                                        size=box_size, mode="nearest")
    lowpass_image = ndimage.gaussian_filter(image,
                                            sigma=sigma, mode="nearest")

    result = lowpass_image - background
    return result


def find_the_target_label(regions, minimal_area=MINIMAL_MASS):
    """
    Parameters:
    ----------
    regions : skimage regionprop object
    """
    mass_list = [[region.label, region.area, region.moments[0, 0],
                  region.weighted_moments[0, 0]]
                 for region in regions if region.area > minimal_area]
    if len(mass_list) > 1:
        mass_list.sort(key=lambda x: x[3], reverse=True)
    return mass_list[0][0]


def locate_binary_particle(image, bit_num=BIT_NUM, threshold_8bit=5):
    threshold = (2 ** bit_num - 1) / 255 * threshold_8bit
    labeled_array, num_features = ndimage.label(
        np.where(image > threshold, image, 0))
    if num_features > 1:
        regions = regionprops(labeled_array, image)
        particle_label = find_the_target_label(regions)
        return np.where(labeled_array == particle_label, 1, 0)
    else:
        return labeled_array


def is_anomalous(length, dl, convexity, area):

    return (length > (2.5 * DIAM / SIZE_PER_PIXEL)) or (
        length < (DIAM / SIZE_PER_PIXEL)) or (
        dl > 3.0) or (convexity < 0.75) or (
        area > (DIAM ** 2) * np.pi) or (
        area < (DIAM ** 2) * np.pi / 4)


def enhanceContrstplot(ax, roi, c):
    # image_display = particle * c + (~c) * 2 ** bit_num
    pmax = (roi[c > 0]).max()
    pmin = (roi[c > 0]).min()
    ax.imshow(roi, cmap='gray', vmin=pmin,
              vmax=1.5 * pmax - 0.5 * pmin)
    return


def plotParticle(ax, roi, c, res, color_of_particle, text_info=True):
    """
    The scale is axisSize by axisSize, in the fig, x ranges from (0,axisSize)
    and y is from (axisSize,0)
    #     ^
    #  0  |
    #     |
    # y   |
    #     |
    #     |
    # 128 |
    #     ----------------->
    #     0                128
    #             x
    """
    xc, yc = res["center_rc"][::-1]
    phi = res["phi_xy[rad]"]
    axisSize = len(roi)
    scaleRatio = axisSize / 128.0
    ax.set_axis_off()
    ax.axis([0, axisSize, axisSize, 0])
    enhanceContrstplot(ax, roi, c)

    # reference line, xc = col_bar, yc =row_bar
    # ax.plot([xc, xc + 10 * scaleRatio],[yc, yc], 'k-', linewidth = 2)
    ax.plot([xc, xc + res["length_up[px]"] * np.cos(phi)],
            [yc, yc + res["length_up[px]"] * np.sin(phi)],
            'r-', linewidth=2)
    ax.plot([xc - res["length_down[px]"] * np.cos(phi), xc],
            [yc - res["length_down[px]"] * np.sin(phi), yc],
            'r--', linewidth=2)
    ax.scatter(xc, yc, s=80, c='r')
    ax.contour(c, 1, colors=color_of_particle)
    if text_info:
        add_text(ax, res, scaleRatio)
    return


def add_text(ax, res, scaleRatio):
    if res["anomalous"]:
        ax.scatter(120 * scaleRatio, 10 * scaleRatio,
                   s=120 * scaleRatio, c='r')
    ax.text(5.0 * scaleRatio, 72.0 * scaleRatio,
            r'$\phi = {0:8.1f}\degree$'.format(np.rad2deg(res["phi_xy[rad]"])),
            fontsize=14)
    ax.text(5.0 * scaleRatio, 82.0 * scaleRatio,
            '$l_P = $' + '${0:8.2f}$'.format(res["projection_length[um]"]) +
            r'$\mathrm{\mu m}$', fontsize=14)
    ax.text(5.0 * scaleRatio, 92.0 * scaleRatio,
            r'$\Delta l_P = {0:8.2f}\ $'.format(res["dl[pix]"]) +
            r"$\mathrm{pix}$", fontsize=14)
    ax.text(5.0 * scaleRatio, 102.0 * scaleRatio,
            r'$r_I = {0:8.1f}$'.format(res["inertia_ratio"]), fontsize=14)
    ax.text(5.0 * scaleRatio, 112.0 * scaleRatio,
            r'$Convx = {0:8.2f}$'.format(res["convexity"]), fontsize=14)
    ax.text(5.0 * scaleRatio, 122.0 * scaleRatio,
            '$Area = {0:8.2f}$'.format(res["area[um^2]"]) +
            r'$\mathrm{\mu m^2}$', fontsize=14)
    return


def single_frame(roi, ax, frame_num, color_of_particle='g',
                 length_per_pixel=SIZE_PER_PIXEL,
                 check_indicator=False,
                 inverse_image=True, sigma=1.0):

    b = pre_process(roi, box_size=int(2.5 * DIAM / SIZE_PER_PIXEL))
    c = locate_binary_particle(b, threshold_8bit=5)
    c = ndimage.binary_closing(c)  # 10122018
    c = ndimage.binary_fill_holes(c)
    binary_bbox, image_bbox, bbox_origin_rc = bbox_particle(c, b)
    binary_center_in_bbox = ndimage.center_of_mass(binary_bbox)
    image_center_in_bbox = ndimage.center_of_mass(image_bbox * binary_bbox)

    phi_center_rc, center_diff = phi_from_center_of_mass(image_center_in_bbox,
                                                         binary_center_in_bbox)

    center_rc = (binary_center_in_bbox[0] + bbox_origin_rc[0],
                 binary_center_in_bbox[1] + bbox_origin_rc[1])
    mesh_grid = position_grid(binary_bbox, binary_center_in_bbox)
    moments_binary_rc = moments(binary_bbox, mesh_grid)
    phi_moments_rc, inertia_ratio, sin_twophi = phi_from_moments(
        moments_binary_rc)
    if np.abs(phi_center_rc - phi_moments_rc) > np.pi / 2:
        phi_moments_rc += np.pi
    phi_rc = which_phi(phi_center_rc, phi_moments_rc, inertia_ratio)
    length_up, length_down = (
        sub_pixel_accuracy(b, c, center_rc, phi_rc),
        sub_pixel_accuracy(b, c, center_rc, phi_rc + np.pi)
    )
#    if center_intensity_diff(image_bbox, binary_center_in_bbox,
#                             phi_rc, length_up, length_down) < 0:
#        phi_rc = np.mod(phi_rc + np.pi, 2 * np.pi)
#        length_up, length_down = length_down, length_up
    projection_length = length_up + length_down
    area = moments_binary_rc[0, 0] * length_per_pixel ** 2
    dl_pix = np.abs(length_up - length_down)
    convexity = 1.0 * moments_binary_rc[0, 0] / convex_hull_image(
        binary_bbox).sum()

    res = {"phi_xy[rad]": phi_xy_to_phi_rc(phi_rc),
           "length[px]": projection_length,
           "length_up[px]": length_up, "length_down[px]": length_down,
           "dl[pix]": dl_pix,
           "center_rc": center_rc, "convexity":  convexity,
           "projection_length[um]": projection_length * length_per_pixel,
           "area[um^2]": area,
           "inertia_ratio": inertia_ratio, "sin_twophi": sin_twophi,
           "bbox_origin_rc": bbox_origin_rc, "bbox_shape": binary_bbox.shape,
           "anomalous": is_anomalous(projection_length, dl_pix,
                                     convexity, area),
           "frame_num": frame_num
           }
    if ax is not None:
        plotParticle(ax, roi, c, res, color_of_particle)
    return res


def next_roi_origin_rc(res, roi_size, roi_origin_rc, frame_shape):
    """
    if the bbox is close the the boundary, reselect roi_origin
    """
    new_bbox_origin_rc = boundary_correction(
        res["bbox_origin_rc"], res["bbox_shape"], (roi_size, roi_size),
        boundary_tolerance=int(DIAM / SIZE_PER_PIXEL)
    )
    if new_bbox_origin_rc != res["bbox_origin_rc"]:
        # recenter bbox
        bbox_center = (res["bbox_origin_rc"][0] + res["bbox_shape"][0] // 2,
                       res["bbox_origin_rc"][1] + res["bbox_shape"][1] // 2)
        proposed_roi_origin = tuple((roi_origin_rc[i] +
                                     bbox_center[i]
                                     - roi_size // 2) for i in range(2))
        new_roi_origin_rc = boundary_correction(
            proposed_roi_origin, roi_size, frame_shape, boundary_tolerance=0)
        return new_roi_origin_rc
    else:
        return roi_origin_rc


def update_tracking(df, f, res, roi_origin_rc):
    df.loc[f] = {"x[pix]": roi_origin_rc[1] + res["center_rc"][1],
                 "y[pix]": roi_origin_rc[0] + res["center_rc"][0],
                 "phi[rad]": res["phi_xy[rad]"],
                 "area[um^2]": res["area[um^2]"],
                 "inertia_ratio": res["inertia_ratio"],
                 "length[um]": res["projection_length[um]"],
                 "sin_twophi": res["sin_twophi"],
                 "dl[pix]": res["dl[pix]"],
                 "convexity": res["convexity"],
                 "anomalous": res["anomalous"],
                 "roi_origin_x[pix]": roi_origin_rc[1],
                 "roi_origin_y[pix]": roi_origin_rc[0]
                 }
    return


def single_window_tracking(frames, frame_num, roi_origin_rc, ith_window,
                           file_name, start_frame=0, color='g',
                           half_window_size=HALF_WINDOW_SIZE,
                           length_per_pixel=SIZE_PER_PIXEL,
                           plot_particle=True, sigma=1.0):
    """
        track ith particle(window)
    """

    df = pd.DataFrame([], columns=["x[pix]", "y[pix]", "phi[rad]",
                                   "area[um^2]", "inertia_ratio", "length[um]",
                                   "sin_twophi", "dl[pix]", "convexity",
                                   "anomalous", "roi_origin_x[pix]",
                                   "roi_origin_y[pix]"])
    frame_shape = frames[0].shape
    metadata = pd.Series(frames.metadata)
    fps = int(round(1000 * (frames.sizes['t'] - 1) / (
        frames[-1].metadata['t_ms'] - frames[0].metadata['t_ms']),
        0))
    pic_folder = file_name.split(
        ".")[0] + "_{0:d}frames_particle{1:d}_{2:d}fps_v{3:d}".format(
        frame_num, ith_window, fps, VERSION_NUMBER)
    if start_frame != 0:
        pic_folder += "_start{0:d}".format(start_frame)
    if plot_particle and (not os.path.isdir(pic_folder)):
        os.makedirs(pic_folder)
    for f in range(start_frame, start_frame + frame_num):
        if plot_particle:
            fig = plt.figure(figsize=(3, 3), frameon=False)
            ax = fig.add_axes([0.0, 0.0, 1.0, 1.0])
        else:
            ax = None
        roi = frames[f][roi_origin_rc[0]: roi_origin_rc[0] +
                        2 * half_window_size,
                        roi_origin_rc[1]: roi_origin_rc[1] +
                        2 * half_window_size]
        tracking_result = single_frame(roi, ax, f, color_of_particle=color)
        update_tracking(df, f, tracking_result, roi_origin_rc)

        roi_origin_rc = next_roi_origin_rc(
            tracking_result, 2 * half_window_size, roi_origin_rc, frame_shape)
        if plot_particle:
            plt.savefig(pic_folder + "/{0:0d}.jpg".format(f))
            plt.close()
    data_folder = pic_folder+'_data'
    if not os.path.isdir(data_folder):
        os.makedirs(data_folder)

    metadata.to_csv(data_folder + "/" + "metadata.csv")
    df.to_csv(data_folder + "/" + "tracking_data.csv", index_label="frames")
    return


def single_file_tracking(file_name, window_centers_xy=None,
                         half_window_size=HALF_WINDOW_SIZE, start_frame=0,
                         length_per_pixel=SIZE_PER_PIXEL, set_frame_num=None,
                         plot_particle=True, sigma=1.0, single_particle=False):
    """Tracking particles in a single file"""
    frames = pims.ND2_Reader(file_name)
    frame_shape = frames[0].shape
    if not set_frame_num:
        frame_num = frames.sizes['t'] - start_frame
    else:
        frame_num = min(set_frame_num, frames.sizes['t'] - start_frame)
    if window_centers_xy is None:
        if single_particle:
            window_centers_xy = pd.DataFrame([], columns=["x", "y", "color"])
            window_centers_xy.loc[1] = {"x": frame_shape[1] // 2,
                                        "y": frame_shape[0] // 2,
                                        "color": "g"}
        else:
            window_centers_xy = show_and_select(
                frames[0], file_name=file_name,
                half_window_size=half_window_size)

    for i in window_centers_xy.index:
        color = window_centers_xy.loc[i, "color"]
        roi_origin_rc = (window_centers_xy.loc[i, "y"] - half_window_size,
                         window_centers_xy.loc[i, "x"] - half_window_size)
        single_window_tracking(frames, frame_num, roi_origin_rc, i, file_name,
                               start_frame, color=color,
                               half_window_size=half_window_size,
                               length_per_pixel=length_per_pixel,
                               plot_particle=plot_particle, sigma=sigma)

    return


def phi_from_center_of_mass(weighted_center_rc, binary_center_rc):
    #
    orient_rc = np.array(weighted_center_rc) - np.array(binary_center_rc)
    center_difference = np.linalg.norm(orient_rc)
    orient_rc = np.array(orient_rc) / center_difference
    if orient_rc[0] == 0:
        phi_rc = np.pi/2 if orient_rc[1] >= 0 else -np.pi/2
    phi_rc = np.arctan(orient_rc[1] / orient_rc[0])
    if orient_rc[0] < 0:
        phi_rc += np.pi
    return np.mod(phi_rc, 2 * np.pi), center_difference


def bbox_particle(binary_particle, image):
    """
    Returns
    -------
    binary_bbox : ndarray
    image_bbox : ndarray
    bbox_lefttop_rc : tuple of int
    """
    loc = ndimage.find_objects(binary_particle, max_label=1)[0]
    bbox_origin_rc = (loc[0].start, loc[1].start)
    binary_bbox = binary_particle[loc]
    image_bbox = image[loc]
    return binary_bbox, image_bbox, bbox_origin_rc


def moments(image, mesh_grid, order=2):
    moments_rc = np.zeros((order + 1, order + 1))
    for i in range(order + 1):
        for j in range(order + 1):
            moments_rc[i, j] = (image * mesh_grid[0] **
                                i * mesh_grid[1] ** j).sum()
    return moments_rc


def _phi_from_SinAndCos(sin_2phi, cos_2phi):
    if sin_2phi >= 0:  # phi is in 0 to 90 degree, cos_2phi are monotonically
        phi = np.arccos(cos_2phi) / 2
    else:  # two phi is in 180 to 360 degree
        phi = np.pi - np.arccos(cos_2phi) / 2
    return np.mod(phi, np.pi)


def phi_from_moments(moments):
    """
    Reference:
    http://homepages.inf.ed.ac.uk/rbf/CVonline/
    LOCAL_COPIES/OWENS/LECT2/node3.html
    """
    inertia_cross = np.sqrt((moments[2, 0] - moments[0, 2]) ** 2 +
                            (2 * moments[1, 1]) ** 2
                            )
    if inertia_cross == 0:
        phi_rc, inertia_ratio, sin_2phi = 0, 1, 0
    else:
        sin_2phi = 2 * moments[1, 1] / inertia_cross
        cos_2phi = (moments[2, 0] - moments[0, 2]) / inertia_cross
        phi_rc = _phi_from_SinAndCos(sin_2phi, cos_2phi)
        inertia_ratio = (moments[2, 0] + moments[0, 2] + inertia_cross
                         ) / (moments[2, 0] + moments[0, 2] - inertia_cross)
    return phi_rc, inertia_ratio, sin_2phi


def position_grid(image, origin_rc=(0, 0)):
    d_row, d_col = np.mgrid[0:image.shape[0], 0:image.shape[1]]
    return d_row - origin_rc[0], d_col - origin_rc[1]


def swap_xy(coords):
    return coords[::-1]


def phi_xy_to_phi_rc(phi_xy):
    return np.mod(np.pi/2 - np.mod(phi_xy, 2 * np.pi), 2 * np.pi)


def phi_rc_to_phi_xy(phi_rc):
    return phi_xy_to_phi_rc(phi_rc)


def orientation_from_phi(phi):
    return (np.cos(phi), np.sin(phi))


def _divide_two_parts(mesh_grid, orient_rc):
    pixels_up = (mesh_grid[0] * orient_rc[0] + mesh_grid[1] * orient_rc[1] > 0)
    pixels_down = (mesh_grid[0] * orient_rc[0] +
                   mesh_grid[1] * orient_rc[1] < 0)
    return pixels_up, pixels_down


def balance_difference(image, phi_rc, mesh_grid):
    orient_rc = orientation_from_phi(phi_rc)
    mass_1 = (image * np.where(mesh_grid[0] * orient_rc[0]
                               - mesh_grid[1] * orient_rc[1] > 0, 1, 0)).sum()
    mass_2 = (image * np.where(mesh_grid[0] * orient_rc[0]
                               - mesh_grid[1] * orient_rc[1] < 0, 1, 0)).sum()
    return (mass_1 - mass_2)/(mass_1 + mass_2)


def which_phi(phi_center, phi_moments, inertia_ratio):
    if inertia_ratio > 1.21:
        return phi_moments
    else:
        return phi_center


def _intensity_at_point(image, coord_rc):
    return image[int(round(coord_rc[0])), int(round(coord_rc[1]))]


def center_intensity_diff(image, center_rc, phi_rc, length_up, length_down):
    up_center = (center_rc[0] + length_up/2 * np.cos(phi_rc),
                 center_rc[1] + length_up/2 * np.sin(phi_rc))
    down_center = (center_rc[0] - length_up/2 * np.cos(phi_rc),
                   center_rc[1] - length_up/2 * np.sin(phi_rc))
    return _intensity_at_point(
        image, up_center) - _intensity_at_point(image, down_center)


def sub_pixel_accuracy(image, binary, center, phi):
    """
    10312018: v7
    """
    length_max = int(2 * np.sqrt(binary.sum()/np.pi))
    y_binary = profile_line(binary, center,
                            (center[0] + length_max * np.cos(phi),
                             center[1] + length_max * np.sin(phi)),
                            order=0
                            )
    rough_boundary = len(y_binary[y_binary > 0]) - 1
    y = profile_line(image, center,
                     (center[0] + length_max * np.cos(phi),
                      center[1] + length_max * np.sin(phi)),
                     order=2
                     )
    dp = (y[rough_boundary]) / (y[rough_boundary] - y[rough_boundary + 1])
    if dp < 2.0:
        return rough_boundary + dp
    else:
        return rough_boundary


def multi_files_tracking(start_frame=0, plot_particle=False,
                         single_particle=False, set_frame_num=None):
    file_list = glob.glob("*.nd2")
    window_centers_caches = {}
    for file in file_list:
        center_file_name = file.split(".")[0] + "_window_centers.csv"
        if os.path.isfile(center_file_name):
            window_centers_caches[file] = pd.read_csv(center_file_name,
                                                      index_col=0)
        else:
            pic = pims.ND2_Reader(file)[0]
            window_centers_caches[file] = show_and_select(pic,
                                                          file.split(".")[0])

    for file in file_list:
        window_centers_xy = window_centers_caches[file]
        single_file_tracking(file, window_centers_xy, start_frame=start_frame,
                             set_frame_num=set_frame_num,
                             plot_particle=plot_particle,
                             single_particle=single_particle)


if __name__ == "__main__":
    multi_files_tracking()
