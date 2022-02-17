import math
import random

import matplotlib.collections
import numpy as np
import numpy.linalg
import scipy.io
import matplotlib.pyplot as plt
from matplotlib import patches
from functools import reduce

from circle_fit import circle_fit
from scipy.signal import argrelmax, argrelmin

# If matplotlib is not working on OSX follow directions in the link below
# https://stackoverflow.com/questions/21784641/installation-issue-with-matplotlib-python


# parameters
# TODO adjust these once rest of code is done to get best fit
PEN_SMOOTHING_WINDOW = 5
TANGENT_WINDOW = 11
CURVATURE_WINDOW = 11
SPEED_THRESHOLD_1 = .25  # a percentage of the average speed
CURVATURE_THRESHOLD = .75  # in degrees per pixel
SPEED_THRESHOLD_2 = .80  # a percentage of the average speed
MINIMUM_DISTANCE_BETWEEN_CORNERS = 0
MINIMUM_ARC_ANGLE = 36  # in degrees
MERGE_LENGTH_THRESHOLD = .2
MERGE_FIT_ERROR_THRESHOLD = .1


def compute_cumulative_arc_lengths(stroke):
    """
    param stroke : a Stroke object with N x,y,t data points

    return : the array (length N) of the cumulative arc lengths between each pair
        of consecutive sampled points in a stroke of length N.
        i.e. make an array where array[i] = length of stroke from stroke[0:i] inclusive
    """

    arc_lengths = [0]
    for i in range(len(stroke.t) - 1):
        distance = math.dist([stroke.x[i], stroke.y[i]], [stroke.x[i + 1], stroke.y[i + 1]])
        cumulative_distance = arc_lengths[-1] + distance
        arc_lengths.append(cumulative_distance)

    return arc_lengths


def compute_smoothed_pen_speeds(stroke, cumulative_arc_lengths,
                                window=PEN_SMOOTHING_WINDOW):
    """
    param stroke : a Stroke object with N x,y,t data points
    param cumulative_arc_lengths : array of the cumulative arc lengths of the
        stroke
    param window : size of the window over which smoothing occurs.
        Handle even window sizes and edge cases wisely

    return : an array (length N) of the smoothed pen speeds at each point on
        a stroke of length N.
    """

    # compute the pen speed
    pen_speeds = []
    for i in range(1, len(stroke.t) - 1):
        speed = (cumulative_arc_lengths[i + 1] - cumulative_arc_lengths[i - 1]) / (stroke.t[i + 1] - stroke.t[i - 1])
        pen_speeds.append(speed)

    # insert initial and final speeds
    initial_speed = pen_speeds[0]
    final_speed = pen_speeds[-1]

    pen_speeds.insert(0, initial_speed)
    pen_speeds.append(final_speed)

    # smoothing
    smoothed_speeds = []
    for i in range(len(pen_speeds)):
        # collect pen speeds in a window
        speeds = [pen_speeds[j] for j in range(max(0, i - window), min(len(pen_speeds), i + window))]
        smoothed_speed = sum(speeds) / len(speeds)
        smoothed_speeds.append(smoothed_speed)

    return smoothed_speeds


def compute_tangents(stroke, window=TANGENT_WINDOW):
    """
    param stroke : a Stroke object with N x,y,t data points
    param window : size of the window over which you calculate the regression
        Handle even window sizes and edge cases wisely

    return : an array (length N) of tangents

    HINT: use https://numpy.org/doc/stable/reference/generated/numpy.linalg.lstsq.html
    The paper refers to these values as "tangents" but you can consider them the same as the slope.
    Make sure you understand what it means to do a linear regression/fit a line on points. You will reuse this function later.
    """

    tangents = []
    for i in range(len(stroke.x)):
        start, end = max(0, i - window), min(len(stroke.x), i + window)
        A = np.vstack([stroke.x[start:end], np.ones(end - start)]).T
        slope, offset = np.linalg.lstsq(A, stroke.y[start:end], rcond=None)[0]
        tangents.append(slope)

    return tangents


def compute_angles(stroke, tangents):
    """
    param stroke : a Stroke object with N x,y,t data points
    param tangents : an array of tangents (length N)

    return : an array of angles (length N)

    HINT: use the math.atan function
    """

    angles = [math.atan(x) for x in tangents]
    return angles


def plot_angles(stroke, angles, corrected=False):
    """
    param stroke : a Stroke object with N x,y,t data points
    param angles : an array of angles

    return : nothing (but should show a plot)

    HINT: plt.figure(id) switches to a new/existing plot workspace
    plt.plot([y_points]) plots on the existing workspace, where y-values are y_points and the x-values are the indices
    """

    plt_num = random.randint(50, 2000)
    fig = plt.figure(plt_num)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_title(f'{"corrected" if corrected else ""} {plt_num}')
    ax.plot(stroke.x, stroke.y)

    # plot angles
    lines = [
        [(stroke.x[i], stroke.y[i]), (stroke.x[i] + math.cos(angles[i]) * 10, stroke.y[i] + math.sin(angles[i]) * 10)]
        for i in range(len(angles))]
    c = np.array([(1, 0, 0, 1) for i in range(len(angles))])

    # color odd conditions differently
    # for i in range(1, len(angles)-1):
    #     if math.fabs(angles[i-1] - angles[i+1]) > math.pi/4:
    #         c[i] = (0, 0, 1, 1)

    lc = matplotlib.collections.LineCollection(lines, colors=c, linewidths=2)

    ax2 = ax.twinx()

    ax2.add_collection(lc)
    ax2.autoscale()

    return


def correct_angles(stroke, angles):
    """
    param stroke : a Stroke object with N x,y,t data points
    param angles : an array of angles (length N)

    return : an array of angles (length N) correcting for the phenomenon you find
    """

    for i in range(1, len(angles) - 1):
        if math.fabs(angles[i - 1] - angles[i + 1]) > math.pi / 4:
            angles[i] = angles[i - 1]

    return angles


def compute_curvatures(stroke, cumulative_arc_lengths, angles,
                       curvature_window=CURVATURE_WINDOW):
    """
    param stroke : a Stroke object with N x,y,t data points
    param cumulative_arc_lengths : an array of the cumulative arc lengths of a stroke
    param angles : an array of angles
    param curvature_window : size of the window over which you calculate the least squares line
        Handle even window sizes and edge cases wisely

    return : an array of curvatures
    """

    curvatures = [0 for x in range(len(stroke.x))]

    for i in range(len(stroke.x)):
        start = max(0, i - int(curvature_window / 2))
        end = min(len(stroke.x), i + int(curvature_window / 2))
        a = np.vstack([cumulative_arc_lengths[start:end], np.ones(end - start)]).T
        m, c = np.linalg.lstsq(a, angles[start:end], rcond=None)[0]
        curvatures[i] = m

    return curvatures


def compute_corners_using_speed_alone(stroke, smoothed_pen_speeds,
                                      speed_threshold_1=SPEED_THRESHOLD_1):
    """
    param stroke : a Stroke object with N x,y,t data points
    param smoothed_pen_speeds : an array of the smoothed pen speeds at each point on a stroke.
    param speed_threshold_1 : a percentage (between 0 and 1). The threshold determines the
        maximum percentage of the average pen speed allowed for a point to be considered a
        segmentation point.

    return : a list of all segmentation points i.e. seg_points = [5, 10] means stroke[5] and stroke[10] are corners
    """

    avg_threshold = sum(smoothed_pen_speeds) / len(smoothed_pen_speeds) * speed_threshold_1

    # Find local minima of speeds
    local_maxima = []
    segment_started = False
    segment_start = -1

    for i in range(len(smoothed_pen_speeds)):
        if smoothed_pen_speeds[i] < avg_threshold:
            if not segment_started:
                segment_started = True
                segment_start = i
        else:
            if segment_started:
                # segment completed
                local_maxima.append([segment_start, i])
                segment_started = False

    if segment_started:
        local_maxima.append([segment_start, i])

    # PLOT SPEED RESULTS
    # fig = plt.figure(56)
    # ax = fig.add_subplot(1, 1, 1)
    # plt.title("Pen speeds")
    #
    # c = []
    # for i in range(len(smoothed_pen_speeds)):
    #     match = False
    #     for seg in local_maxima:
    #         if i >= seg[0] and i <= seg[1]:
    #             match = True
    #     c.append(1 if match else 0)
    #
    # ax.scatter([i for i in range(len(smoothed_pen_speeds))], smoothed_pen_speeds, c=c)
    # ax.plot([0, len(smoothed_pen_speeds)], [avg_threshold, avg_threshold])

    return local_maxima


def compute_corners_using_curvature_and_speed(stroke, smoothed_pen_speeds, curvatures,
                                              curvature_threshold=CURVATURE_THRESHOLD,
                                              speed_threshold_2=SPEED_THRESHOLD_2):
    """
    param stroke : a Stroke object with N x,y,t data points
    param smoothed_pen_speeds : an array of the smoothed pen speeds at each point on a stroke.
    param curvatures : an array of curvatures
    param curvature_threshold : in degress per pixel. The minimum threshold for the curvature of
        a point for the point to be considered a segmentation point.
    param speed_threshold_2 : a percentage (between 0 and 1). The threshold determines the
        maximum percentage of the average pen speed allowed for a point to be considered a
        segmentation point.


    return : a list of all segmentation points i.e. seg_points = [5, 10] means stroke[5] and stroke[10] are corners
    """

    # find local maxima of curvature
    local_maxima = []
    segment_started = False
    segment_start = -1
    avg_threshold = sum(smoothed_pen_speeds) / len(smoothed_pen_speeds) * speed_threshold_2

    for i in range(len(curvatures)):
        if curvatures[i] * 180 * math.pi < curvature_threshold and smoothed_pen_speeds[i] < avg_threshold:
            if not segment_started:
                segment_started = True
                segment_start = i
        else:
            if segment_started:
                # segment completed
                local_maxima.append([segment_start, i])
                segment_started = False

    if segment_started:
        local_maxima.append([segment_start, i])

    return local_maxima


def find_min(i, stroke, combined, minimum_distance_between_corners, combined_indices=None):
    """
    Finds the minimum index of the stroke when combining strokes
    """
    if combined_indices is None:
        combined_indices = [i]

    for j in range(len(combined)):
        if math.dist([stroke.x[combined[i][0]], stroke.y[combined[i][0]]], [stroke.x[combined[j][0]], stroke.y[combined[j][0]]]) <= minimum_distance_between_corners \
                or math.dist([stroke.x[combined[i][0]], stroke.y[combined[i][0]]], [stroke.x[combined[j][1]], stroke.y[combined[j][1]]]) <= minimum_distance_between_corners:

            if j not in combined_indices:
                combined_indices.append(j)

                # recursively run with new index
                return find_min(j, stroke, combined, minimum_distance_between_corners, combined_indices)

    return combined_indices, combined[min(combined_indices)][0]


def find_max(i, stroke, combined, minimum_distance_between_corners, combined_indices=None):
    """
    Finds the maximum index of the stroke when combining strokes
    """
    if combined_indices is None:
        combined_indices = [i]

    for j in range(len(combined)):
        if math.dist([stroke.x[combined[i][1]], stroke.y[combined[i][1]]], [stroke.x[combined[j][1]], stroke.y[combined[j][1]]]) <= minimum_distance_between_corners \
                or math.dist([stroke.x[combined[i][1]], stroke.y[combined[i][1]]], [stroke.x[combined[j][0]], stroke.y[combined[j][0]]]) <= minimum_distance_between_corners:

            if j not in combined_indices:
                combined_indices.append(j)

                # recursively run with new index
                return find_max(j, stroke, combined, minimum_distance_between_corners, combined_indices)

    return combined_indices, combined[max(combined_indices)][1]


def combine_corners(stroke, cumulative_arc_lengths, corners_using_speed_alone,
                    corners_using_curvature_and_speed,
                    minimum_distance_between_corners=MINIMUM_DISTANCE_BETWEEN_CORNERS):
    """
    param stroke : a Stroke object with N x,y,t data points
    param cumulative_arc_lengths : an array of the cumulative arc lengths of the stroke
    param corners_using_speed_alone : a list of all segmentation points found using speed
    param corners_using_curvature_and_speed : a list of all segmentation points found using
        curvature and speed
    param minimum_distance_between_corners : minimum distance allowed between two segmentation
        points. Must use a DISTANCE metric, index is not a distance metric

    return : a list of all segmentation points, with nearly coincident points removed. The list
    should be sorted from first to last segmentation point along the stroke.
    """

    combined = corners_using_curvature_and_speed[:]
    combined.extend(corners_using_speed_alone)

    combined_corners = []
    searched_indices = set()

    for i in range(len(combined)):
        if i in searched_indices:
            continue

        # find smallest beginning point within distance
        combined_min_indices, min_i_val = find_min(i, stroke, combined, minimum_distance_between_corners)   # min relative to this
        combined_max_indices, max_i_val = find_max(i, stroke, combined, minimum_distance_between_corners)   # min relative to this

        new_segment = [min_i_val, max_i_val]
        combined_corners.extend(new_segment)

        searched_indices.update(combined_min_indices)
        searched_indices.update(combined_max_indices)

    return sorted(combined_corners)


def compute_linear_error(stroke, start_point, end_point):
    """
    param stroke : a Stroke object with N x,y,t data points
    param start_point : a segmentation point, representing the index into the stroke
        where the segment begins
    param end_point : a segmentation point, respresenting the index into the stroke
        where the segment ends

    return : the residual error of the linear fit

    HINT: look at the other return values in numpy.linalg.lstsq
    """

    A = np.vstack([stroke.x[start_point:end_point], np.ones(end_point - start_point)]).T
    error = np.linalg.lstsq(A, stroke.y[start_point:end_point], rcond=None)[1]

    return 0 if not error else error[0]

def compute_circular_error(stroke, start_point, end_point):
    """
    param stroke : a Stroke object with N x,y,t data points
    param start_point : a segmentation point, representing the index into the stroke
        where the segment begins
    param end_point : a segmentation point, respresenting the index into the stroke
        where the segment ends

    return : the residual error of the circle/curve fit

    HINT: you need to implement residual error by hand / not as easy as linear error
    May be helpful draw on paper an example stroke and a circle fit
    HINT: error can be positive or negative, make sure you take the absolute value
    Spotcheck your result, it should be a positive number similar
    """

    cx, cy, r = circle_fit(stroke.x[start_point:end_point], stroke.y[start_point:end_point])

    residual = 0
    for i in range(start_point, end_point):
        x, y = stroke.x[i], stroke.y[i]
        residual += (np.sqrt((x-cx)**2 + (y-cy)**2) - r)**2

    return residual

def compute_subtended_angle(stroke, start_point, end_point):
    """
    param stroke : a Stroke object with N x,y,t data points
    param start_point : a segmentation point, representing the index into the stroke
        where the segment begins
    param end_point : a segmentation point, representing the index into the stroke
        where the segment ends

    return : the angle subtended by the arc of the circle fit to the segment

    Hard problem with some trig! Google/StackOverflow is your friend
    Incomplete pseudocode:
    1. Fit a circle on the stroke
    2. Select a few stroke points to use for this calculation
    3. Find the closest point on the circle from the stroke point.
    4. Calculate the subtended angle with chords connecting the closest circle points

    Hint: You can ignore segments >360 degrees, but you must handle those >180.
        Is it possible to calculate a subtended angle >180 from a chord?
    Hint: asin may be useful in calculating the subtended angle,
        but you will have to adjust its range and domain
    """

    # TODO: your part 7b code here

    return 0


def choose_segment_type(stroke, linear_error, circular_error, subtended_angle, minimum_arc_angle=MINIMUM_ARC_ANGLE):
    """
    param stroke : a Stroke object with N x,y,t data points
    param linear_error : residual error of the linear fit of the segment
    param circular_error : residual error of the circular fit of the segment
    param subtended_angle : angle subtended by the arc of the circle
    param minimum_arc_angle : minimum angle necessary for classification as a curve

    return : 0 if the segment should be a line; 1 if the segment should be a curve
    """

    # TODO: your part 7c code here

    return 0


def merge(stroke, segpoints, segtypes):
    """
    TODO (optional): define your function signature. You may change the function signature,
    but please name your function 'merge'. You may use helper functions.
    """

    # TODO (optional): your part 10 code here

    return segpoints, segtypes


def segment_stroke(stroke):
    """
    param stroke : a Stroke object with N x,y,t data points

    return :
        segpoints : an array of length M containing the segmentation points
            of the stroke. Each element in the array is an index into the stroke.
        segtypes : an array of length M-1 containing 0's (indicating a line)
            and 1's (indicating an arc) that describe the type of segment between
            segmentation points. Element i defines the type of segment between
            segmentation points i and i+1.
    """

    segpoints, segtypes = [], []

    # PART 1
    cumulative_arc_lengths = compute_cumulative_arc_lengths(stroke)

    # PART 2
    smoothed_pen_speeds = compute_smoothed_pen_speeds(stroke, cumulative_arc_lengths)

    # PART 3
    tangents = compute_tangents(stroke)

    # PART 4
    angles = compute_angles(stroke, tangents)
    plot_angles(stroke, angles)
    corrected_angles = correct_angles(stroke, angles)
    plot_angles(stroke, corrected_angles, True)
    curvatures = compute_curvatures(stroke, cumulative_arc_lengths, corrected_angles)

    # PART 5
    corners_using_speed_alone = compute_corners_using_speed_alone(stroke, smoothed_pen_speeds)
    corners_using_curvature_and_speed = compute_corners_using_curvature_and_speed(stroke, smoothed_pen_speeds,
                                                                                  curvatures)

    # PART 6
    segpoints = combine_corners(stroke, cumulative_arc_lengths, corners_using_speed_alone,
                                corners_using_curvature_and_speed)

    # PART 7
    for i in range(len(segpoints) - 1):
        start_point = segpoints[i]
        end_point = segpoints[i + 1]
        linear_error = compute_linear_error(stroke, start_point, end_point)
        circular_error = compute_circular_error(stroke, start_point, end_point)
        subtended_angle = compute_subtended_angle(stroke, start_point, end_point)
        segment_type = choose_segment_type(stroke, linear_error, circular_error, subtended_angle)
        segtypes.append(segment_type)

    # OPTIONAL: PART 10
    segpoints, segtypes = merge(stroke, segpoints, segtypes)

    return segpoints, segtypes
