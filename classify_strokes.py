import math
from stroke_segmentation import segment_stroke

def normalize_segpoints(stroke):
    """
    param stroke : a Stroke object with N x,y,t data points

    return :
        (template_x, template_y): a tuple of arrays representing the normalized X
        and Y coordinates, ordered by time, of the points to be
        used in the MHD calculation to compare against the given templates

        Coordinates are normalized so that points span between 0 and 1

        Relevant points would be the endpoints of all segments,
        plus the curve segment midpoints
    """
    #NORMALIZE THE STROKE POINTS

    #GET THE SET OF POINTS TO BE USED IN THE MHD CALCULATION

    return ([],[])


def calculate_MHD(stroke, template):
    """
    param stroke : a Stroke object with N x,y,t data points
    param template : a Template object with x,y template points and name

    return :
        float representing the Modified Hausdorf Distance of the normalized segpoints
        and the template points,
        The formula for the Modified Hausdorf Distance can be found in the
        paper "An image-based, trainable symbol recognizer for
        hand-drawn sketches" by Kara and Stahovichb

    """
    return 0

def classify_stroke(stroke, templates):
    """
    param stroke : a Stroke object with N x,y,t data points
    param templates: a list of Template objects, each with name, x, y
                     Each template represents a different symbol.

    return :
        string representing the name of the best matched Template of a stroke
    """

    return "NOT IMPLEMENTED"
