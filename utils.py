import casadi
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import sys

import os, random

from copy import deepcopy

from scipy.spatial import ConvexHull, convex_hull_plot_2d
from sklearn.cluster import DBSCAN

def rotate(angle, point):  # rotate a point by angle in radians
    return np.dot(
        [[np.cos(angle), -np.sin(angle)], [np.sin(angle), np.cos(angle)]], point
    )


def flatten_comprehension(matrix):
    return [item for row in matrix for item in row]


def bounding_box(state, length, width, ds=0.25):
    x = state[0]
    y = state[1]
    yaw = state[2]
    # Define the bounding coordinates
    half_diag = np.sqrt((length / 2) ** 2 + (width / 2) ** 2)
    diag_angle = np.arctan(width / length)
    topLeft = rotate(
        yaw, [half_diag * np.cos(diag_angle), half_diag * np.sin(diag_angle)]
    )
    topRight = rotate(
        yaw, [half_diag * np.cos(diag_angle), -half_diag * np.sin(diag_angle)]
    )
    bottomLeft = rotate(
        yaw, [-half_diag * np.cos(diag_angle), half_diag * np.sin(diag_angle)]
    )
    bottomRight = rotate(
        yaw, [-half_diag * np.cos(diag_angle), -half_diag * np.sin(diag_angle)]
    )
    top = [
        topLeft,
        topRight,
    ]  # [[array([1.06066017, 3.8890873 ]), array([3.8890873 , 1.06066017])]
    left = [topLeft, bottomLeft]  # a list of two points
    right = [topRight, bottomRight]
    bottom = [bottomLeft, bottomRight]
    coords = [top, left, right, bottom]  # a list of 4 list of 2 points (np.array)

    # Generate the bounding boxes
    ds = 0.25  # step size [m]
    bbox_xs = []
    bbox_ys = []
    for coord in coords:  # Note: iterate for 4 sides (edges) of the bounding box!
        p1 = coord[0]  # a point (x1, y1)
        p2 = coord[1]  # a point (x2, y2)
        dist = np.sqrt(
            (p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2
        )  # distance between 2 points
        N = int(
            np.ceil(dist / ds)
        )  # ceil: min integer >= (dist/ds) => Number of interval
        ss = np.linspace(0, dist, N)  # step_size: N
        xs = np.interp(ss, [0, dist], [p1[0], p2[0]])  # spawn a list of x coordinate
        ys = np.interp(ss, [0, dist], [p1[1], p2[1]])  # spawn a list of y coordinate
        bbox_xs.append(xs + x)
        bbox_ys.append(ys + y)

    # Flatten the arrays
    bbox_xs_flat = flatten_comprehension(bbox_xs)
    bbox_ys_flat = flatten_comprehension(bbox_ys)
    bbox = [bbox_xs_flat, bbox_ys_flat]
    return bbox


def find_closest_idx_frenet_path(point_s, path_s):
    """
    Return the index of point in path_s that is closest to point_s
    Args:
        - point_s: (float) the s value of the query point_s.
        - path_s: ([float]) a list of s values
    Return:
        - index: an integer
    """
    ds = path_s[1] - path_s[0]
    index = int(np.floor((point_s - path_s[0])/ds))
    if index > len(path_s)-1 or index <0:
        return -1 # invalid
    elif index == len(path_s)-1:
        return len(path_s)-2
    elif index == 0:
        return 1
    else:
        return index


def get_bounds_extrema(bboxs, lane_min, lane_max, peds = {}):
    """
    Return the (lb, ub) given bounding box & pedestrian points
    Args:
        - bboxs: dictionary of bounding boxes
            - key: 'lb', 'ub'
            - value: list of bbox points (NOT Corners! Entire bounding box!)
        - peds: dictionary of pedestrian points
            - key: 'lb', 'ub'
            - value: list of pedestrian coordinates (plus safety buffer in d-axis) in frenet frame
        - global_route: np.array ((ego_s, 0), (ego_s+ds, 0), ..., (ego_s+S, 0))
        - lane_min = np.array ((ego_s, DEFAULT_LB), (ego_s+ds, DEFAULT_LB), ..., (ego_s+S, DEFAULT_LB))
        - lane_max = np.array ((ego_s, DEFAULT_UB), (ego_s+ds, DEFAULT_UB), ..., (ego_s+S, DEFAULT_UB))
    Returns:
        - (lb, ub)
            - lb: list of (s, d) lower bound points in Frenet space
            - ub: list of (s, d) upper bound points in Frenet space
    """

    # 1. Lower bound: Pick the maximum d value
    # Initialize lower bound starting from ego position [ego_s, ego_s+S] with d = lane_min
    lb = deepcopy(lane_min).astype(float)

    # Iterage bboxs[lb] points, for each point
    for box in bboxs['lb']:
        for pt in box:
            # Find the index this point is closest to. If out of boundary -> skip this point
            # Update lower_bound at this index if pt.d > lb[idx].d
            s, d = pt
            idx = find_closest_idx_frenet_path(s, lb[:,0])                
            if idx==-1: # Skip if out of bound
                continue
            else:
                if idx < len(lb)-1:
                    if d > lb[idx][1]:
                        lb[idx][1] = d
                    if d > lb[idx+1][1]:
                        lb[idx+1][1] = d
                else:
                    if d > lb[idx][1]:
                        lb[idx][1] = d

    # Iterage peds[lb] points, for each point
    for ped in peds['lb']:
        # Find the index this point is closest to. If out of boundary -> skip this point
        # Update lower_bound at this index if pt.d > lb[idx].d.
        s, d = ped
        idx = find_closest_idx_frenet_path(s, lb[:,0])
        if idx==-1: # Skip if out of bound
            continue
        else:
            if idx < len(lb)-1:
                if d > lb[idx][1]:
                    lb[idx][1] = d
                if d > lb[idx+1][1]:
                    lb[idx+1][1] = d
            else:
                if d > lb[idx][1]:
                    lb[idx][1] = d                


    # 2. Upper bound: Pick the minimum d value
    # Initialize upper bound starting from ego position [ego_s, ego_s+S] with d = lane_max
    ub = deepcopy(lane_max).astype(float)
    # Iterage bboxs[ub] points, for each point
    for box in bboxs['ub']:
        for pt in box:
        # Find the index this point is closest to. If out of boundary -> skip this point
        # Update upper_bounds at this index if pt.d < ub[idx].d
            s, d = pt
            idx = find_closest_idx_frenet_path(s, ub[:,0])
            if idx==-1: # Skip if out of bound
                continue
            else:
                if idx < len(ub)-1:
                    if d < ub[idx][1]:
                        ub[idx][1] = d
                    if d < ub[idx+1][1]:
                        ub[idx+1][1] = d
                else:
                    if d < ub[idx][1]:
                        ub[idx][1] = d                    

    # Iterage peds[ub] points, for each point
    for ped in peds['ub']:
        # Find the index this point is closest to. If out of boundary -> skip this point
        # Update upper_bounds at this index if pt.d < ub[idx].d
        s, d = ped
        idx = find_closest_idx_frenet_path(s, ub[:,0])
        if idx==-1: # Skip if out of bound
            continue
        else:
            if idx < len(ub)-1:
                if d < ub[idx][1]:
                    ub[idx][1] = d
                if d < ub[idx+1][1]:
                    ub[idx+1][1] = d
            else:
                if d < ub[idx][1]:
                    ub[idx][1] = d  

    return (lb, ub)



def vis_default():
    plt.figure(figsize=(50,5))
    plt.clf()
    plt.axhline(y=-1.75, color='k', linestyle='-')
    plt.axhline(y=1.75, color='y', linestyle='--')
    plt.axhline(y=1.75+3.5, color='k', linestyle='-')
    plt.locator_params(axis='both', nbins=5)


def vis_scenario(ego_state,lb,ub,stat_obss,dyn_obss,bboxs,peds=[],dyn_preds= [], length=5,width=2):
    """
    Visualizes a planning scenario in the Frenet frame, including the ego vehicle, static and dynamic obstacles, 
    bounding boxes, pedestrians, and predicted trajectories.

    Args:
        ego_state (array-like): Ego vehicle state [x, y, yaw] in the Frenet frame.
        lb (array-like): Lower boundary coordinates as a sequence of (x, y) points.
        ub (array-like): Upper boundary coordinates as a sequence of (x, y) points.
        stat_obss (list of array-like): List of static obstacles, each specified by [x, y, yaw, length, width].
        dyn_obss (list of array-like): List of dynamic obstacles with the same format as static obstacles.
        bboxs (list of array-like): List of points representing sampled bounding boxes (e.g., for uncertainty visualization).
        peds (list of array-like, optional): List of pedestrian positions [x, y]. Defaults to an empty list.
        dyn_preds (list of array-like, optional): List of predicted trajectories for dynamic obstacles. 
            Each trajectory is an array of shape (T, 4): [x, y, yaw, speed]. Defaults to an empty list.
        length (int, optional): Length of the ego vehicle (for visualization). Defaults to 5.
        width (int, optional): Width of the ego vehicle (for visualization). Defaults to 2.

    Notes:
        - If `dyn_preds` is not provided, a simple constant-velocity prediction is visualized.
        - Static and dynamic obstacles are rendered as rectangles.
        - Pedestrians are shown as yellow circles.
        - Bounding boxes are shown as red scatter points.
        - Upper and lower boundaries are shown as dashed blue and red lines, respectively.
        - The ego vehicle is rendered as a blue rectangle.

    Saves:
        Visualization output is shown via matplotlib but not saved by default.
    """
    vis_default()
    # Visualize bounding boxes
    for i in range(len(bboxs)):
        plt.scatter(bboxs[i][0],bboxs[i][1],s=0.1, color='r',linewidths=10)

    # Static obstacle
    for obs in stat_obss:
        plt.gca().add_patch(patches.Rectangle((obs[0]-obs[-2]/2, obs[1]-obs[-1]/2),obs[-2],obs[-1], color='gray', angle=np.rad2deg(obs[2]), rotation_point='center'))

    # Pedestrians
    for ped in peds:
        plt.scatter(ped[0],ped[1],5e2,'y')

    # Dynamic obstacle with prediction
    preds = []
    if len(dyn_preds) > 0: # given predictions
        for pred, obs in zip(dyn_preds, dyn_obss):
            plt.gca().add_patch(patches.Rectangle((pred[0][0]-obs[-2]/2, pred[0][1]-obs[-1]/2),obs[-2],obs[-1], color='red', angle=np.rad2deg(pred[0][2]), rotation_point='center'))
            preds = pred[1:,:] 
    else: # prediction not given, vis constant prediction
        for obs in dyn_obss:
            plt.gca().add_patch(patches.Rectangle((obs[0]-obs[-2]/2, obs[1]-obs[-1]/2),obs[-2],obs[-1], color='red', angle=np.rad2deg(obs[2]), rotation_point='center'))
            preds = np.array([[obs[0] + obs[3]*k*1*np.cos(obs[2]),obs[1]+obs[3]*k*1*np.sin(obs[2]),obs[2],obs[3]] for k in range(10)]) # over N
    if len(preds) > 0: plt.plot(preds[:,0],preds[:,1],'r',alpha=0.2,linewidth=68);

    plt.xlim(-5, 150)
    plt.ylim(-2.5, 6)
    # plt.legend(fontsize="30", loc="upper left")

    if len(ub) > 0 and len(lb) > 0:
        plt.plot(ub[:,0],ub[:,1],'--b',label='Upper bound',linewidth=10)
        plt.plot(lb[:,0],lb[:,1],'--r', label='Lower bound',linewidth=10)
    plt.gca().add_patch(patches.Rectangle((ego_state[0]-length/2, ego_state[1]-width/2),length,width, color='blue', angle=np.rad2deg(ego_state[2]), rotation_point='center'))


def which_bound(state0, box, lane_min_val = -2, lane_max_val = 5, evaluate = True, buffer=0):
    """
    Return to which boundary the bounding box needs to be included

    Args:
        state0 (_type_): _description_
        box (_type_): _description_
        lane_min_val (int, optional): _description_. Defaults to -2.
        lane_max_val (int, optional): _description_. Defaults to 5.
        evaluate (bool, optional): _description_. Defaults to True.
        buffer (int, optional): _description_. Defaults to 0.

    Returns:
        0 -- included in the lower bound
        1 -- included in the upper bound
        2 -- considered as a soft constraint
    """
    # NOTE:: the buffer needs to reflect the noise range.
    
    box = np.array(box)

    min_d = min(box[1])
    max_d = max(box[1])

    # If the upper gap is not enough
    ugap = lane_max_val-max_d
    ugap_feas = ugap > -buffer
    
    # If the lower gap is not enough
    lgap = min_d - lane_min_val
    lgap_feas = lgap > -buffer

    # Both gaps are not feasible. Consider it as a soft constraint
    if not ugap_feas and not lgap_feas: 
        return 2

    # Lower gap is feasible
    if not ugap_feas and lgap_feas:
        return 1

    # Upper gap is feasible
    if not lgap_feas and ugap_feas:
        return 0
    
    if evaluate:
        # Both gaps are feasible then we evaluate
        # (i) euclidean distance to prev gap (ii) gap size (iii) distance to the reference path
        # NOTE:: for (i) if a prev gap does not exist, it's based on the ego state. For (ii) penalize more strongly if the gap buffer is active
        ugap_cost = np.linalg.norm(np.array(state0[:2])-np.array([np.mean(box[0]),(lane_max_val + max_d)/2])) \
                    - (ugap**2 + (ugap < 0)*5*ugap)\
                    + np.log(abs((lane_max_val + max_d)/2))
        # print("ugap euc {} gap {} dev {}".format(np.linalg.norm(np.array(state0[:2])-np.array([np.mean(box[0]),(lane_max_val + max_d)/2])),- (lane_max_val-max_d),abs((lane_max_val + max_d)/2)))
        lgap_cost = np.linalg.norm(np.array(state0[:2])-np.array([np.mean(box[0]),(min_d + lane_min_val)/2])) \
                    - (lgap**2 + (lgap < 0)*5*lgap)\
                    + np.log(abs((min_d + lane_min_val)/2))
        # print("lgap euc {} gap {} dev {}".format(np.linalg.norm(np.array(state0[:2])-np.array([np.mean(box[0]),(min_d + lane_min_val)/2])),- (min_d-lane_min_val),abs((min_d + lane_min_val)/2)))

        # print("ugap cost {} lgap cost {} diff std {}".format(ugap_cost,lgap_cost,np.std([ugap_cost, lgap_cost])))
        # if np.std([ugap_cost, lgap_cost]) < 0.01: # if the difference is marginal, leave FCP to decide
        #     # print("marginal cost difference")
        #     return 2
        if ugap_cost > lgap_cost:
            # print("lower gap is better")
            return 1
        else:
            # print("upper gap is better")
            return 0
    else:
        # leave it to other module to decide
        return 2


def get_ped_convex_hull(list_of_peds, eps=3, min_samples=2):
    ped_clusters = get_clusters(list_of_peds, eps=eps, min_samples=min_samples)
    if len(ped_clusters) == 0:
        return np.array([])
    # print("ped_clusters: ", ped_clusters)
    cluster_vertices = get_convex_hull(ped_clusters)
    if len(cluster_vertices) == 0:
        return np.array([])
    # Flatten the list of vertices
    cluster_vertices_flat = [vertex for cluster in cluster_vertices for vertex in cluster]
    # print(f"cluster_vertices: {cluster_vertices}")
    # print(f"cluster_vertices_flat: {cluster_vertices_flat}")
    return cluster_vertices_flat
    

def get_clusters(list_of_points, eps=3, min_samples=2):
    """
    Cluster a list of points using DBSCAN.

    Args:
        list_of_points (list): List of points to be clustered.
        eps (float): The maximum distance between two samples for one to be considered as in the neighborhood of the other.
        min_samples (int): The number of samples in a neighborhood for a point to be considered as a core point.

    Returns:
        dict: A dictionary where keys are cluster labels and values are lists of points in each cluster.
        If no points are clustered, returns an empty dictionary.
    """
    if len(list_of_points) == 0:
        return np.array([])

    db = DBSCAN(eps=eps, min_samples=min_samples).fit(np.array(list_of_points))
    ped_clusters = {}
    for idx, label in enumerate(db.labels_):
        if label == -1:
            print("Noise point detected in clustering.")
            continue
        ped_clusters.setdefault(label, []).append(list_of_points[idx])
    # print("ped clusters: ", ped_clusters)
    return ped_clusters


def get_convex_hull(clusters, num_points_per_edge=10):
    """
    Compute the convex hull for each cluster of points.
    Args:
        clusters (dict): A dictionary where keys are cluster labels and values are lists of points in each cluster.
    Returns:
        list: A list of convex hull vertices for each cluster.
    """
    cluster_vertices = []
    for label, peds_pos in clusters.items():
        """
        label: cluster label 0,1,...
        peds_pos: [ped_pos1, ped_pos2, ...]
        """
        # peds_pos, peds_pos_frenet = zip(*peds_pos)
        peds_pos = np.array(peds_pos)
        if len(peds_pos) < 3:
            # print(
            #     f"Cluster {label} doesn't have sufficient number of peds to form a convex hull. Just append entire ped_pos_list."
            # )
            cluster_vertices.append(peds_pos)
            continue

        # print(f"    Get convex hull for cluster {label}")
        hull = ConvexHull(np.asarray(peds_pos), qhull_options='QJ')
        
        # Interpolate between the vertices of the convex hull
        # to create a smooth boundary -- NOTE:: this is not necessary for the FCP, but it can be useful for visualization
        interpolated = []
        for simplex in hull.simplices:
            p1, p2 = peds_pos[simplex]
            # Interpolate between p1 and p2
            for t in np.linspace(0, 1, num_points_per_edge, endpoint=False):
                interp_point = (1 - t) * p1 + t * p2
                interpolated.append(interp_point)
        interpolated.append(peds_pos[hull.simplices[0][0]])

        # print(f"    Num vertices in this hull: {len(hull.vertices)}")
        cluster_vertices.append(interpolated)
    return cluster_vertices


def update_bounds(s0, stat_obss, lane_min_val = -2, lane_max_val = 5, lane_lb = -2, 
                  lane_ub = 2, ref_lb = -2, ref_ub = 5, ds=1, S=100,length=5, width=2,
                  evaluate = True, which_bound_check = True, buffer_scale=1.7,wb_buffer=1, peds=[]):
    N = int(S/ds)
    lon_buffer = length*buffer_scale  # NOTE:: the margin must be at the (ego_length/2+obs_length/2) at minimum
    lat_buffer = width*buffer_scale # NOTE:: the margin must be at the (ego_width/2+obs_width/2) at minimum

    # Get the bounding boxes of obstacles
    bboxs = [bounding_box(obs, length + lon_buffer, width + lat_buffer) for obs in stat_obss]

    # Compute the boundaries
    lane_lb = min(max(lane_min_val, lane_lb), ref_lb) # hard bound
    lane_ub = max(min(lane_max_val, lane_ub),ref_ub) # hard bound

    # Get the boundaries
    bboxs_trans_lb = []
    bboxs_trans_ub = []
    prev_gap = [s0,0]
    if not which_bound_check:
        for box,obs in zip(bboxs,stat_obss): 
            if min(box[1]) > 0: # upper bound
                bboxs_trans_ub.append(np.transpose(box))
            else:
                bboxs_trans_lb.append(np.transpose(box))
    else:
        for box,obs in zip(bboxs,stat_obss): # NOTE:: make sure that stat_obss is sorted ascending w.r.t. s
            wb = which_bound(prev_gap, box, lane_min_val = lane_min_val, lane_max_val = lane_max_val, evaluate=evaluate,buffer=wb_buffer)
            # print("which bound {}".format(wb))
            if wb == 1: # should be an upper bound
                bboxs_trans_ub.append(np.transpose(box))
                prev_gap = [np.mean(box[0]), np.mean([min(box[1]), lane_min_val])]
            elif wb == 0: # should be an lower bound
                bboxs_trans_lb.append(np.transpose(box))
                prev_gap = [np.mean(box[0]), np.mean([max(box[1]), lane_max_val])]
            else:
                pass
                # print("decision cannot be made. Soft constrain this obstacle.")

    
    lane_min = np.array([[s0+i*ds, lane_lb] for i in range(N)])
    lane_max = np.array([[s0+i*ds, lane_ub] for i in range(N)])
    
    # Assign pedestrians to lower and upper bounds
    peds_dict = {'lb':[], 'ub':[]}
    for ped in peds:
        lb_ind = find_closest_idx_frenet_path(ped[0], lane_min[:,0])
        ub_ind = find_closest_idx_frenet_path(ped[0], lane_max[:,0])
        if ped[1] < lane_min[lb_ind][1]: # lower bound
            peds_dict['lb'].append([ped[0], ped[1] + 1])
            # peds_dict['lb'].append([ped[0], ped[1]])
        elif ped[1] > lane_max[ub_ind][1]: # upper bound
            peds_dict['ub'].append([ped[0], ped[1] - 1])
            # peds_dict['ub'].append([ped[0], ped[1]])
        else:
            # Otherwise, choose a closer side
            if abs(ped[1] - lane_min[lb_ind][1]) < abs(ped[1] - lane_max[ub_ind][1]):
                peds_dict['lb'].append([ped[0], ped[1] + 1])
                # peds_dict['lb'].append([ped[0], ped[1]])
            else:
                peds_dict['ub'].append([ped[0], ped[1] - 1])
                # peds_dict['ub'].append([ped[0], ped[1]])
        
    # Get clusters of pedestrians in each side (lower and upper bounds)
    # NOTE:: this is not necessary for the FCP, but it can be useful for visualization and/or boundary generations 
    #        since it reduces the number of bounding boxes.
    # If the number of pedestrians is large, we can cluster them to reduce the number of bounding boxes
    lb_convex_hull = get_ped_convex_hull(peds_dict['lb'], eps=4, min_samples=1) if any(peds_dict['lb']) else np.array([])
    ub_convex_hull = get_ped_convex_hull(peds_dict['ub'], eps=4, min_samples=1) if any(peds_dict['ub']) else np.array([])
    # Easiest way could be to add clusters to the lower and upper bounds
    bboxs_trans_lb.append(lb_convex_hull) if any(peds_dict['lb']) else None
    bboxs_trans_ub.append(ub_convex_hull) if any(peds_dict['ub']) else None
        
    bboxs_dict = {}
    bboxs_dict['lb'] = bboxs_trans_lb
    bboxs_dict['ub'] = bboxs_trans_ub

    (lb, ub) = get_bounds_extrema(bboxs_dict, lane_min, lane_max, peds={'lb':[], 'ub':[]})
    return lb,ub,bboxs


def adaptive_bounds(ego_state, stat_obss, 
                    lane_lb, lane_ub, # variable
                    ref_lb=-2, ref_ub=2, # global path range
                    lane_min_val = -2, lane_max_val = 5, # drivable space rage
                    width=2, length=5, # ego
                    S=50, ds = 1, 
                    evaluate = True, which_bound_check = True,
                    wb_buffer=1,
                    ):
    """Adaptively adjust the bounds

    Args:
        ego_state (_type_): _description_
        stat_obss (_type_): _description_
        lane_lb (_type_): _description_
        lane_ub (_type_): _description_
        ref_ub (int, optional): _description_. Defaults to 2.
        lane_max_val (int, optional): _description_. Defaults to 5.
        length (int, optional): _description_. Defaults to 5.
        ds (int, optional): _description_. Defaults to 1.

    Returns:
        _type_: _description_
    """

    # NOTE:: check the function update_bounds() to hard limit the boundaries.
    # TODO:: input arguments can be simplified if the two functions have the same arguments.
    (lb,ub,bboxs) = update_bounds(ego_state[0],stat_obss, 
                                lane_lb=lane_lb, lane_ub=lane_ub, 
                                ref_lb=ref_lb, ref_ub=ref_ub,
                                S=S, ds = ds, 
                                evaluate = evaluate, which_bound_check = which_bound_check,
                                wb_buffer=wb_buffer)

    if min(ub[:,1])-max(lb[:,1]) < width: # not enough space -- need to deviate if allowable space
        while min(ub[:,1])-max(lb[:,1]) < width \
            and not (min(lb[:,1]) == lane_min_val and max(ub[:,1]) == lane_max_val):
            print("Not enough space. Expand the boundaries until allowable.")
            lane_lb = np.sign(lane_lb)*1.1*abs(lane_lb)
            lane_ub = np.sign(lane_ub)*1.1*abs(lane_ub)
            (lb,ub,bboxs) = update_bounds(ego_state[0],stat_obss, 
                                          lane_lb=lane_lb, lane_ub=lane_ub, 
                                          ref_lb=ref_lb, ref_ub=ref_ub,
                                          S=S, ds = ds, 
                                          evaluate = evaluate, which_bound_check = which_bound_check, wb_buffer=wb_buffer)
    else:
        while min(ub[:,1])-max(lb[:,1]) > ref_ub - ref_lb \
            and not (min(lb[:,1]) == ref_lb and max(ub[:,1]) == ref_ub):
            print("Enough space. Reduce the boundaries.")
            lane_lb = np.sign(lane_lb)*0.9*abs(lane_lb)
            lane_ub = np.sign(lane_ub)*0.9*abs(lane_ub)

            (lb_,ub_,bboxs_) = update_bounds(ego_state[0],stat_obss, 
                                          lane_lb=lane_lb, lane_ub=lane_ub, 
                                          ref_lb=ref_lb, ref_ub=ref_ub,
                                          S=S, ds = ds, 
                                          evaluate = evaluate, which_bound_check = which_bound_check,wb_buffer=wb_buffer)
            if min(ub_[:,1])-max(lb_[:,1]) <= ref_ub - ref_lb:
                print("do not update as we may have to expand it again")
                break
            elif ego_state[1] > max(ub_[:,1]):
                print("solution might be infeasible")
                break
            else:
                lb, ub, bboxs = lb_,ub_,bboxs_

            if min(lb[:,1]) == lane_min_val and max(ub[:,1]) == lane_max_val:
                print("Cannot reduce further.")
                break
    
    return (lb,ub,bboxs,lane_lb,lane_ub)


def smooth_path(prevs, new, window_size=3):
    """ Return an averaged path with window_size

    Args:
        prevs (_type_): [solution at t-N, ..., solution at t-1]
        new (_type_): new solution, [[s0,s1,s2,...],[d0,d1,d2,...]]
        window_size (int, optional): averaging window. Defaults to 2.
    """
    prevs_in_windows = prevs[-min(len(prevs),(window_size-1)):] # last index is the most recent
    dds = []
    
    # Interpolate "d"s in the previous path to match "s" in the new solution
    for prev in prevs_in_windows:
        prev_dd_interp = np.interp(new[0],prev[0],prev[1])
        dds.append(prev_dd_interp)
    
    # Get the average including the new path
    dds.append(new[1])
    sm_dd = np.mean(dds,axis=0)

    # Get the smoothed path
    return [new[0],sm_dd]

def FCP(state0, lb, ub, yr= 0, ds = 1, S = 50, lam_y =1, lam_U=1, lam_risk=1, lam_curve = 1000):
    N = int(S/ds)
    ell_r = 1.5
    ell_f = 1.5
    x0,y0,psi0 = state0[0],state0[1],state0[2]

    opti = casadi.Opti()
    X = opti.variable(3,N) # [x,y,psi]
    U = opti.variable(1,N-1) # [delta]
    x = X[0,:]
    y = X[1,:]
    psi = X[2,:]

    # Set dynamics
    for k in range(N-1):
        # Runge-Kutta 4 integration
        opti.subject_to(x[k+1]==x[k] + ds)
        opti.subject_to(y[k+1]==y[k] + ds * np.tan(psi[k]))
        opti.subject_to(psi[k+1]==psi[k] + 1/ell_r * ds * np.sin(U[k]*ell_r/(ell_f+ell_r))/np.cos(psi[k]))

        # opti.subject_to(1/ell_r*np.tan(U[k])*ds <= 0.01) # curvature limit 
        # opti.subject_to(1/ell_r*np.tan(U[k])*ds >= -0.01) # curvature limit
        
    # Set cost
    c_risk_boundary = 0
    c_curve = 0
    for k in range(N):             
        c_risk_boundary += 1/np.sqrt(np.sqrt((lb[k]-y[k])**2)) \
                        + 1/np.sqrt(np.sqrt((ub[k]-y[k])**2))
        if k < N-1:
            c_curve += np.tan(U[k])**2

    # Set initial conditions
    opti.subject_to(x[0] == x0)
    opti.subject_to(y[0] == y0)
    opti.subject_to(psi[0] == psi0)
        
    # Set boundaries
    opti.subject_to(opti.bounded(lb, y.T, ub)) # frenet corridor
    opti.subject_to(opti.bounded(-0.1, U.T, 0.1)) # steering limit

    # Optimization
    opti.minimize(lam_y * (y-yr)@(y-yr).T + lam_U*U@U.T + lam_risk*c_risk_boundary + lam_curve*c_curve)

    # Solve
    p_opts = {"expand": False}
    s_opts = {"max_iter": 100,'print_level':0}
    opti.solver("ipopt",p_opts,s_opts)
    sol = opti.solve()
    out = [sol.value(x),sol.value(y)]
    
    return out

def fcp_simple(state0, lb, ub, yr= 0, ds = 1, S = 50, consist_dist = 20,
                lam_y =0.1, lam_U=1, lam_risk=10, lam_curve = 1000, 
                lam_dyn_obs = 1000, lam_prev = 100, lam_slack=1e3, decay_alpha = 0.95,
                ell_f = 1.1, ell_r = 1.7,
                dyn_preds = [],prev_sol=False, return_time=False):
    """_summary_

    Args:
        state0 (_type_): ego state
        lb (_type_): lower bounds
        ub (_type_): upper bounds
        yr (int, optional): reference y. Defaults to 0.
        ds (int, optional): planning step. Defaults to 1.
        S (int, optional): planning horizon. Defaults to 50.
        consist_dist (int, optional): distance to keep consistent with previous solution. Defaults to 20.
        lam_y (float, optional): penalty on y. Defaults to 0.1.
        lam_U (int, optional): penalty on U. Defaults to 1.
        lam_risk (int, optional): penalty on getting close to boundaries. Defaults to 10.
        lam_curve (int, optional): penalty on curvatures of the path. Defaults to 1000.
        lam_dyn_obs (int, optional): penalty on getting close to dynamic obstacles. Defaults to 1000.
        lam_prev (int, optional): penalty on deviation from previous solution. Defaults to 100.
        lam_slack (_type_, optional): penalty on boundary slack variable. Defaults to 1e3.
        decay_alpha (float, optional): decaying factor for penalty on lam_dyn_obs. Defaults to 0.95.
        ell_f (float, optional): _description_. Defaults to 1.1.
        ell_r (float, optional): _description_. Defaults to 1.7.
        dyn_preds (list, optional): predictions of dynamic obstacles [obs1_pred, obs2_pred, ...]. Defaults to [].
        prev_sol (bool, optional): previous solution. Defaults to False.
        return_time (bool, optional): return NLP solver time. Defaults to False.

    Returns:
        _type_: optimal path in [[s0,s1,...],[d0,d1,...]]
    """
    N = int(S/ds)
    x0,y0,psi0 = state0[0],state0[1],state0[2]

    opti = casadi.Opti()
    X = opti.variable(4,N) # [x,y,psi,b_slack]
    U = opti.variable(1,N-1) # [delta]
    x = X[0,:]
    y = X[1,:]
    psi = X[2,:]
    b_slack = X[3,:]

    # Set dynamics
    for k in range(N-1):
        # Runge-Kutta 4 integration
        opti.subject_to(x[k+1]==x[k] + ds)
        opti.subject_to(y[k+1]==y[k] + ds * np.tan(psi[k]))
        opti.subject_to(psi[k+1]==psi[k] + 1/ell_r * ds * np.sin(U[k]*ell_r/(ell_f+ell_r))/np.cos(psi[k]))

        # opti.subject_to(1/ell_r*np.tan(U[k])*ds <= 0.01) # curvature limit 
        # opti.subject_to(1/ell_r*np.tan(U[k])*ds >= -0.01) # curvature limit
        
    # Set cost
    c_risk_boundary = 0
    c_dyn_risk = 0
    c_curve = 0
    c_prev_sol = 0
    for k in range(N):             
        # c_risk_boundary += 1/np.sqrt((lb[k]-y[k])**2) + 1/np.sqrt((ub[k]-y[k])**2)
        c_risk_boundary += ((lb[k]+ub[k])/2-y[k])**2
        if k < N-1:
            c_curve += np.tan(U[k])**2

        if dyn_preds:
            for j, pred in enumerate(dyn_preds):
                s_ind = round(x0+k*ds) == np.round(pred[:,0])
                if any(s_ind): # risk exists
                    if ds*k >= 40: # this threshold can change based on the current speed
                        decay = decay_alpha
                    else:
                        decay = 1
                    for p in pred[:,1][s_ind]:
                        c_dyn_risk += decay**(ds*k-40)/(p-y[k])**2
        
        if prev_sol and ds*k <= consist_dist:
            c_prev_sol += (y[k]-prev_sol[1][k])**2

    # Set initial conditions
    opti.subject_to(x[0] == x0)
    opti.subject_to(y[0] == y0)
    opti.subject_to(psi[0] == psi0)
    opti.subject_to(b_slack <= 1) # NOTE:: this must reflect the noise and margin
        
    # Set boundaries
    opti.subject_to(opti.bounded(lb-b_slack.T, y.T, ub+b_slack.T)) # frenet corridor
    opti.subject_to(opti.bounded(-0.25, U.T, 0.25)) # steering limit # TODO:: make the min max parametric

    # Optimization
    opti.minimize(lam_y * (y-yr)@(y-yr).T + lam_U*U@U.T 
                + lam_risk*c_risk_boundary + lam_curve*c_curve 
                + lam_dyn_obs*c_dyn_risk + lam_prev*c_prev_sol 
                + lam_slack*b_slack@b_slack.T)

    # Solve
    p_opts = {"expand": True,
             "ipopt.print_level": 0,
             "print_time": 0,
             "record_time": 1
             }

    s_opts = {"print_level": 0,
                "sb": "yes",
            #   "alpha_for_y": "primal-and-full",
                "max_iter": 250,
            #   "acceptable_tol": 1e-4,
            #   "nlp_scaling_method": "gradient-based",
            #   "bound_frac": 0.1,
            #   "bound_relax_factor": 1e-5,
            #   "acceptable_iter": 5,
            #    "bound_push": 0.1,
            #    "mu_strategy": "adaptive",
            # #   "hsllib": "/opt/coinhsl/lib/libcoinhsl.so",
                # "linear_solver": "ma57"
             }
    opti.solver("ipopt",p_opts,s_opts)
    # opti.solver("ipopt",{"expand": False},{"max_iter": 1000,'print_level':0})

    
    try:
        sol = opti.solve()   # actual solve

    except:
        sol = opti.debug
        print("Infeasible solution")
    # sol = opti.solve()
    
    if return_time:
        return [sol.value(x),sol.value(y)], opti.stats().get('t_wall_total')
    else:
        return [sol.value(x),sol.value(y)]
    

def get_visible_obss(ego,obss):
    
    # Compute the bounding box w.r.t the actual body shape
    (lb,ub,bboxs) = update_bounds(ego[0],obss, ref_ub = 5,which_bound_check=True,evaluate=True,buffer_scale=0)
    bboxs_trans = []
    for box in bboxs:
        bboxs_trans.append(np.transpose(box))
    bboxs_trans = np.array(bboxs_trans)

    # Find the occluded obstacles: 
    # - check if there is any obstacle between the ego and the obstacle
    #   if false, the obstacle is visible
    #   if true, do next
    # - check if the vectors from the ego center to (s(min_d),min_d) and (s(max_d),max_d) intersects with any bounding boxes in between
    #   if both intersect, the obstacle is not visible
    #   if not, the obstacle is visible
    # NOTE:: to be more accurate, we need to check all four edges, but it will incrase the computation time.
    obss_vis = []
    for i,(target,obs) in enumerate(zip(bboxs_trans,obss)): # target obstacle
        # print("Checking obstacle {}".format(i))
        target_s = np.mean(target[:,0])
        maxd = max(target[:,1]); maxs = target[target[:,1]==maxd,0][0]
        mind = min(target[:,1]); mins = target[target[:,1]==mind,0][0]
        target_max,target_min = [maxs,maxd], [mins,mind]
        max_ss = np.arange(ego[0],maxs,0.25)
        min_ss = np.arange(ego[0],mins,0.25)
        vec_max_dd = np.interp(max_ss, [ego[0],maxs],[ego[1],maxd])
        vec_min_dd = np.interp(min_ss, [ego[0],mins],[ego[1],mind])
        vec_max = [[s,d] for s,d in zip(max_ss, vec_max_dd)]
        vec_min = [[s,d] for s,d in zip(min_ss, vec_min_dd)]
        max_occ = False
        min_occ = False
        for j,test in enumerate(bboxs_trans):
            if i == j: continue
            # Make sure the test obstacle is in between
            test_s = np.mean(test[:,0])
            if test_s < ego[0] or test_s > target_s: # not in range
                # print("test obs {} is not in range".format(j))
                continue
            # For the obstacles in between, check the intersection of the two vectors
            for test_p in test:
                if not max_occ:
                    for p in vec_max:
                        dist = np.linalg.norm(np.array(p)-np.array(test_p))
                        if dist <= 0.3: # the vector is occluded
                            max_occ = True
                            # print("target {} max vec is occluded by obs {}".format(i,j))
                            break
                if not min_occ:
                    for p in vec_min:
                        dist = np.linalg.norm(np.array(p)-np.array(test_p))
                        if dist <= 0.3: # the vector is occluded
                            min_occ = True
                            # print("target {} min vec is occluded by obs {}".format(i,j))
                            break

        if max_occ and min_occ:
            # print("target {} is invisible".format(i))
            pass
        else:
            obss_vis.append(obs)
            # print("target {} is visible".format(i))
    return obss_vis


def vis_occlusions(ego,lb,ub,obss,obss_vis,bboxs,dyn_preds=[]):
    """Visualize scenarios with raytracing

    Args:
        ego (_type_): _description_
        lb (_type_): _description_
        ub (_type_): _description_
        obss (_type_): _description_
        obss_vis (_type_): _description_
        bboxs (_type_): _description_
        dyn_preds (list, optional): _description_. Defaults to [].
    """
    # Visualize the scenario
    vis_scenario(ego,lb,ub,obss,dyn_preds,bboxs);

    if len(dyn_preds) > 0:
        for pred in dyn_preds:
            obss_vis.append(pred[0])

    # Visualize field of view
    vds = 0.1
    vis_range = 50
    vis_N = int(vis_range/vds)
    bboxs_trans = []
    (lb,ub,bboxs) = update_bounds(ego[0],obss_vis, ref_ub = 5,which_bound_check=True,evaluate=True,buffer_scale=0)
    for box in bboxs:
        bboxs_trans.append(np.transpose(box))
    bboxs_trans = np.array(bboxs_trans)
    for angle in np.linspace(-np.pi/3,np.pi/3,51):
        # For each angle, generate
        vline = np.array([[ego[0]+i*vds*np.cos(ego[2]+angle), ego[1]+i*vds*np.sin(ego[2]+angle)] for i in range(vis_N)])

        # For each line, check if it intersects any obstacles
        min_ind = len(vline)
        
        for i,box in enumerate(bboxs_trans): # obstacle ahead the ego
            # if abs(angle-np.arctan((np.mean(box[1])-ego[1])/(np.mean(box[0])-ego[0]))) < np.pi/3: # only if the heading is on the same direction
            for j,p in enumerate(vline): # each point in vline
                for q in box:
                    dist = np.linalg.norm(np.array(p)-np.array(q))
                    if dist <= 0.3:
                        min_ind = j
                        # print("dist: {} to box {} j {} min_ind {}".format(dist,i,j,min_ind))
                        break
                if min_ind != len(vline):
                    break
            if min_ind != len(vline):
                break
        
        plt.plot(vline[:min_ind+1,0],vline[:min_ind+1,1],color = 'green',alpha=0.5);