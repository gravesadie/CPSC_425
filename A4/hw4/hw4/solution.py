def RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement):
    """
    This function takes in `matched_pairs`, a list of matches in indices
    and return a subset of the pairs using RANSAC.
    Inputs:
        matched_pairs: a list of tuples [(i, j)],
            indicating keypoints1[i] is matched
            with keypoints2[j]
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        *_agreement: thresholds for defining inliers, floats
    Output:
        largest_set: the largest consensus set in [(i, j)] format

    HINTS: the "*_agreement" definitions are well-explained
           in the assignment instructions.
    """
    assert isinstance(matched_pairs, list)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    ## START

    largest_set=[]
    rounds=10
    for i in range(rounds):
        consistent=[]
        rand = int(np.random.rand()*len(matched_pairs))
        match1 = matched_pairs[rand]
        key1, key2 = keypoints1[match1[0]], keypoints2[match1[1]]
        theta1 = (key1[3] - key2[3])*180/math.pi
        scale1 = key1[2]/key2[2]
        # check every other match for consistency
        for j in range(len(matched_pairs)):
            match2=matched_pairs[j]
            key1, key2 = keypoints1[match2[0]], keypoints2[match2[1]]
            theta2 = (key1[3] - key2[3])*180/math.pi
            scale2 = key1[2]/key2[2]
            # check orientation agreement
            orient_flag = abs(theta1 - theta2) < orient_agreement
            # check scale agreement
            if scale1 >= scale2:
                scale_flag = scale2 > scale1*(1 - scale_agreement)
            else:
                scale_flag = scale2 < scale1*(1 + scale_agreement)
            # check both flags and append match if good
            if orient_flag and scale_flag:
                consistent.append(match2)
        if len(consistent) > len(largest_set):
            largest_set = consistent
    
    ## END
    assert isinstance(largest_set, list)
    return largest_set


def FindBestMatches(descriptors1, descriptors2, threshold):
    """
    This function takes in descriptors of image 1 and image 2,
    and find matches between them. See assignment instructions for details.
    Inputs:
        descriptors: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
    Outputs:
        matched_pairs: a list in the form [(i, j)] where i and j means
                       descriptors1[i] is matched with descriptors2[j].
    """
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    # START
    matched_pairs=[]
    # compare each vector in descriptors1 to descriptors2
    for i in range(len(descriptors1)):
        vec1 = descriptors1[i]
        angles=[]
        for j in range(len(descriptors2)):
            vec2 = descriptors2[j]
            # find dot product
            dot = np.dot(vec1, vec2)
            # find angle
            angle = math.acos(dot)
            angles.append(angle)
    
        # sort angles and compare best and second best
        angles_s = sorted(angles)
        ratio = angles_s[0]/angles[1]
        if ratio < threshold:
            # append to pairs if above thresh
            vec2_index = angles.index(angles_s[0])
            matched_pairs.append(([i,vec2_index]))
    # END
    return matched_pairs


def KeypointProjection(xy_points, h):
    """
    This function projects a list of points in the source image to the
    reference image using a homography matrix `h`.
    Inputs:
        xy_points: numpy array, (num_points, 2)
        h: numpy array, (3, 3), the homography matrix
    Output:
        xy_points_out: numpy array, (num_points, 2), input points in
        the reference frame.
    """
    assert isinstance(xy_points, np.ndarray)
    assert isinstance(h, np.ndarray)
    assert xy_points.shape[1] == 2
    assert h.shape == (3, 3)
    # START
    # create homogeneous axis
    num_pts = xy_points.shape[0]
    new_dim = np.ones((num_pts, 1))
    xy_homo = np.hstack([xy_points,new_dim])
    res = []
    # project each point
    for vec in xy_homo:
        row = h.dot(vec)
        res.append(row)
    res = np.array(res)
    #res = np.array(np.matmul(h, xy_homo.T)).T
    # divide out third coord
    xy_points_out = []
    for row in res:
        z = row[2]
        # make non-zero
        if z == 0:
            z = 1e-10
        new_row = [row[0]/z, row[1]/z]
        xy_points_out.append(new_row)
    # END
    return np.array(xy_points_out)


def RANSACHomography(xy_src, xy_ref, num_iter, tol):
    """
    Given matches of keyponit xy coordinates, perform RANSAC to obtain
    the homography matrix. At each iteration, this function randomly
    choose 4 matches from xy_src and xy_ref.  Compute the homography matrix
    using the 4 matches.  Project all source "xy_src" keypoints to the
    reference image.  Check how many projected keyponits are within a `tol`
    radius to the coresponding xy_ref points (a.k.a. inliers).  During the
    iterations, you should keep track of the iteration that yields the largest
    inlier set. After the iterations, you should use the biggest inlier set to
    compute the final homography matrix.
    Inputs:
        xy_src: a numpy array of xy coordinates, (num_matches, 2)
        xy_ref: a numpy array of xy coordinates, (num_matches, 2)
        num_iter: number of RANSAC iterations.
        tol: float
    Outputs:
        h: The final homography matrix.
    """
    assert isinstance(xy_src, np.ndarray)
    assert isinstance(xy_ref, np.ndarray)
    assert xy_src.shape == xy_ref.shape
    assert xy_src.shape[1] == 2
    assert isinstance(num_iter, int)
    assert isinstance(tol, (int, float))
    tol = tol*1.0
    # START
    # begin iterating
    largest_set = [[],[]]
    best_h = []
    for i in range(num_iter):
        curr_set = [[],[]]
        # randomly choose 4 matches
        indices = np.random.choice(xy_src.shape[0], size=4, replace=False)
        #indices = [i, i-105, i-306, i-207]
        src_matches, ref_matches = xy_src[indices], xy_ref[indices]
        # find homography matrix
        h, _ = cv2.findHomography(src_matches, ref_matches)
        # project keypoints from src
        proj_pts = KeypointProjection(xy_src, h)
        # check each match distance against tol
        for j, xy in enumerate(proj_pts):
            distance = np.sqrt(np.sum((xy - xy_ref[j])**2))
            #distance = np.sqrt((xy[0] - xy_ref[j][0])**2 + (xy[1] - xy_ref[j][1])**2)
            if distance < tol:
                curr_set[0].append(np.array(xy_src[j]))
                curr_set[1].append(np.array(xy_ref[j]))
        # monitor largest set to pick best h
        if len(curr_set[0]) > len(largest_set[0]):
            largest_set = curr_set
    best_h, _ = cv2.findHomography(np.array(largest_set[0]), np.array(largest_set[1]))
    h = best_h
    # END
    assert isinstance(h, np.ndarray)
    assert h.shape == (3, 3)
    return h


def FindBestMatchesRANSAC(
        keypoints1, keypoints2,
        descriptors1, descriptors2, threshold,
        orient_agreement, scale_agreement):
    """
    Note: you do not need to change this function.
    However, we recommend you to study this function carefully
    to understand how each component interacts with each other.

    This function find the best matches between two images using RANSAC.
    Inputs:
        keypoints1, 2: keypoints from image 1 and image 2
            stored in np.array with shape (num_pts, 4)
            each row: row, col, scale, orientation
        descriptors1, 2: a K-by-128 array, where each row gives a descriptor
        for one of the K keypoints.  The descriptor is a 1D array of 128
        values with unit length.
        threshold: the threshold for the ratio test of "the distance to the nearest"
                   divided by "the distance to the second nearest neighbour".
                   pseudocode-wise: dist[best_idx]/dist[second_idx] <= threshold
        orient_agreement: in degrees, say 30 degrees.
        scale_agreement: in floating points, say 0.5
    Outputs:
        matched_pairs_ransac: a list in the form [(i, j)] where i and j means
        descriptors1[i] is matched with descriptors2[j].
    Detailed instructions are on the assignment website
    """
    orient_agreement = float(orient_agreement)
    assert isinstance(keypoints1, np.ndarray)
    assert isinstance(keypoints2, np.ndarray)
    assert isinstance(descriptors1, np.ndarray)
    assert isinstance(descriptors2, np.ndarray)
    assert isinstance(threshold, float)
    assert isinstance(orient_agreement, float)
    assert isinstance(scale_agreement, float)
    matched_pairs = FindBestMatches(
        descriptors1, descriptors2, threshold)
    matched_pairs_ransac = RANSACFilter(
        matched_pairs, keypoints1, keypoints2,
        orient_agreement, scale_agreement)
    return matched_pairs_ransac