import json
import os.path as osp
import warnings

import numpy as np
from visionsuite.utils.dataset.formats.labelme.utils import get_polygon_points_from_labelme_shape

from visionsuite.engines.data.slicer.utils.polygons import get_intersected_points


def get_roi_image(roi, img_width, img_height):
    if len(roi) != 0:
        tl_x, tl_y, br_x, br_y = roi[0], roi[1], roi[2], roi[3]
    else:
        tl_x, tl_y, br_x, br_y = 0, 0, img_width, img_height

    return [tl_x, tl_y, br_x, br_y]


def get_center_point_from_points(points):
    xs, ys = [], []
    for point in points:
        xs.append(point[0])
        ys.append(point[1])

    cx = (np.max(xs) - np.min(xs)) / 2 + np.min(xs)
    cy = (np.max(ys) - np.min(ys)) / 2 + np.min(ys)

    return [cx, cy]


def finetune_patch_for_roi(cx, cy, patch_width, patch_height, br_x, br_y, logger=None):
    br_offset_x = cx + patch_width / 2 - br_x
    br_offset_y = cy + patch_height / 2 - br_y
    if br_offset_x > 0:
        cx -= br_offset_x
    if br_offset_y > 0:
        cy -= br_offset_y

    tl_offset_x = cx - patch_width / 2
    tl_offset_y = cy - patch_height / 2
    if tl_offset_x < 0:
        cx -= tl_offset_x
    if tl_offset_y < 0:
        cy -= tl_offset_y

    patch_coord = [
        int(cx - patch_width / 2),
        int(cy - patch_height / 2),
        int(cx + patch_width / 2),
        int(cy + patch_height / 2),
    ]
    
    patch_coord[0] -= (patch_width - (patch_coord[2] - patch_coord[0]))
    patch_coord[2] -= (patch_width - (patch_coord[2] - patch_coord[0]))
    patch_coord[1] -= (patch_height - (patch_coord[3] - patch_coord[1]))
    patch_coord[3] -= (patch_height - (patch_coord[3] - patch_coord[1]))

    if logger is not None:
        logger.assertion_log(
            patch_coord[2] - patch_coord[0] == patch_width
            and patch_coord[3] - patch_coord[1] == patch_height,
            RuntimeError(f"patch coord is wrong"),
            parent_fn=finetune_patch_for_roi.__name__,
        )

    return patch_coord


def get_annotations_from_labelme_json(img_file, logger=None):
    json_file = osp.splitext(img_file)[0] + ".json"
    # TODO: how to log?
    try:
        with open(json_file) as jf:
            anns = json.load(jf)
    except Exception as error:
        raise RuntimeError(
            f"There is something wrong within {json_file} to load: {error}"
        )

    return anns


def get_available_points_from_annotations(
    img_file, anns, mode, classes, roi, include_point_positive
):
    points = []
    for shape in anns["shapes"]:
        shape_type = str(shape["shape_type"]).lower()
        label = shape["label"].lower()

        if label in classes or label.upper() in classes:  # get all points
            _points = get_polygon_points_from_labelme_shape(
                shape, shape_type, include_point_positive, mode
            )
            if len(_points) == 0:  # there is no points
                continue

            if len(roi) != 0:  # get intersected points
                _points = get_points_in_roi(_points, roi, filename=img_file)
                if len(_points) == 0:  # when out of roi
                    continue

                for _point in _points:
                    points.append(
                        {"points": _point, "label": label, "shape_type": shape_type}
                    )
            else:
                points.append(
                    {"points": _points, "label": label, "shape_type": shape_type}
                )

    return points


def is_point_in_roi(point, roi):

    assert isinstance(point, list) and not isinstance(point[0], list), ValueError(
        f"point({point}) should be 1 dimension list"
    )
    assert len(point) == 2, ValueError(
        f"The length of points should be 2, not {len(point)}"
    )
    assert isinstance(roi, list) and not isinstance(roi[0], list), ValueError(
        f"roi({roi}) should be 1 dimension list"
    )
    assert len(roi) == 4, ValueError(f"The length of roi should be 4, not {len(roi)}")

    if (
        roi[0] <= point[0]
        and point[0] <= roi[2]
        and roi[1] <= point[1]
        and point[1] <= roi[3]
    ):
        return True
    else:
        return False


def get_points_in_roi(points, roi, filename=None):
    assert isinstance(points, list) and isinstance(points[0], list), ValueError(
        f"points({points}) should be 2 dimension list"
    )

    if len(points) <= 2:  # for a point and a line
        points_in_roi = []
        for point in points:
            if is_point_in_roi(point, roi):
                points_in_roi.append(point)

        if len(points_in_roi) != 0:
            return [points_in_roi]
        else:
            return []

    else:  # for a polygon
        is_fully_in = True
        for point in points:
            if not is_point_in_roi(point, roi):
                is_fully_in = False
                # If at least one of point is out of roi, we need to get intersection points.
                intersected_ponits = get_intersected_points(
                    roi,
                    points,
                    is_points1_roi=True,
                    output_list=True,
                    filename=filename,
                )

                if len(intersected_ponits) == 0:
                    return []
                else:
                    points = intersected_ponits
                    is_fully_in = False
                    warnings.warn(
                        f"Object is split, because some of points are out of RoI for {filename}"
                    )
                    break

        if is_fully_in:
            return [points]
        else:
            return points


def get_points_in_patch(
    points, roi, image_roi, include_point_positive=False, filename=None
):
    points_in_patch = []
    for points_dict in points:
        _points = points_dict["points"]

        if len(_points) > 2:  # for positive samples,
            _points = get_points_in_roi(_points, roi, filename=filename)
            for _point in _points:
                points_in_patch.append(
                    {
                        "points": _point,
                        "label": points_dict["label"],
                        "shape_type": points_dict["shape_type"],
                        "roi": image_roi,
                    }
                )

        elif (len(_points) > 0 and len(_points) <= 2):
            if include_point_positive:
                _points = get_points_in_roi(_points, roi, filename=filename)
                if len(_points) != 0:
                    points_in_patch.append({"points": [], 'label': "", 'shape_type': "", 'roi': image_roi})
            else:
                pass

        else:
            raise NotImplementedError(f"There is no such case for {_points} points")

    return points_in_patch


# def get_points_in_patches(points, patches_coord, image_roi, include_point_positive=False, filename=None):
#     points_in_patches = []
#     for patch_coord in patches_coord:
#         points_in_patch = get_points_in_patch(points, patch_coord, image_roi, \
#                             include_point_positive=include_point_positive, filename=filename)

#         points_in_patches.append(points_in_patch)

#     return points_in_patches
