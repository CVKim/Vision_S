import numpy as np

from visionsuite.engines.data.slicer.utils.functional import (
    finetune_patch_for_roi,
    get_center_point_from_points,
    get_points_in_patch,
    get_roi_image,
    is_point_in_roi,
)

from .sliding_patches_v2 import get_sliding_patches_v2


def get_centric_patches(
    points,
    patch_width,
    patch_height,
    img_width,
    img_height,
    roi=[],
    include_point_positive=False,
    filename=None,
    logger=None,
):

    # TODO: Need to consider number of pixels overlapped b/t patch and points b/c of the finetuning?

    [tl_x, tl_y, br_x, br_y] = get_roi_image(roi, img_width, img_height)

    if logger is not None:
        logger.assertion_log(
            patch_width <= br_x - tl_x,
            RuntimeError(f"patch width({patch_width}) should be bigger than width({br_x - tl_x})"),
            parent_fn=get_centric_patches.__name__,
        )
        logger.assertion_log(
            patch_height <= br_y - tl_y,
            RuntimeError(f"patch height({patch_height}) should be bigger than height({br_y - tl_y})"),
            parent_fn=get_centric_patches.__name__,
        )

    centric_patches_coord = []
    centric_patches_num_data = 0
    labels_in_patches = []

    # for points_dict in points:
    #     _points = points_dict["points"]

    #     [cx, cy] = get_center_point_from_points(_points)

    #     if len(_points) == 1:
    #         x, y = _points[0]
    #         if x < tl_x:
    #             x = tl_x
    #         elif x > br_x:
    #             x = br_x

    #         if y < tl_y:
    #             y = tl_y
    #         elif y > br_y:
    #             y = br_y
    #         cx, cy = x, y

    #     # if roi is not None and not is_point_in_roi([cx, cy], roi): # just for in case,
    #     #     return [], 0, []
    #     if logger is not None:
    #         logger.assertion_log(
    #             is_point_in_roi([cx, cy], [tl_x, tl_y, br_x, br_y]),
    #             RuntimeError(f"Center point({cx, cy}) of points must not be out of RoI({roi})"),
    #             parent_fn=get_centric_patches.__name__,
    #         )

    #     patch_coord = finetune_patch_for_roi(cx, cy, patch_width, patch_height, br_x, br_y, logger)
    #     points_in_patch = get_points_in_patch(
    #         points,
    #         patch_coord,
    #         [tl_x, tl_y, br_x, br_y],
    #         include_point_positive=include_point_positive,
    #         filename=filename,
    #     )
    #     if len(points_in_patch) != 0:
    #         labels_in_patches.append(points_in_patch)
    #         centric_patches_coord.append(patch_coord)
    #         centric_patches_num_data += 1

    # return centric_patches_coord, centric_patches_num_data, labels_in_patches

    for points_dict in points:
        _points = points_dict["points"]

        _xs, _ys = [], []
        for _point in _points:
            _xs.append(_point[0])
            _ys.append(_point[1])

        width = np.max(_xs) - np.min(_xs)
        height = np.max(_ys) - np.min(_ys)

        if width >= patch_width or height >= patch_height:
            roi_margin_x, roi_margin_y = 50, 50
            roi_x1 = int(min(tl_x, np.min(_xs)))
            roi_y1 = int(min(tl_y, np.min(_ys)))
            if roi_x1 - roi_margin_x > tl_x:
                roi_x1 -= roi_margin_x
            else:
                roi_x1 -= roi_margin_x - abs(roi_x1 - roi_margin_x - tl_x)

            if roi_y1 - roi_margin_y > tl_x:
                roi_y1 -= roi_margin_y
            else:
                roi_y1 -= roi_margin_y - abs(roi_y1 - roi_margin_y - tl_y)

            roi_x2 = int(max(br_x, np.max(_xs)))
            roi_y2 = int(max(br_y, np.max(_ys)))
            if roi_x2 + roi_margin_x < br_x:
                roi_x2 += roi_margin_x
            else:
                roi_x2 += roi_margin_x - abs(roi_x2 + roi_margin_x - br_x)

            if roi_y2 + roi_margin_y < br_y:
                roi_y2 += roi_margin_y
            else:
                roi_y2 += roi_margin_y - abs(roi_y2 + roi_margin_y - br_y)

            patch_coord, patches_num_data, points_in_patch = get_sliding_patches_v2(
                0,
                0,
                patch_height,
                patch_width,
                # [points_dict],
                points,
                roi=[roi_x1, roi_y1, roi_x2, roi_y2],
            )

            if len(points_in_patch) != 0:
                labels_in_patches += points_in_patch
                centric_patches_coord += patch_coord
                centric_patches_num_data += patches_num_data

        else:
            [cx, cy] = get_center_point_from_points(_points)

            if len(_points) == 1:
                x, y = _points[0]
                if x < tl_x:
                    x = tl_x
                elif x > br_x:
                    x = br_x

                if y < tl_y:
                    y = tl_y
                elif y > br_y:
                    y = br_y
                cx, cy = x, y

            if logger is not None:
                logger.assertion_log(
                    is_point_in_roi([cx, cy], [tl_x, tl_y, br_x, br_y]),
                    RuntimeError(f"Center point({cx, cy}) of points must not be out of RoI({roi})"),
                    parent_fn=get_centric_patches.__name__,
                )

            patch_coord = finetune_patch_for_roi(cx, cy, patch_width, patch_height, br_x, br_y, logger)
            points_in_patch = get_points_in_patch(
                points,
                patch_coord,
                [tl_x, tl_y, br_x, br_y],
                include_point_positive=include_point_positive,
                filename=filename,
            )
            if len(points_in_patch) != 0:
                labels_in_patches.append(points_in_patch)
                centric_patches_coord.append(patch_coord)
                centric_patches_num_data += 1

    return centric_patches_coord, centric_patches_num_data, labels_in_patches
