from visionsuite.engines.data.slicer.src.background_patches import get_background_patches
from visionsuite.engines.data.slicer.src.centric_patches import get_centric_patches
from visionsuite.engines.data.slicer.src.sliding_patches import get_sliding_patches
from visionsuite.engines.data.slicer.utils.functional import (
    get_annotations_from_labelme_json,
    get_available_points_from_annotations,
)
from visionsuite.engines.data.slicer.utils.polygons import handle_self_intersection


# TODO: Need to add logger
def get_imgs_info_from_patches(
    mode, img_file, classes, patch_info, roi=[], logger=None
):

    if logger is not None:
        logger.assertion_log(
            patch_info["sliding"] or patch_info["centric"],
            RuntimeError(
                f"If you want to use patch, need to choose at least one of sliding or centric"
            ),
            parent_fn=get_imgs_info_from_patches.__name__,
        )

    anns = get_annotations_from_labelme_json(img_file, logger=logger)
    img_width, img_height = anns["imageWidth"], anns["imageHeight"]
    num_data = 0
    rois = []
    background_patches_rois, background_patches_num_data = [], 0
    labels = []

    if len(anns["shapes"]) != 0:  # for positive samples

        # get candidate patch coordinates ---
        points = get_available_points_from_annotations(
            img_file, anns, mode, classes, roi, patch_info["include_point_positive"]
        )

        points = handle_self_intersection(points)

        # background patches ---
        # if patch_info['bg_ratio_by_image'] > 0:
        background_patches_rois, background_patches_num_data = get_background_patches(
            img_height=img_height,
            img_width=img_width,
            patch_height=patch_info['height'],
            patch_width=patch_info['width'],
            points=points,
            overlap_ratio=patch_info['overlap_ratio'],
            roi=roi,
            skip_highly_overlapped_tiles=False)

        if len(points) > 0:
            # centric patches ---
            if patch_info["centric"]:
                centric_patches_rois, centric_patches_num_data, rois_labels = (
                    get_centric_patches(
                        points,
                        patch_info["width"],
                        patch_info["height"],
                        img_width,
                        img_height,
                        roi=roi,
                        include_point_positive=patch_info["include_point_positive"],
                        filename=img_file,
                        logger=logger,
                    )
                )
                rois += centric_patches_rois
                num_data += centric_patches_num_data
                labels += rois_labels

                if logger is not None:
                    logger.assertion_log(
                        (len(rois) == num_data) and (len(labels) == num_data),
                        RuntimeError(
                            f"The number rois ({len(rois)}) should be same to the number of labels ({len(labels)})"
                        ),
                        parent_fn=get_imgs_info_from_patches.__name__,
                    )

            # sliding patches ---
            if patch_info["sliding"]:
                overlap_ratio = patch_info["overlap_ratio"]
                num_involved_pixel = patch_info["num_involved_pixel"]
                bg_ratio = patch_info["sliding_bg_ratio"]

                patch_coords, num_patch_sliding, rois_labels = get_sliding_patches(
                    img_height=img_height,
                    img_width=img_width,
                    patch_height=patch_info["height"],
                    patch_width=patch_info["width"],
                    points=points,
                    overlap_ratio=overlap_ratio,
                    num_involved_pixel=num_involved_pixel,
                    bg_ratio=bg_ratio,
                    roi=roi,
                    include_point_positive=patch_info["include_point_positive"],
                    skip_highly_overlapped_tiles=(mode != "train"),
                    filename=img_file,
                )
                # for patch_coord in patch_coords:
                #     try:
                #         assert patch_coord[2] - patch_coord[0] == patch_info['width'] and patch_coord[3] - patch_coord[
                #             1] == patch_info['height'], RuntimeError(f"patch coord ({patch_coord}) is wrong")
                #     except AssertionError as e:
                #         if logger is not None:
                #             logger.error(e, get_imgs_info_from_patches.__name__)
                #         raise e

                #     rois.append(patch_coord)
                rois += patch_coords
                num_data += num_patch_sliding
                labels += rois_labels

    # TODO: Need to consider empty json by using sliding patch?
    # else: # for negative samples
    #     if patch_info['sliding']:
    #         overlap_ratio=patch_info['overlap_ratio']
    #         num_involved_pixel=patch_info['num_involved_pixel']
    #         bg_ratio=patch_info['sliding_bg_ratio']
    #     else:
    #         overlap_ratio = 0.1
    #         num_involved_pixel = 10
    #         bg_ratio = 0

    #     patch_coords, num_patch_sliding, rois_labels = get_sliding_patches(img_height=img_height, img_width=img_width, \
    #                         patch_height=patch_info['height'], patch_width=patch_info['width'], points=[], \
    #                         overlap_ratio=overlap_ratio, num_involved_pixel=num_involved_pixel, \
    #                         bg_ratio=bg_ratio, roi=roi, include_point_positive=patch_info['include_point_positive'], \
    #                         skip_highly_overlapped_tiles=(mode is not 'train'), filename=img_file)

    #     for patch_coord, roi_labels in zip(patch_coords, rois_labels):
    #         try:
    #             assert patch_coord[2] - patch_coord[0] == patch_info['width'] and patch_coord[3] - patch_coord[1] == patch_info['height'], \
    #                     f"patch coord is wrong"
    #         except AssertionError as e:
    #             if logger is not None:
    #                 logger.error(e, get_imgs_info_from_patches.__name__)
    #             raise e

    #         rois.append(patch_coord)
    #         labels.append([roi_labels])
    #     num_data += num_patch_sliding

    return rois, num_data, background_patches_rois, labels
