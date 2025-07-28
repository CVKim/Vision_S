from visionsuite.engines.data.slicer.utils.functional import get_intersected_points, get_points_in_patch


def get_sliding_patches(
    img_width,
    img_height,
    patch_height,
    patch_width,
    points,
    overlap_ratio=0.2,
    num_involved_pixel=2,
    bg_ratio=-1,
    roi=[],
    include_point_positive=False,
    skip_highly_overlapped_tiles=False,
    filename=None,
):

    # TODO: Need to consider num_involved_pixel !!!!!!!!!
    # TODO: ignore the bg_ratio b/c there is already function for background_patches

    if len(roi) != 0 and roi is not None:
        tl_x, tl_y, br_x, br_y = roi[0], roi[1], roi[2], roi[3]
    else:
        tl_x, tl_y, br_x, br_y = 0, 0, img_width, img_height

    dx = int((1.0 - overlap_ratio) * patch_width)
    dy = int((1.0 - overlap_ratio) * patch_height)

    sliding_patches_rois = []
    sliding_patches_num_data = 0
    labels_in_patches = []
    for y0 in range(tl_y, br_y, dy):
        for x0 in range(tl_x, br_x, dx):
            if y0 + patch_height > br_y:
                if skip_highly_overlapped_tiles:  # skip if too much overlap (> 0.6)
                    if (y0 + patch_height - br_y) > (0.6 * patch_height):
                        continue
                    else:
                        y = br_y - patch_height
                else:
                    y = br_y - patch_height
            else:
                y = y0

            if x0 + patch_width > br_x:
                if skip_highly_overlapped_tiles:  # skip if too much overlap (> 0.6)
                    if (x0 + patch_width - br_x) > (0.6 * patch_width):
                        continue
                    else:
                        x = br_x - patch_width
                else:
                    x = br_x - patch_width
            else:
                x = x0

            xmin, xmax, ymin, ymax = x, x + patch_width, y, y + patch_height

            # when more than one point lies on the window, the window is considered patch
            is_part_in = []
            areas = []
            area = 0
            if len(points) > 0:
                for b in points:
                    for xb0, yb0 in b["points"]:
                        if (xb0 >= xmin) and (xb0 <= xmax) and (yb0 <= ymax) and (yb0 >= ymin):
                            is_part_in.append(1)
                        else:
                            is_part_in.append(0)

                    if all(is_part_in) or not all(is_part_in) and sum(is_part_in) != 0:
                        if len(b["points"]) > 2:
                            intersection_points, area = get_intersected_points(
                                [x, y, x + patch_width, y + patch_height],
                                b["points"],
                                is_points1_roi=True,
                                output_list=True,
                                get_area=True,
                                filename=filename,
                            )
                            areas.append(area)
                        elif 0 < len(b["points"]) and len(b["points"]) <= 2:
                            is_part_in.append(1)
                        else:
                            NotImplementedError(f"There is no such case considered When b['points'] is {b['points']}")

            flag_area_satisfied = False
            for _area in areas:
                if _area > num_involved_pixel:
                    flag_area_satisfied = True
                    break

            if not flag_area_satisfied:
                if sum(areas) > num_involved_pixel:
                    flag_area_satisfied = True

            if len(areas) == 0:
                flag_area_satisfied = True

            if sum(is_part_in) == 0 or not flag_area_satisfied:
                continue

            patch_coord = [x, y, x + patch_width, y + patch_height]
            points_in_patch = get_points_in_patch(
                points,
                patch_coord,
                [tl_x, tl_y, br_x, br_y],
                include_point_positive=include_point_positive,
                filename=filename,
            )
            if len(points_in_patch) != 0:
                labels_in_patches.append(points_in_patch)
                sliding_patches_rois.append(patch_coord)
                sliding_patches_num_data += 1

    return sliding_patches_rois, sliding_patches_num_data, labels_in_patches
