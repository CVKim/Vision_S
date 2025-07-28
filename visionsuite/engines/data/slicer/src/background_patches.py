def get_background_patches(
    img_width,
    img_height,
    patch_height,
    patch_width,
    points,
    overlap_ratio,
    roi=[],
    skip_highly_overlapped_tiles=True,
):

    # TODO: Need to ignore the segment of line for a negative sample?
    if len(roi) != 0 and roi is not None:
        tl_x, tl_y, br_x, br_y = roi[0], roi[1], roi[2], roi[3]
    else:
        tl_x, tl_y, br_x, br_y = 0, 0, img_width, img_height

    dx = int((1.0 - overlap_ratio) * patch_width)
    dy = int((1.0 - overlap_ratio) * patch_height)

    background_patches_rois = []  # x1y1x2y2
    background_patches_num_data = 0
    for y0 in range(tl_y, br_y, dy):
        for x0 in range(tl_x, br_x, dx):
            if y0 + patch_height > br_y:
                if skip_highly_overlapped_tiles:
                    if (y0 + patch_height - br_y) > (0.6 * patch_height):
                        continue
                    else:
                        y = br_y - patch_height
                else:
                    y = br_y - patch_height
            else:
                y = y0

            if x0 + patch_width > br_x:
                if skip_highly_overlapped_tiles:
                    if (x0 + patch_width - br_x) > (0.6 * patch_width):
                        continue
                    else:
                        x = br_x - patch_width
                else:
                    x = br_x - patch_width
            else:
                x = x0

            xmin, xmax, ymin, ymax = (
                x,
                x + patch_width,
                y,
                y + patch_height,
            )  # patch coordinates

            is_inside = False
            if len(points) > 0:
                for points_dict in points:
                    b = points_dict["points"]
                    for xb0, yb0 in b:
                        if (
                            (xb0 >= xmin)
                            and (xb0 <= xmax)
                            and (yb0 <= ymax)
                            and (yb0 >= ymin)
                        ):
                            is_inside = True
                            break
                    if is_inside:
                        break

            if is_inside:
                continue

            background_patches_rois.append([x, y, x + patch_width, y + patch_height])
            background_patches_num_data += 1

    return background_patches_rois, background_patches_num_data
