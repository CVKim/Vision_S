def check_label_roi(roi, label):

    for _label in label:
        points = _label["points"]
        for point in points:
            print(points, roi)
            assert roi[0] <= point[0] and point[0] <= roi[2], ValueError(
                f"Point ({point}) is out of roi ({roi})"
            )
            assert roi[1] <= point[1] and point[1] <= roi[3], ValueError(
                f"Point ({point}) is out of roi ({roi})"
            )
