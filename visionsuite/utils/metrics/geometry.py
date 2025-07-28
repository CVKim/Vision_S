# import cv2 
# from visionsuite.utils.metrics.metrics import get_iou

# def merge_polygons(polygons, threshold=0.25):
    
#     while True:
#         polygon = polygons.pop(0)
        
#         candidate_indexes = []
#         for idx, candidate in enumerate(polygons):
#             iou = get_iou(polygon, candidate, 'polygon')
    
#             if iou > threshold:
#                 candidate_indexes.append(idx)

#         if                 
#         for candidate_index in candidate_indexes:
#             polygons.pop(candidate_index)

from shapely.geometry import Polygon

def close_polygon(polygon):
    if polygon[0] != polygon[-1]:
        polygon.append(polygon[0])
    return polygon

def calculate_iou(polygon1, polygon2):
    poly1 = Polygon(close_polygon(polygon1))
    poly2 = Polygon(close_polygon(polygon2))
    
    if not poly1.is_valid or not poly2.is_valid:
        return 0  # 유효하지 않은 폴리곤은 IoU 0으로 처리
    
    intersection_area = poly1.intersection(poly2).area
    union_area = poly1.union(poly2).area
    
    if union_area == 0:
        return 0
    
    return intersection_area / union_area

def merge_polygons(polygons, threshold=0.5):
    merged_polygons = []
    used = [False] * len(polygons)
    
    for i, polygon1 in enumerate(polygons):
        if used[i]:
            continue
        
        current_poly = Polygon(close_polygon(polygon1))
        for j, polygon2 in enumerate(polygons):
            if i != j and not used[j]:
                iou = calculate_iou(polygon1, polygon2)
                if iou > threshold:
                    current_poly = current_poly.union(Polygon(close_polygon(polygon2)))
                    used[j] = True
        
        merged_polygons.append(list(current_poly.exterior.coords))
        used[i] = True
    
    return merged_polygons