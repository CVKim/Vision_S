from check_repeatability import run

base_dir = '/DeepLearning/etc/_athena_tests/benchmark/tenneco/outer/outputs/DETECTION'
dir_names = ['yolov7_w_epochs200']
rect_iou = True 
offset = 10
filename_postfix = ''

# ### ===================================
# iou_thresholds = [0.05, 0.1, 0.2, 0.3]
# area_thresholds = [10, 50, 100, 150, 200]
# figs = True 
# vis = False
# run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, offset=offset)
# ### ===================================
# iou_thresholds = [0.05]
# area_thresholds = [100]
# figs = False
# vis = True
# run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, offset=offset)
                

iou_thresholds = [0.05, 0.2, 0.4, 0.6]
area_thresholds = [10, 50, 100, 200, 300]
conf_thresholds = [0.1, 0.3, 0.5, 0.7]
figs = True
vis = False
run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, offset=offset,
    filename_postfix=filename_postfix, conf_thresholds=conf_thresholds)
                

iou_thresholds = [0.05]
area_thresholds = [100]
conf_thresholds = [0.5]
figs = False
vis = True
run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, offset=offset,
    filename_postfix=filename_postfix, conf_thresholds=conf_thresholds)
                
                            
            
                
                    
                    
               
                
    
    


        
    
        
    
    
    