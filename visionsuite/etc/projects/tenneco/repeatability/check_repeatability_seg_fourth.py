import os.path as osp
import pandas as pd
from check_repeatability import run
import matplotlib.pyplot as plt

base_dir = '/HDD/etc/repeatablility/talos2/1st/benchmark/lps_w1120_h768'
cases = ['오염', '경계성', '딥러닝', 'repeated_ng', 'repeated_ok']


dir_names = ['']
filename_postfix = ''
rect_iou = True 
offset = 100

### result
result = {}

### ===================================
iou_thresholds = [0.05, 0.1, 0.2, 0.3]
area_thresholds = [20, 100, 150, 200]
figs = True 
vis = False
for case in cases:
    run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, 
        offset=offset, filename_postfix=filename_postfix, case=case)
### ===================================
iou_thresholds = [0.05]
area_thresholds = [20]
figs = False
vis = True
for case in cases:
    run(base_dir, dir_names, iou_thresholds, area_thresholds, vis, figs, rect_iou=rect_iou, 
        offset=offset, filename_postfix=filename_postfix, case=case)    
    
iou = 0.05
area = 20
res = {}
for case in cases:
    if case == '딥러닝 체크':
        _case = 'deeplearning false'
    elif case == '얼라인':
        _case = 'align'
    elif case == '경계성':
        _case = 'vague'
    else:
        _case = case

    res[_case] = {}
    for dir_name in dir_names:
        csv_file = osp.join(base_dir, dir_name, case, 'diff.csv')
        
        df = pd.read_csv(csv_file)
        data = df[f'iou{iou}_area{area}_conf0']
        no_diff_no_points = data[0]
        no_diff_points = data[1]
        diff_points = data[2]
        
        res[_case]['NO Defect'] = f'{data[0]} ({round(data[0]/(data[0] + data[1] + data[2])*100, 2)}%)'
        res[_case]['repeat OK'] = f'{data[1]} ({round(data[1]/(data[0] + data[1] + data[2])*100, 2)}%)'
        res[_case]['repeat Fail'] = f'{data[2]} ({round(data[2]/(data[0] + data[1] + data[2])*100, 2)}%)'
        res[_case]['Total'] = data[0] + data[1] + data[2]

    columns = list(res.keys())
    rows = list(next(iter(res.values())).keys())

    cell_text = []
    for row in rows:
        # cell_text.append([f"{res[col][row]:.3f}" for col in columns])
        cell_text.append([f"{res[col][row]}" for col in columns])

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.axis('off')

    table = ax.table(cellText=cell_text,
                    rowLabels=rows,
                    colLabels=columns,
                    loc='center',
                    cellLoc='center',
                    colLoc='center')

    table.scale(1, 1.5)
    plt.tight_layout()

    plt.savefig(osp.join(base_dir, dir_name, "table.png"), dpi=300)
            
