import os 
import os.path as osp 
import json 
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
from pathlib import Path

# ### 1st
# input_dir = '/HDD/etc/repeatablility/talos3/1st/benchmark'
# output_dir = '/HDD/etc/repeatablility/talos3/1st/benchmark'
# defects = ['오염', '시인성', '한도 경계성', '딥러닝 바보', 'repeated_ok']
# models = os.listdir(input_dir)


### 2nd
input_dir = '/HDD/etc/repeatablility/talos3/2nd/benchmark'
output_dir = '/HDD/etc/repeatablility/talos3/2nd/benchmark'
defects = ['오염', '시인성', '한도 경계성', '딥러닝 바보', '종횡비 경계성', '기타 불량', 'repeated_ok']
models = os.listdir(input_dir)


# ## etc
# input_dir = '/HDD/etc/repeatablility/talos2/1st/benchmark'
# output_dir = '/HDD/etc/repeatablility/talos2/1st/benchmark'
# defects = ['오염', '딥러닝', '경계성', 'repeated_ok', 'repeated_ng']
# models = os.listdir(input_dir)

ignore_models = ['summary.png', 'deeplabv3plus_w1120_h768']#, 'lps_w1120_h768', 'segformer_b2_unfrozen_w1120_h768_nohsv_tta', 'segnext_w1120_h768_tta']
print(models)

results = {}
for model in models:
    if model in ignore_models:
        continue 
    
    json_file = osp.join(input_dir, model, 'outputs/outputs.json')
    assert osp.exists(json_file), ValueError(f"THere is no such json-file: {json_file}")
    with open(json_file, 'r') as jf:
        result = json.load(jf)
        
    results[model] = {}
    
    for defect in defects:
        if defect == '오염':
            _defect = 'polluted'
        elif defect == '한도 경계성':
            _defect = 'spec in/out'
        elif defect == '종횡비 경계성':
            _defect = 'aspect in/out'
        elif defect == '시인성':
            _defect = 'visibility'
        elif defect == '경계성':
            _defect = 'vague'
        elif defect == '딥러닝 바보' or defect == '딥러닝':
            _defect = 'deeplearning'
        elif defect == '기타 불량':
            _defect = 'etc.'
        else:
            _defect = defect
        
        if defect in result:
            results[model][_defect] = result[defect]['repeated_percentage']
        else:
            results[model][_defect] = '-'
            
    
    results[model]['total'] = result['repeated_percentage']

results = dict(sorted(results.items()))

# 데이터프레임으로 변환
df = pd.DataFrame(results).T  # Transpose for better orientation

# '-'를 NaN으로 바꾸고 float 형으로 시도
df = df.replace('-', float('nan')).astype(float)

# 플롯 설정
fig, ax = plt.subplots(figsize=(14, 6))  # 크기 조절

# 테이블 생성
table = ax.table(cellText=df.round(2).values,
                 colLabels=df.columns,
                 rowLabels=df.index,
                 loc='center',
                 cellLoc='center',
                 colLoc='center')

# 그래프 영역 숨기기
ax.axis('off')

# 제목 추가 (선택 사항)
plt.title("Repeatabillty Benchmark (%)", fontsize=16, pad=20)

# 레이아웃 조정
plt.tight_layout()

# 이미지 저장
plt.savefig(osp.join(output_dir, "summary.png"), dpi=300)

    