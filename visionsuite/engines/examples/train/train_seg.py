from visionsuite.engines.segmentation import Engine



if __name__ == "__main__":
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[2]
    
    engine = Engine('segmentation')
    engine.train(ROOT / "segmentation/cfgs/datasets/lx.yaml")  
    # engine.train(ROOT / "segmentation/cfgs/datasets/tenneco_inner.yaml")  
    # engine.train(ROOT / "segmentation/cfgs/datasets/sungwoo_bottom.yaml")    
    # engine.train(ROOT / "segmentation/cfgs/datasets/coco.yaml")    

