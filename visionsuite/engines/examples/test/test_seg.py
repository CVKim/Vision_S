from visionsuite.engines.segmentation import Engine


if __name__ == "__main__":
    from pathlib import Path
    FILE = Path(__file__).resolve()
    ROOT = FILE.parents[2]
    
    runner = Engine('segmentation')
    runner.test(ROOT / "visionsuite/engines/segmentation/cfgs/datasets/sungwoo_bottom.yaml")    
    # runner.test(ROOT / "visionsuite/engines/segmentation/cfgs/datasets/coco.yaml")    

