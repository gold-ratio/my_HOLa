import sys
sys.path.append('/mnt/disk2/qinqian/v-coco')
from vsrl_eval import VCOCOeval
# import utils

if __name__ == "__main__":
    vsrl_annot_file = "/mnt/disk2/qinqian/v-coco/data/vcoco/vcoco_test.json"
    coco_file = "/mnt/disk2/qinqian/v-coco/data/instances_vcoco_all_2014.json"
    split_file = "/mnt/disk2/qinqian/v-coco/data/splits/vcoco_test.ids"

    # Change this line to match the path of your cached file
    det_file = "/mnt/disk2/qinqian/ADA-CM" + sys.argv[1] # vcoco-injector/10/cache.pkl
    
    print(f"Loading cached results from {det_file}.")
    vcocoeval = VCOCOeval(vsrl_annot_file, coco_file, split_file)
    vcocoeval._do_eval(det_file, ovr_thresh=0.5)