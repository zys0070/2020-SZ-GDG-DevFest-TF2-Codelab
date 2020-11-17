# Both sync and async mode are supported to run these object detection ssd samples.
# Batch mode with any batch size (need regenerate .xml IR) is also supported.
# 
# You can try different batch size and inference request number to compare the performance.
# Detected results(boxes on original images) are stored in current folder.


# Run object_detection_sample_ssd sample in batch=1 and sync mode

python object_detection_sample_ssd_batch_sync.py -m ../../models/ssd_mobilenet_v2_coco_tf2_ov_ir/FP32/ssd_mobilenet_v2_coco.xml -i ../../data/val2017_first100/ --labels ../../data/mscoco_labels.txt


# Run object_detection_sample_ssd sample in batch=4 and sync mode

python object_detection_sample_ssd_batch_sync.py -m ../../models/ssd_mobilenet_v2_coco_tf2_ov_ir/FP32/ssd_mobilenet_v2_coco_b4.xml -i ../../data/val2017_first100/ --labels ../../data/mscoco_labels.txt


# Run object_detection_sample_ssd sample in batch=1 and async mode

python object_detection_sample_ssd_batch_async.py -m ../../models/ssd_mobilenet_v2_coco_tf2_ov_ir/FP32/ssd_mobilenet_v2_coco.xml -i ../../data/val2017_first100/ --labels ../../data/mscoco_labels.txt


# Run object_detection_sample_ssd sample in batch=4 and async mode

python object_detection_sample_ssd_batch_async.py -m ../../models/ssd_mobilenet_v2_coco_tf2_ov_ir/FP32/ssd_mobilenet_v2_coco_b4.xml -i ../../data/val2017_first100/ --labels ../../data/mscoco_labels.txt
