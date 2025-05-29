python ./voc2coco/voc2coco.py --ann_dir ./data/VOCdevkit/VOC2012/Annotations \
                   --seg_dir ./data/VOCdevkit/VOC2012/SegmentationClass \
                   --image_dir ./data/VOCdevkit/VOC2012/JPEGImages \
                   --output ./data/VOCdevkit/output_coco.json \
                   --auto_generate_labels