# CUDA_VISIBLE_DEVICES=6,7 python visualize_sparse_rcnn.py --num-images 4
# CUDA_VISIBLE_DEVICES=6,7 python visualize_sparse_rcnn.py --num-images 4 --img-dir /mnt/data/liyu/mm/data/coco_test_image
# CUDA_VISIBLE_DEVICES=6,7 python visualize_mask_rcnn.py --num-images 4 --conf-threshold 0.8 --output-dir visualizations/mask_rcnn_results_random4testimg
# CUDA_VISIBLE_DEVICES=6,7 python visualize_mask_rcnn.py --num-images 4 --conf-threshold 0.8 --img-dir /mnt/data/liyu/mm/data/coco_test_image

# CUDA_VISIBLE_DEVICES=6,7 python visualize_sparse_rcnn.py --num-images 4 --img-dir /mnt/data/liyu/mm/data/djw --output-dir visualizations/sparse_rcnn_results_djw
CUDA_VISIBLE_DEVICES=6,7 python visualize_mask_rcnn.py --num-images 4 --img-dir /mnt/data/liyu/mm/data/djw --output-dir visualizations/mask_rcnn_results_djw_coco --conf-threshold 0.8

