set -ex

# DATASET=UTK

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/datasets/test_lite/avgs/utk_avg.jpg --K 1000 --dataset utk --save_vector --vector_path utk_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_UTK_label0.txt --K 100 --dataset utk --dataroot /media/ligong/Passport/Active/ws-gan/datasets/UTK --vector_path utk_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_UTK_label1.txt --K 100 --dataset utk --dataroot /media/ligong/Passport/Active/ws-gan/datasets/UTK --vector_path utk_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_UTK_label2.txt --K 100 --dataset utk --dataroot /media/ligong/Passport/Active/ws-gan/datasets/UTK --vector_path utk_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_UTK_label3.txt --K 100 --dataset utk --dataroot /media/ligong/Passport/Active/ws-gan/datasets/UTK --vector_path utk_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_UTK_label4.txt --K 100 --dataset utk --dataroot /media/ligong/Passport/Active/ws-gan/datasets/UTK --vector_path utk_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_YAN_label0.txt --K 20 --dataset yan --dataroot /media/ligong/Passport/Active/ws-gan/datasets/SCUT-FBP --vector_path yan_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_YAN_label1.txt --K 20 --dataset yan --dataroot /media/ligong/Passport/Active/ws-gan/datasets/SCUT-FBP --vector_path yan_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_YAN_label2.txt --K 20 --dataset yan --dataroot /media/ligong/Passport/Active/ws-gan/datasets/SCUT-FBP --vector_path yan_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_YAN_label3.txt --K 20 --dataset yan --dataroot /media/ligong/Passport/Active/ws-gan/datasets/SCUT-FBP --vector_path yan_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_YAN_label4.txt --K 20 --dataset yan --dataroot /media/ligong/Passport/Active/ws-gan/datasets/SCUT-FBP --vector_path yan_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_CACD_label0.txt --K 200 --dataset cacd --dataroot /media/ligong/Passport/Active/ws-gan/datasets/CACD --vector_path cacd_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_CACD_label1.txt --K 200 --dataset cacd --dataroot /media/ligong/Passport/Active/ws-gan/datasets/CACD --vector_path cacd_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_CACD_label2.txt --K 200 --dataset cacd --dataroot /media/ligong/Passport/Active/ws-gan/datasets/CACD --vector_path cacd_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_CACD_label3.txt --K 200 --dataset cacd --dataroot /media/ligong/Passport/Active/ws-gan/datasets/CACD --vector_path cacd_vec.npz

# python demo2_get_label.py /media/ligong/Passport/Active/ws-gan/sourcefiles/dfi_CACD_label4.txt --K 200 --dataset cacd --dataroot /media/ligong/Passport/Active/ws-gan/datasets/CACD --vector_path cacd_vec.npz

python demo2_label.py ../ws-gan/sourcefiles/CACD_sample.txt --output results/class_cacd --K 100 --dataset cacd --dataroot ../ws-gan/datasets/CACD --vector_path cacd_vec.npz --attr_bins 15 25 35 45 55 --load_size 200 --how_many 500

python demo2_label.py ../ws-gan/sourcefiles/UTK_sample.txt --output results/class_utk --K 100 --dataset utk --dataroot ../ws-gan/datasets/UTK --vector_path utk_vec.npz --attr_bins 10 30 50 70 90 --load_size 200 --how_many 500

python demo2_label.py ../ws-gan/sourcefiles/YAN_sample.txt --output results/class_yan --K 50 --dataset yan --dataroot /media/ligong/Passport/Active/ws-gan/datasets/SCUT-FBP --vector_path yan_vec.npz --attr_bins 1.3750 2.1250 2.8750 3.6250 4.5000 --load_size 200 --how_many 500
