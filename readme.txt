FEAT + FGFL

paper: 
    Mini:   1-shot: 69.14 +- 0.80  5-shot: 86.01 +- 0.62
    Tiered: 1-shot: 73.21 +- 0.88  5-shot: 87.21 +- 0.61

MiniImageNet 5way-1shot

sigma: 0.5 omega:100 temp:12.5

acc: 70.89 +- 0.84 tag: propre15_featpre_1e-4_step20_ep200_notempgrad_nodefqmean_badtemp

python train_fsl.py --model_class GAIN_Feat --shot 1 --lr 0.0002 --step_size 20 --eval_shot 1 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-15w05s15q-Pre-SIM/10_0.5_lr0.001mul10_step_T164.0T264.0_b0.01_bsz300-NoAugproto_freq_pre_15w/max_acc.pth --gpu 2 --use_euclidean --tag propre15_featpre_1e-4_step20_ep200_notempgrad_nodefqmean_badtemp --init_weights ./saves/feat-1-shot.pth --episodes_per_epoch 200

acc: 70.87 +- 0.84 tag: propre15_featpre_1e-4_step20_ep200_notempgrad_nodefqmean

python train_fsl.py --model_class GAIN_Feat --shot 1 --lr 0.0002 --step_size 20 --eval_shot 1 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-15w05s15q-Pre-SIM/10_0.5_lr0.001mul10_step_T164.0T264.0_b0.01_bsz300-NoAugproto_freq_pre_15w/max_acc.pth --gpu 1 --use_euclidean --tag propre15_featpre_1e-4_step20_ep200_notempgrad_nodefqmean --init_weights ./saves/feat-1-shot.pth --episodes_per_epoch 200


MiniImageNet 5way-5shot

Pre-Proto-Freq: 70.51

python train_fsl.py --model_class ProtoNet --lr 0.001 --step_size 10 --init_weights ./MiniImageNet-Res12-Pre/0.1_0.1_84_fq_norm/max_acc_sim.pth --gpu 0 --way 10 --eval_way 10 --shot 10 --eval_shot 10 --tag proto_freq_pre_10w10s --episodes_per_epoch 600

balance: 0.1
temperature2: 32

acc:  tag: propre15_featpre_1e-4_step20_ep200_notempgrad_nodefqmean_badtemp gpu:3

python train_fsl.py --model_class GAIN_Feat --shot 5 --lr 0.0002 --step_size 20 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-15w05s15q-Pre-SIM/10_0.5_lr0.001mul10_step_T164.0T264.0_b0.01_bsz300-NoAugproto_freq_pre_15w/max_acc.pth --gpu 3 --use_euclidean --tag propre15_featpre_1e-4_step20_ep200_notempgrad_nodefqmean_badtemp --init_weights ./saves/feat-5-shot.pth --episodes_per_epoch 200 --balance 0.1 --temperature2 32

acc: tag: propre15_featpre_1e-4_step20_ep200_notempgrad_nodefqmean_badtemp_sig0_25 gpu:1

python train_fsl.py --model_class GAIN_Feat --shot 5 --lr 0.0002 --step_size 20 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-15w05s15q-Pre-SIM/10_0.5_lr0.001mul10_step_T164.0T264.0_b0.01_bsz300-NoAugproto_freq_pre_15w/max_acc.pth --gpu 1 --use_euclidean --init_weights ./saves/feat-5-shot.pth --episodes_per_epoch 200 --balance 0.1 --temperature2 32 --tag propre15_featpre_1e-4_step20_ep200_notempgrad_nodefqmean_badtemp_sig0_25

acc: 83.3 +- 0.14 -> 81.71 +- 0.58 tag: propre15_featpre_1e-4_step20_ep600_notempgrad_nodefqmean_badtemp_sig0_1

python train_fsl.py --model_class GAIN_Feat --shot 5 --lr 0.0002 --step_size 20 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth --gpu 3 --use_euclidean --init_weights ./saves/feat-5-shot.pth --episodes_per_epoch 600 --balance 0.1 --temperature2 32 --tag propre10_featpre_notempgrad_nodefqmean_badtemp_sig0_1


./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth

################################################################################################################################################################

0.1, 100, 12.5

acc: 84.36 +- 1.28  tag: onehot_enhnew


0.1, 10, 64 onehot 0.0002 40    acc:83.76
0.1, 1, 64 onehot 0.001 20      acc:84.00
0.1, 100, 12.5 onehot 0.0002 40 acc:84.36
0.1, 100, 12.5 onehot 0.0002 40 acc:84.48




python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 \
--init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-15w05s15q-Pre-SIM/10_0.5_lr0.001mul10_step_T164.0T264.0_b0.01_bsz300-NoAugproto_freq_pre_tete/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.1 --init_weights ./saves/feat-5-shot.pth \
--episodes_per_epoch 200 --lr 0.0001 --step_size 20 --gpu 0 --tag propre15_featpre_sig05_ome100_temp125_kd01


python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 \
--init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-15w05s15q-Pre-SIM/10_0.5_lr0.001mul10_step_T164.0T264.0_b0.01_bsz300-NoAugproto_freq_pre_tete/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.1 --init_weights ./saves/feat-5-shot.pth \
--episodes_per_epoch 600 --lr 0.0002 --step_size 40 --gpu 1 --tag propre15_featpre_sig05_ome100_temp125_kd01


python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 \
--init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-15w05s15q-Pre-SIM/10_0.5_lr0.001mul10_step_T164.0T264.0_b0.01_bsz300-NoAugproto_freq_pre_tete/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.1 --init_weights ./saves/feat-5-shot.pth \
--episodes_per_epoch 600 --lr 0.001 --step_size 20 --gpu 2 --tag propre15_featpre_sig01_ome100_temp125_kd01


python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 \
--init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-15w05s15q-Pre-SIM/10_0.5_lr0.001mul10_step_T164.0T264.0_b0.01_bsz300-NoAugproto_freq_pre_tete/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.1 --init_weights ./saves/feat-5-shot.pth \
--episodes_per_epoch 600 --lr 0.0001 --step_size 20 --gpu 3 --tag propre15_featpre_sig01_ome100_temp125_kd01


kd 0.5 t 32 w 0.01    val: 88.31 +-1.10 test: 85.00 +- 1.31

84.93 +- 1.38

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 10 --gpu 0 --max_epoch 200 --tag pre10_temp12.5

84.96 +- 1.35

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 10 --gpu 2 --max_epoch 200 --tag pre10_temp12.5_grl

84.75 +- 1.37

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.0001 --step_size 10 --gpu 2 --max_epoch 200 --tag pre10_temp12.5_grl

84.63

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.0001 --step_size 20 --gpu 3 --max_epoch 200 --tag pre10_temp64_grl_lr

84.95    ->184

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 10 --gpu 0 --max_epoch 200 --tag pre10_temp64_grl

85.16    ->184

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 10 --gpu 1 --max_epoch 200 --tag pre10_temp12.5_grl_mul1 --lr_mul 1

85.09    ->10

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.0001 --step_size 20 --gpu 2 --max_epoch 200 --tag pre10_temp12.5_grl_lr_mul1 --lr_mul 1

84.95    ->184

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 10 --gpu 3 --max_epoch 200 --tag pre10_temp12.5_grl_probs


84.83

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 40 --gpu 1 --max_epoch 200 --tag pre10_temp12.5_grl_mul1_step40 --lr_mul 1

85.13

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 10 --gpu 0 --max_epoch 200 --tag pre10_temp64_grl_mul1 --lr_mul 1

85.24

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 10 --gpu 2 --max_epoch 200 --tag pre10_temp12.5_grl_mul1_w001 --lr_mul 1

85.19

python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 10 --gpu 3 --max_epoch 200 --tag pre10_temp125_grl_mul1_sig10 --lr_mul 1






python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 10 --gpu 0 --max_epoch 200 --tag pre10_temp12.5_grl05_mul1_w001 --lr_mul 1



python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 10 --gpu 2 --max_epoch 200 --tag pre10_temp64_grl_mul1_sig10_w001 --lr_mul 1


python train_fsl.py --model_class GAIN_Feat --shot 5 --eval_shot 5 --init_weights2 ./checkpoints/MiniImageNet-ProtoNet-Res12-10w10s15q-Pre-SIM/10_0.5_lr0.0002mul10_step_T164.0T264.0_b0.01_bsz250-NoAugproto_freq_pre_10w10s/max_acc.pth \
--use_euclidean --temperature2 32 --balance 0.01 --init_weights ./saves/feat-5-shot.pth \
--lr 0.00005 --step_size 10 --gpu 3 --max_epoch 200 --tag pre10_temp125_grl_mul1_sig10_w001 --lr_mul 1


TieredImageNet 5way-1shot

Pre-Proto-Freq: 68.25 +- 0.23

python train_fsl.py --model_class ProtoNet --way 15 --shot 5 --lr 0.0001 --step_size 10 --init_weights ./TieredImagenet-Res12-Pre/0.01_0.1_0.01_0.1_jpegdct_nonorm_2/max_acc_sim.pth --gpu 0 --tag proto_freq_pre_15w --episodes_per_epoch 600 --dataset TieredImageNet


balance: 0.1 sigma: 0.5 omega:100 temp:12.5

acc: 77.43  tag: propre15_featpre_1e-4_step20_ep200_notempgrad_nodefqmean_badtemp_clamp

python train_fsl.py --model_class GAIN_Feat --shot 1 --lr 0.0001 --step_size 10 --eval_shot 1 --init_weights2 ./checkpoints/TieredImageNet-ProtoNet-Res12-15w05s15q-Pre-SIM/10_0.5_lr0.0001mul10_step_T164.0T264.0_b0.01_bsz300-NoAugproto_freq_pre_15w/max_acc.pth --gpu 1 --use_euclidean --tag propre15_featpre_1e-4_step20_ep600_notempgrad_nodefqmean_badtemp_sig0_1_clamp --init_weights ./saves/tiered_feat/feat-1-shot.pth --episodes_per_epoch 600 --dataset TieredImageNet --balance 0.01


TieredImageNet 5way-5shot

balance: 0.1 sigma: 0.5 omega:100


acc: 89.03 +- 1.39 acc_aug: 86.55 +- 1.43 balance: 0.1 temperature: 32 temp: 12.5 lr: 0.0002

python train_fsl.py --model_class GAIN_Feat --shot 5 --lr 0.0002 --step_size 20 --eval_shot 5 --init_weights2 ./checkpoints/TieredImageNet-ProtoNet-Res12-15w05s15q-Pre-SIM/10_0.5_lr0.0001mul10_step_T164.0T264.0_b0.01_bsz300-NoAugproto_freq_pre_15w/max_acc.pth --use_euclidean --init_weights ./saves/tiered_feat/feat-5-shot.pth --episodes_per_epoch 600 --dataset TieredImageNet --balance 0.1 --temperature2 32 --gpu 2 --tag temp125_pre_enhnew

acc: 87.75 +- 1.33 acc_aug: 85.15 +- 1.59 balance: 0.1 temperature: 32 temp: 12.5 lr: 0.0001

python train_fsl.py --model_class GAIN_Feat --shot 5 --lr 0.0001 --step_size 20 --eval_shot 5 --init_weights2 ./checkpoints/TieredImageNet-ProtoNet-Res12-15w05s15q-Pre-SIM/10_0.5_lr0.0001mul10_step_T164.0T264.0_b0.01_bsz300-NoAugproto_freq_pre_15w/max_acc.pth --use_euclidean --init_weights ./saves/tiered_feat/feat-5-shot.pth --episodes_per_epoch 600 --dataset TieredImageNet --balance 0.1 --temperature2 32 --gpu 2 --tag temp125_pre_enhnew_lowlr