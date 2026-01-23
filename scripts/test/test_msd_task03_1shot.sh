dataset=msd_task03  # pascal/coco
exp_name=split0  # 0/1/2/3
shot=1  # 1/5
arch=FSSAM  # FSSAM
net=tiny  # small

if [ $shot -eq 1 ]; then
  postfix=batch
elif [ $shot -eq 5 ]; then
  postfix=5s_batch
else
  echo "Only 1 and 5 shot are supported"
  exit 1
fi

config=config/${dataset}/${net}/${dataset}_${exp_name}_${net}_${postfix}.yaml

declare -a ckpt=(
  train_epoch_0_0.2693.pth
  train_epoch_1_0.3184.pth
  train_epoch_2_0.3464.pth
  train_epoch_3_0.3650.pth
  train_epoch_4_0.3767.pth
  train_epoch_5_0.3838.pth
  train_epoch_6_0.3870.pth
  train_epoch_7_0.3912.pth
  train_epoch_8_0.3942.pth
  train_epoch_9_0.3958.pth
  train_epoch_10_0.3974.pth
  train_epoch_11_0.3987.pth
  train_epoch_12_0.4012.pth
  train_epoch_14_0.4041.pth
  train_epoch_15_0.4043.pth
  train_epoch_16_0.4050.pth
  train_epoch_17_0.4062.pth
  train_epoch_19_0.4080.pth
  train_epoch_20_0.4085.pth
  train_epoch_22_0.4098.pth
  train_epoch_23_0.4104.pth
  train_epoch_25_0.4113.pth
  train_epoch_27_0.4119.pth
  train_epoch_28_0.4122.pth
  train_epoch_29_0.4130.pth
  train_epoch_31_0.4132.pth
  train_epoch_32_0.4140.pth
  train_epoch_34_0.4142.pth
)

for weight in ${ckpt[@]};
do
  python test.py \
    --config=${config} \
    --arch=${arch} \
    --num_refine=3 \
    --ver_refine=v1 \
    --ver_dino=dinov2_vitb14 \
    --opts Test_Finetune.weight $weight
done