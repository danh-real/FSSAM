dataset=msd_task04  # pascal/coco
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
  train_epoch_0_0.2486.pth
  train_epoch_1_0.2990.pth
  train_epoch_2_0.3284.pth
  train_epoch_3_0.3553.pth
  train_epoch_4_0.3862.pth
  train_epoch_5_0.4131.pth
  train_epoch_6_0.4343.pth
  train_epoch_7_0.4491.pth
  train_epoch_8_0.4621.pth
  train_epoch_9_0.4734.pth
  train_epoch_10_0.4857.pth
  train_epoch_11_0.4918.pth
  train_epoch_12_0.4975.pth
  train_epoch_13_0.5013.pth
  train_epoch_14_0.5070.pth
  train_epoch_16_0.5115.pth
  train_epoch_17_0.5162.pth
  train_epoch_19_0.5179.pth
  train_epoch_20_0.5203.pth
  train_epoch_21_0.5260.pth
  train_epoch_22_0.5270.pth
  train_epoch_23_0.5299.pth
  train_epoch_24_0.5301.pth
  train_epoch_25_0.5322.pth
  train_epoch_27_0.5337.pth
  train_epoch_28_0.5351.pth
  train_epoch_31_0.5371.pth
  train_epoch_32_0.5389.pth
  train_epoch_33_0.5392.pth
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