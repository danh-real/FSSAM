dataset=msd_task04  # pascal/coco
exp_name=split0  # 0/1/2/3
shot=5  # 1/5
arch=FSSAM5s  # FSSAM
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
  train5_epoch_0_0.4481.pth
  train5_epoch_1_0.4889.pth
  train5_epoch_2_0.5078.pth
  train5_epoch_3_0.5222.pth
  train5_epoch_4_0.5327.pth
  train5_epoch_5_0.5421.pth
  train5_epoch_6_0.5487.pth
  train5_epoch_7_0.5550.pth
  train5_epoch_8_0.5588.pth
  train5_epoch_9_0.5604.pth
  train5_epoch_10_0.5658.pth
  train5_epoch_11_0.5681.pth
  train5_epoch_12_0.5721.pth
  train5_epoch_13_0.5745.pth
  train5_epoch_14_0.5764.pth
  train5_epoch_16_0.5796.pth
  train5_epoch_17_0.5811.pth
  train5_epoch_18_0.5850.pth
  train5_epoch_19_0.5854.pth
  train5_epoch_20_0.5874.pth
  train5_epoch_21_0.5886.pth
  train5_epoch_22_0.5899.pth
  train5_epoch_23_0.5911.pth
  train5_epoch_25_0.5921.pth
  train5_epoch_26_0.5931.pth
  train5_epoch_28_0.5931.pth
  train5_epoch_29_0.5936.pth
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