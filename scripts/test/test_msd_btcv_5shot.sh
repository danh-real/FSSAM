dataset=btcv  # pascal/coco
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
  train5_epoch_0_0.2188.pth
  train5_epoch_1_0.2298.pth
  train5_epoch_2_0.2442.pth
  train5_epoch_3_0.2609.pth
  train5_epoch_4_0.2700.pth
  train5_epoch_5_0.2759.pth
  train5_epoch_6_0.2847.pth
  train5_epoch_7_0.2876.pth
  train5_epoch_8_0.2933.pth
  train5_epoch_9_0.3001.pth
  train5_epoch_10_0.3039.pth
  train5_epoch_11_0.3063.pth
  train5_epoch_12_0.3109.pth
  train5_epoch_13_0.3125.pth
  train5_epoch_14_0.3157.pth
  train5_epoch_15_0.3168.pth
  train5_epoch_16_0.3195.pth
  train5_epoch_17_0.3202.pth
  train5_epoch_19_0.3219.pth
  train5_epoch_20_0.3220.pth
  train5_epoch_22_0.3234.pth
  train5_epoch_24_0.3239.pth
  train5_epoch_25_0.3246.pth
  train5_epoch_26_0.3248.pth
  train5_epoch_27_0.3258.pth
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