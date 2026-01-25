dataset=btcv  # pascal/coco
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
  train_epoch_0_0.1193.pth
  train_epoch_1_0.1338.pth
  train_epoch_2_0.1381.pth
  train_epoch_3_0.1420.pth
  train_epoch_4_0.1459.pth
  train_epoch_5_0.1473.pth
  train_epoch_6_0.1496.pth
  train_epoch_7_0.1527.pth
  train_epoch_8_0.1532.pth
  train_epoch_9_0.1559.pth
  train_epoch_10_0.1578.pth
  train_epoch_11_0.1602.pth
  train_epoch_12_0.1623.pth
  train_epoch_13_0.1650.pth
  train_epoch_14_0.1679.pth
  train_epoch_15_0.1697.pth
  train_epoch_16_0.1721.pth
  train_epoch_17_0.1735.pth
  train_epoch_18_0.1764.pth
  train_epoch_19_0.1773.pth
  train_epoch_21_0.1807.pth
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