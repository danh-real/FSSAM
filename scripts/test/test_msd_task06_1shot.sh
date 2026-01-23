dataset=msd_task06  # pascal/coco
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
  train_epoch_0
  train_epoch_1
  train_epoch_2
  train_epoch_3
  train_epoch_4
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