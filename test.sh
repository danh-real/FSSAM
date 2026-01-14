dataset=sarcoma  # pascal/coco
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

CUDA_VISIBLE_DEVICES=1 python test.py \
        --config=${config} \
        --arch=${arch} \
        --num_refine=3 \
        --ver_refine=v1 \
        --ver_dino=dinov2_vitb14 \
        --episode=1000