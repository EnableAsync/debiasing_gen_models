gpu="7"
config="custom.yml"

timesteps="1,50"
attribute_list="1,1,0,0"
scale=15,15
attribute=1
meh="vanilla"


CUDA_VISIBLE_DEVICES=$gpu python main.py --run_test                         \
                        --config $config                                    \
                        --exp ./runs/${attribute_list}/${meh}                               \
                        --n_test_img 10000                                 \
                        --seed $RANDOM                                        \
                        --bs_test 100                                         \
                        --savepath ""   \
                        --male $attribute      \
                        --timestep_list $timesteps    \
                        --scale $scale   \
                        --attribute_list $attribute_list    \
                        --male 0.5 \
                        --eyeglasses 0.5


