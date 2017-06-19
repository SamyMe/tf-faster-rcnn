for conf_file in `ls experiments/confs`
do
    repeat=true
    cp experiments/confs/$conf_file lib/model/config.py
    while $repeat
    do
        python tools/trainval_net.py --imdb mappy_train --imdbval mappy_test --iters 15 --net vgg16 --weight data/imagenet_weights/vgg16.ckpt
        if (($?==0)); then
            repeat=false
        fi
    done
done

