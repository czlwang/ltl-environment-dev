#!/bin/bash
for i in 0 1 2 3  
do
    #ffmpeg -r 2 -s 800x800 -f image2 -i videos/$i/%03d.png -vcodec libx264 -crf 25  -pix_fmt yuv420p out_videos/$i.mp4
    for j in 0 1 2
        do
        for k in 0 
            do
                mkdir -p out_gifs/$i/$j/$k;
                ffmpeg -y -r 3 -s 800x800 -f image2 -i videos_1/$i/$j/$k/%03d.png out_gifs/$i/$j/$k.gif;
            done
        done
done
