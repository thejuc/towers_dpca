#!/usr/bin/env bash

cd /jukebox/witten/yousuf/rotation/pickles2/loop_files/
dirlist=(`ls ${prefix}*.pickle`)

for i in "${dirlist[@]}"
do 
    cd /jukebox/witten/yousuf/rotation/pickles2/loop_files/
    echo "$i"
    fname=$i
    path="/jukebox/witten/yousuf/rotation/pickles2/loop_files/"
    fpath="$path$fname"
    echo $fpath
    filesize=$(stat -c%s "$i")
    filesize=$(($filesize/1000000000))
    echo $filesize
    mem=$(($filesize*3/2+2))
    mem="$mem"G""
    echo $mem
    n_ten=$(($filesize/10))
    t=$((15+25*$n_ten))
    t="$t"
    echo $t
    cd /jukebox/witten/yousuf/rotation/towers_dpca_repo/
    sbatch --mem-per-cpu=$mem --time=00:$t:00 --job-name=$fname dpca_batch.sh $fpath $fname $mem $t
done

