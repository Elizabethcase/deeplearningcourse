#!/bin/bash

# prepares photos for labelling


#Grabs all photos & slices them into 500x500 bite size pieces

#only for GO PRO DISTORTED IMAGES
#mogrify -shave ‘500x500’ image name

echo Input photo directory

read dir

filelist=`cd $dir | ls | grep ".JPG"`
for file in $filelist
do
n=`convert $file -format %t info:`
convert $file -crop 512x512 ${n}_%d.JPG
echo $file chopped
done
