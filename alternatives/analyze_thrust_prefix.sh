#!/bin/bash

tsvfile="thrust_prefix.stats"

if [ -a $tsvfile ]
then
    rm $tsvfile
fi

touch $tsvfile

for n in `seq 0 20 2000`
do    
    for((i=0;i<20;i+=1))
    do
         out=$(./analyze-iir-thrust $n)
         echo $out >> $tsvfile
         echo $out
    done
done
