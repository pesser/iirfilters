#!/bin/bash

tsvfile="sequential.stats"

if [ -a $tsvfile ]
then
    rm $tsvfile
fi

touch $tsvfile

for n in `seq 0 20 2000`
do    
    for((i=0;i<20;i+=1))
    do
         cd ..
         out=$(./time_seq_deriche $n)
         cd alternatives
         echo $out >> $tsvfile
         echo $out
    done
    
done
