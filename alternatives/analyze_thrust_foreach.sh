#!/bin/bash

tsvfile="thrust_foreach.stats"

if [ -a $tsvfile ]
then
    rm $tsvfile
fi

touch $tsvfile

for n in `seq 20 20 2000`
do    
    for((i=0;i<20;i+=1))
    do
         cd ..
         out=$(./time_thrust_deriche $n)
         cd alternatives
         echo $out >> $tsvfile
         echo $out
    done
    
done
