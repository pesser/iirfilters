#!/bin/bash

tsvfile="$1"

if [ -a $tsvfile ]
then
    rm $tsvfile
fi

touch $tsvfile

for n in 50 100 200 400 800 1600 3200 
do    
    total0=0
    total1=0
    totacc=0

    for i in {1..10}
    do
         out=$(./analyze-iir-cuda $n $n)
         stringarray=($out)
         time0=${stringarray[0]}
         total0=$((total0 + time0))
         time1=${stringarray[1]}
         total1=$((total1 + time1))
         acc=${stringarray[2]}
         totacc=$((totacc + acc))
    done

    total0=$(bc -l <<< '$total0/10')
    total1=$(bc -l <<< '$total1/10')
    totacc=$(bc -l <<< '$totacc/10')
    
    echo "$n $total0 $total1 $totacc" >> $tsvfile
    echo "$n $total0 $total1 $totacc"
done