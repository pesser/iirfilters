#!/bin/bash

tsvfile="cuda.stats"

if [ -a $tsvfile ]
then
    rm $tsvfile
fi

touch $tsvfile

for n in `seq 0 20 2000`
do    
    total0=0
    total1=0
    totacc=0
    
    ntimes=10
    nacc=$ntimes
    for((i=0;i<${ntimes};++i))
    do
         out=$(./analyze-iir-cuda $n $n)
         stringarray=($out)
         time0=${stringarray[0]}
         time1=${stringarray[1]}
         acc=${stringarray[2]}

         total0=$(bc -l <<< "$total0 + $time0")
         total1=$(bc -l <<< "$total1 + $time1")

         if [ "$acc" = "nan" ]
         then
             nacc=$((nacc-1))
         else
             totacc=$(bc -l <<< "$totacc + $acc")
         fi
        
    done

    total0=$(bc -l <<< "$total0 / $ntimes")
    total1=$(bc -l <<< "$total1 / $ntimes")

    totacc=$(bc -l <<< "$totacc/$nacc")
    
    echo "$n $total0 $total1 $totacc" >> $tsvfile
    echo "$n $total0 $total1 $totacc"
done
