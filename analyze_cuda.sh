#!/bin/bash

tsvfile="cuda.stats"

if [ -a $tsvfile ]
then
    rm $tsvfile
fi

touch $tsvfile

for n in `seq 0 100 2000`
do    
    total0=0
    total1=0
    totacc=0
    
    ntimes=100
    nacc=$ntimes
    for i in {1..$ntimes}
    do
         out=$(./analyze-iir-cuda $n $n)
         stringarray=($out)
         time0=${stringarray[0]}
         time1=${stringarray[1]}
         acc=${stringarray[2]}

         total0=$((total0 + time0))
         total1=$((total1 + time1))

         if [ "$acc" = "nan" ]
         then
             nacc=$((nacc-1))
         else
             totacc=$(bc -l <<< "$totacc + $acc")
         fi
        
    done

    total0=$((total0/ntimes))
    total1=$((total1/ntimes))

    totacc=$(bc -l <<< "$totacc/$nacc")
    
    echo "$n $total0 $total1 $totacc" >> $tsvfile
    echo "$n $total0 $total1 $totacc"
done
