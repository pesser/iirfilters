#!/usr/bin/env bash

samples=$((4 * 128))

githash=$(git log --pretty=format:'%h' -n 1)
exe=time_thrust_deriche
log="data_${exe}_${githash}"

# header
echo "N,PreInit,Horizontal,Vertical,PostInit" > ${log}

# build
make $exe
# collect from 2**7 to 2**14
for((N=128;N<=16384;N*=2))
do
  for((sample=0;sample<samples;++sample))
  do
    echo $N
    ./${exe} $N >> ${log}
  done
  samples=$((samples / 2))
done

echo "Done."
