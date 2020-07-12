#!/bin/ksh
run_ts=$(date +"%m%d%y%H%M")
op_file2="/tmp/52_py2_runs_${run_ts}.txt"
op_file3="/tmp/52_py3_runs_${run_ts}.txt"
n=2000
for run in {1..${n}}
do
python2 -R -c 'print (hash("a")-hash("b"))' >> ${op_file2}
python3 -c 'print(hash("a")-hash("b"))' >> ${op_file3}
done
cnt_2=`cat ${op_file2} | sort | uniq | wc -l`
cnt_3=`cat ${op_file3} | sort | uniq | wc -l`
echo "Number of Collisions in randomized Python 2 is "$((${n}-${cnt_2}))" in ${n} runs."
echo "Number of Collisions in Python 3 is "$((${n}-${cnt_3}))" in ${n} runs."
exit 0