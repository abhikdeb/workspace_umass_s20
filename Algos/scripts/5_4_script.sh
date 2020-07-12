#!/bin/ksh
run_ts=$(date +"%m%d%y%H%M")
op_file="/tmp/54_py2_runs_${run_ts}.txt"
n=$1
for run in {1..${n}}
do
python2 -R -c 'print (hash("8177111679642921702")-hash("6826764379386829346"))' >> $op_file
done
cnt=`cat ${op_file} | grep "^0$" | wc -l`
echo "Collisions occur ${cnt} times in ${n} runs with a probability of $((${cnt}*10000/${n})) e^(-4)."
exit 0