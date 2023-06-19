cur=`pwd`
cd $cur
nohup mlflow ui --host=0.0.0.0 > /dev/null 2>&1 &

set -e 
bash ./run_benchmark.sh -p 1 -b 2
bash ./run_benchmark.sh -p 2 -b 2
bash ./run_benchmark.sh -p 4 -b 2
bash ./run_benchmark.sh -p 8 -b 2

bash ./run_benchmark.sh -p 1 -b 4
bash ./run_benchmark.sh -p 2 -b 4
bash ./run_benchmark.sh -p 4 -b 4
bash ./run_benchmark.sh -p 8 -b 4

bash ./run_benchmark.sh -p 1 -b 6
bash ./run_benchmark.sh -p 2 -b 6
bash ./run_benchmark.sh -p 4 -b 6
bash ./run_benchmark.sh -p 8 -b 6
