cd ..
cd adding-submodule-utils/bin/
source activate
cd ..
cd ..
cd DSS-Quant-Factors/
cd utils/
export $(grep -v '^#' .env | xargs -d '\n') 
cd ..
export PYTHONPATH=$(pwd)
export DB_USERNAME=$1
export DB_PASSWORD=$2
export DEBUG=$3

