log_file="time.log"
rm $log_file

for i in train_x_*_*.csv
do
    cp ${i} train_x.csv
    echo "${i} copied"

    cp ${i/x/y} train_y.csv
    echo "${i/x/y} copied"

    /data/local/tmp/bangwhe/LinearRegression_main >> $log_file

done