# RNN model

# Sampling uniform, classification

#python src/train.py tmp/train1_unif_10 tmp/test1_unif_10 --validation tmp/validation_unif_10  tmp/statsontests_u10g2r -e 20 -d classification --cuda
#python src/train.py tmp/train1_unif_20 tmp/test1_unif_20 --validation tmp/validation_unif_20  tmp/statsontests_u20g2r -e 20 -d classification --cuda
#python src/train.py tmp/train1_unif_30 tmp/test1_unif_30 --validation tmp/validation_unif_30  tmp/statsontests_u30g2r -e 20 -d classification --cuda
#python src/train.py tmp/train1_unif_50 tmp/test1_unif_50 --validation tmp/validation_unif_50  tmp/statsontests_u50g2r -e 20 -d classification --cuda
#
## Sampling by similarity, classification
#
#python src/train.py tmp/train1_sim_10 tmp/test1_sim_10 --validation tmp/validation_sim_10  tmp/statsontests_s10g3r -e 20 -d classification --cuda
#python src/train.py tmp/train1_sim_20 tmp/test1_sim_20 --validation tmp/validation_sim_20  tmp/statsontests_s20g2r -e 20 -d classification --cuda
#python src/train.py tmp/train1_sim_30 tmp/test1_sim_30 --validation tmp/validation_sim_30  tmp/statsontests_s30g2r -e 20 -d classification --cuda
#python src/train.py tmp/train1_sim_50 tmp/test1_sim_50 --validation tmp/validation_sim_50  tmp/statsontests_s50g2r -e 20 -d classification --cuda
#
## Sampling uniform, 3 classes classification
#python src/train.py tmp/train1_unif_10 tmp/test1_unif_10 --validation tmp/validation_unif_10  tmp/statsontests_u10c3r -e 4 -d allinone --cuda
#python src/train.py tmp/train1_unif_20 tmp/test1_unif_20 --validation tmp/validation_unif_20  tmp/statsontests_u20c3r -e 4 -d allinone --cuda
#python src/train.py tmp/train1_unif_30 tmp/test1_unif_30 --validation tmp/validation_unif_30  tmp/statsontests_u30c3r -e 4 -d allinone --cuda
#python src/train.py tmp/train1_unif_50 tmp/test1_unif_50 --validation tmp/validation_unif_50  tmp/statsontests_u50c3r -e 4 -d allinone --cuda
#
## Sampling uniform, cleaning
#python src/train.py tmp/train1_unif_10 tmp/test1_unif_10 --validation tmp/validation_unif_10  tmp/statsontests_u10c2r -e 10 -d cleaning --cuda
#python src/train.py tmp/train1_unif_20 tmp/test1_unif_20 --validation tmp/validation_unif_20  tmp/statsontests_u20c2r -e 10 -d cleaning --cuda
#python src/train.py tmp/train1_unif_30 tmp/test1_unif_30 --validation tmp/validation_unif_30  tmp/statsontests_u30c2r -e 10 -d cleaning --cuda
#python src/train.py tmp/train1_unif_50 tmp/test1_unif_50 --validation tmp/validation_unif_50  tmp/statsontests_u50c2r -e 10 -d cleaning --cuda
#
## Sampling by similarity, 3 classes
#python src/train.py tmp/train1_sim_10 tmp/test1_sim_10 --validation tmp/validation_sim_10  tmp/statsontests_s10c3r -e 4 -d allinone --cuda
#python src/train.py tmp/train1_sim_20 tmp/test1_sim_20 --validation tmp/validation_sim_20  tmp/statsontests_s20c3r -e 4 -d allinone --cuda
#python src/train.py tmp/train1_sim_30 tmp/test1_sim_30 --validation tmp/validation_sim_30  tmp/statsontests_s30c3r -e 4 -d allinone --cuda
#python src/train.py tmp/train1_sim_50 tmp/test1_sim_50 --validation tmp/validation_sim_50  tmp/statsontests_s50c3r -e 4 -d allinone --cuda
#
## Sampling by similarity, cleaning
#python src/train.py tmp/train1_sim_10 tmp/test1_sim_10 --validation tmp/validation_sim_10  tmp/statsontests_s10c2r -e 10 -d cleaning --cuda
#python src/train.py tmp/train1_sim_20 tmp/test1_sim_20 --validation tmp/validation_sim_20  tmp/statsontests_s20c2r -e 10 -d cleaning --cuda
#python src/train.py tmp/train1_sim_30 tmp/test1_sim_30 --validation tmp/validation_sim_30  tmp/statsontests_s30c2r -e 10 -d cleaning --cuda
#python src/train.py tmp/train1_sim_50 tmp/test1_sim_50 --validation tmp/validation_sim_50  tmp/statsontests_s50c2r -e 10 -d cleaning --cuda

# fixed CNN model

# Sampling uniform, classification

python src/train.py tmp/train1_unif_10 tmp/test1_unif_10 --validation tmp/validation_unif_10 tmp/statsontests_u10g2c -e 20 -d classification -s 10 --cuda
python src/train.py tmp/train1_unif_20 tmp/test1_unif_20 --validation tmp/validation_unif_20 tmp/statsontests_u20g2c -e 20 -d classification  -s 20 --cuda
python src/train.py tmp/train1_unif_30 tmp/test1_unif_30 --validation tmp/validation_unif_30 tmp/statsontests_u30g2c -e 20 -d classification -s 30 --cuda
python src/train.py tmp/train1_unif_50 tmp/test1_unif_50 --validation tmp/validation_unif_50 tmp/statsontests_u50g2c -e 20 -d classification -s 50 --cuda

# Sampling by similarity, classification

python src/train.py tmp/train1_sim_10 tmp/test1_sim_10 --validation tmp/validation_sim_10 tmp/statsontests_s10g3c -e 20 -d classification -s 10 --cuda
python src/train.py tmp/train1_sim_20 tmp/test1_sim_20 --validation tmp/validation_sim_20 tmp/statsontests_s20g2c -e 20 -d classification -s 20 --cuda
python src/train.py tmp/train1_sim_30 tmp/test1_sim_30 --validation tmp/validation_sim_30 tmp/statsontests_s30g2c -e 20 -d classification -s 30 --cuda
python src/train.py tmp/train1_sim_50 tmp/test1_sim_50 --validation tmp/validation_sim_50 tmp/statsontests_s50g2c -e 20 -d classification -s 50 --cuda

# Sampling uniform, 3 classes classification
python src/train.py tmp/train1_unif_10 tmp/test1_unif_10 --validation tmp/validation_unif_10 tmp/statsontests_u10c3c -e 4 -d allinone -s 10 --cuda
python src/train.py tmp/train1_unif_20 tmp/test1_unif_20 --validation tmp/validation_unif_20 tmp/statsontests_u20c3c -e 4 -d allinone -s 20 --cuda
python src/train.py tmp/train1_unif_30 tmp/test1_unif_30 --validation tmp/validation_unif_30 tmp/statsontests_u30c3c -e 4 -d allinone -s 30 --cuda
python src/train.py tmp/train1_unif_50 tmp/test1_unif_50 --validation tmp/validation_unif_50 tmp/statsontests_u50c3c -e 4 -d allinone -s 50 --cuda

# Sampling uniform, cleaning
python src/train.py tmp/train1_unif_10 tmp/test1_unif_10 --validation tmp/validation_unif_10 tmp/statsontests_u10c2c -e 10 -d cleaning -s 10 --cuda
python src/train.py tmp/train1_unif_20 tmp/test1_unif_20 --validation tmp/validation_unif_20 tmp/statsontests_u20c2c -e 10 -d cleaning -s 20 --cuda
python src/train.py tmp/train1_unif_30 tmp/test1_unif_30 --validation tmp/validation_unif_30 tmp/statsontests_u30c2c -e 10 -d cleaning -s 30 --cuda
python src/train.py tmp/train1_unif_50 tmp/test1_unif_50 --validation tmp/validation_unif_50 tmp/statsontests_u50c2c -e 10 -d cleaning -s 50 --cuda

# Sampling by similarity, 3 classes
python src/train.py tmp/train1_sim_10 tmp/test1_sim_10 --validation tmp/validation_sim_10 tmp/statsontests_s10c3c -e 4 -d allinone -s 10 --cuda
python src/train.py tmp/train1_sim_20 tmp/test1_sim_20 --validation tmp/validation_sim_20 tmp/statsontests_s20c3c -e 4 -d allinone -s 20 --cuda
python src/train.py tmp/train1_sim_30 tmp/test1_sim_30 --validation tmp/validation_sim_30 tmp/statsontests_s30c3c -e 4 -d allinone -s 30 --cuda
python src/train.py tmp/train1_sim_50 tmp/test1_sim_50 --validation tmp/validation_sim_50 tmp/statsontests_s50c3c -e 4 -d allinone -s 50 --cuda

# Sampling by similarity, cleaning
python src/train.py tmp/train1_sim_10 tmp/test1_sim_10 --validation tmp/validation_sim_10 tmp/statsontests_s10c2c -e 10 -d cleaning -s 10 --cuda
python src/train.py tmp/train1_sim_20 tmp/test1_sim_20 --validation tmp/validation_sim_20 tmp/statsontests_s20c2c -e 10 -d cleaning -s 20 --cuda
python src/train.py tmp/train1_sim_30 tmp/test1_sim_30 --validation tmp/validation_sim_30 tmp/statsontests_s30c2c -e 10 -d cleaning -s 30 --cuda
python src/train.py tmp/train1_sim_50 tmp/test1_sim_50 --validation tmp/validation_sim_50 tmp/statsontests_s50c2c -e 10 -d cleaning -s 50 --cuda


