python cifar10_download_and_extract.py --dara_dir ../data/cifar10 > download.txt

python cifar10_main.py --model_dir ../results_01_128_T --data_dir ../data/cifar10 --sparsity 0.1 --number_of_b 128 --shared_weights True > 01_128_T.txt
python cifar10_main.py --model_dir ../results_01_256_T --data_dir ../data/cifar10 --sparsity 0.1 --number_of_b 256 --shared_weights True > 01_256_T.txt
python cifar10_main.py --model_dir ../results_01_512_T --data_dir ../data/cifar10 --sparsity 0.1 --number_of_b 512 --shared_weights True > 01_512_T.txt

python cifar10_main.py --model_dir ../results_01_128_F --data_dir ../data/cifar10 --sparsity 0.1 --number_of_b 128 --shared_weights False > 01_128_F.txt
python cifar10_main.py --model_dir ../results_01_256_F --data_dir ../data/cifar10 --sparsity 0.1 --number_of_b 256 --shared_weights False > 01_256_F.txt
python cifar10_main.py --model_dir ../results_01_512_F --data_dir ../data/cifar10 --sparsity 0.1 --number_of_b 512 --shared_weights False > 01_512_F.txt

python cifar10_main.py --model_dir ../results_05_128_T --data_dir ../data/cifar10 --sparsity 0.5 --number_of_b 128 --shared_weights True > 05_128_T.txt
python cifar10_main.py --model_dir ../results_05_256_T --data_dir ../data/cifar10 --sparsity 0.5 --number_of_b 256 --shared_weights True > 05_256_T.txt
python cifar10_main.py --model_dir ../results_05_512_T --data_dir ../data/cifar10 --sparsity 0.5 --number_of_b 512 --shared_weights True > 05_512_T.txt

python cifar10_main.py --model_dir ../results_05_128_F --data_dir ../data/cifar10 --sparsity 0.5 --number_of_b 128 --shared_weights False > 05_128_F.txt
python cifar10_main.py --model_dir ../results_05_256_F --data_dir ../data/cifar10 --sparsity 0.5 --number_of_b 256 --shared_weights False > 05_256_F.txt
python cifar10_main.py --model_dir ../results_05_512_F --data_dir ../data/cifar10 --sparsity 0.5 --number_of_b 512 --shared_weights False > 05_512_F.txt

python cifar10_main.py --model_dir ../results_09_128_T --data_dir ../data/cifar10 --sparsity 0.9 --number_of_b 128 --shared_weights True > 09_128_T.txt
python cifar10_main.py --model_dir ../results_09_256_T --data_dir ../data/cifar10 --sparsity 0.9 --number_of_b 256 --shared_weights True > 09_256_T.txt
python cifar10_main.py --model_dir ../results_09_512_T --data_dir ../data/cifar10 --sparsity 0.9 --number_of_b 512 --shared_weights True > 09_512_T.txt

python cifar10_main.py --model_dir ../results_09_128_F --data_dir ../data/cifar10 --sparsity 0.9 --number_of_b 128 --shared_weights False > 09_128_F.txt
python cifar10_main.py --model_dir ../results_09_256_F --data_dir ../data/cifar10 --sparsity 0.9 --number_of_b 256 --shared_weights False > 09_256_F.txt
python cifar10_main.py --model_dir ../results_09_512_F --data_dir ../data/cifar10 --sparsity 0.9 --number_of_b 512 --shared_weights False > 09_512_F.txt


