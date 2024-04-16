# Preprocess
Map each sample into a vector by a sentence encoder  
```
python preprocess.py --dataset WA
```

# run the selection and evaluation for several datasets
For example, 
```
nohup python -u run_eval.py 0,1 llama2-70b --dataset cameras computers FZ AB DA WA > logs/01.log 2>&1 &
```

# run the selection 
```
CUDA_VISIBLE_DEVICES=0 python -u main.py --lm llama2-13b --gpus 0 --dataset cameras --selection_method ICESEM --budget 60 --k 10 --batch_size 1 --version 0416_test --order o7 --serialization s6 --metric f1  --argmax 
```

# run the evaluation
```
CUDA_VISIBLE_DEVICES=0 python -u evaluation.py --lm llama2-13b --gpus 0 --dataset cameras --selection_method ICESEM --budget 60 --k 10 --batch_size 1 --version 0416_test --order o7 --serialization s6 --metric f1  --argmax 
```