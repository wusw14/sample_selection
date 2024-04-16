# python preprocess.py --dataset AG > logs/emb/AG.log
# python preprocess.py --dataset BR > logs/emb/BR.log
# python preprocess.py --dataset DA > logs/emb/DA.log
# python preprocess.py --dataset DS > logs/emb/DS.log
# python preprocess.py --dataset FZ > logs/emb/FZ.log
# python preprocess.py --dataset WA > logs/emb/WA.log
# python preprocess.py --dataset AB > logs/emb/AB.log
# python preprocess.py --dataset watches > logs/emb/watches.log
# python preprocess.py --dataset cameras > logs/emb/cameras.log
# python preprocess.py --dataset shoes > logs/emb/shoes.log
# python preprocess.py --dataset computers > logs/emb/computers.log

# nohup python -u run_eval.py 2,3 --dataset FZ AG WA > logs/01.log 2>&1 &

# nohup python -u run_eval.py 2,3 --dataset WA DA AG DS > logs/23.log 2>&1 &
# nohup python -u run_eval.py 0,1 --dataset BR IA FZ watches cameras computers computers AB > logs/01.log 2>&1 &

# nohup python -u run_eval.py 2,3 --dataset BR IA cameras > logs/23.log 2>&1 &
# nohup python -u run_eval.py 4,5 --dataset FZ watches shoes computers > logs/45.log 2>&1 &



# nohup python -u run_eval.py 4,5 --dataset WA > logs/45.log 2>&1 &

# nohup python -u run_eval.py 6,7 llama2-70b --dataset BR WA DS AB DA FZ IA AG watches shoes cameras computers > logs/67.log 2>&1 &
# nohup python -u run_select.py 0,1,2,3,4,5 --dataset > logs/67.log 2>&1 &
nohup python -u run_eval.py 0 llama2-7b --dataset BR WA DS AB DA FZ IA AG watches shoes cameras computers > logs/0.log 2>&1 &
nohup python -u run_eval.py 1 llama2-7b --dataset BR WA DS AB DA FZ IA AG watches shoes cameras computers > logs/1.log 2>&1 &

nohup python -u run_eval.py 0 llama2-13b --dataset BR WA DS AB DA FZ IA AG watches shoes cameras computers > logs/0.log 2>&1 &
nohup python -u run_eval.py 1 llama2-13b --dataset BR WA DS AB DA FZ IA AG watches shoes cameras computers > logs/1.log 2>&1 &
nohup python -u run_eval.py 0,1 llama2-70b --dataset BR WA DS AB DA FZ IA AG watches shoes cameras computers > logs/01.log 2>&1 &
nohup python -u run_eval.py 2,3 llama2-70b --dataset BR WA DS AB DA FZ IA AG watches shoes cameras computers > logs/23.log 2>&1 &
nohup python -u run_eval.py 4,5 llama2-70b --dataset BR WA DS AB DA FZ IA AG watches shoes cameras computers > logs/45.log 2>&1 &
nohup python -u run_eval.py 6,7 llama2-70b --dataset BR WA DS AB DA FZ IA AG watches shoes cameras computers > logs/67.log 2>&1 &


nohup python -u run_eval.py 0 llama2-13b --dataset cameras computers FZ AB DA WA > logs/0.log 2>&1 &
nohup python -u run_eval.py 1 llama2-13b --dataset cameras computers FZ AB DA WA > logs/1.log 2>&1 &
nohup python -u run_eval.py 0,1 llama2-70b --dataset cameras computers FZ AB DA WA > logs/01.log 2>&1 &
nohup python -u run_eval.py 2,3 llama2-70b --dataset cameras computers FZ AB DA WA > logs/23.log 2>&1 &
nohup python -u run_eval.py 4,5 llama2-70b --dataset cameras computers FZ AB DA WA > logs/45.log 2>&1 &
nohup python -u run_eval.py 6,7 llama2-70b --dataset cameras computers FZ AB DA WA > logs/67.log 2>&1 &


nohup python -u run_eval.py 0 llama2-13b --dataset BR shoes > logs/0.log 2>&1 &
nohup python -u run_eval.py 1 llama2-13b --dataset WA watches > logs/1.log 2>&1 &
nohup python -u run_eval.py 2 llama2-13b --dataset DS > logs/2.log 2>&1 &
nohup python -u run_eval.py 3 llama2-13b --dataset IA cameras > logs/3.log 2>&1 &
nohup python -u run_eval.py 4 llama2-13b --dataset FZ computers > logs/4.log 2>&1 &
nohup python -u run_eval.py 4 llama2-13b --dataset FZ > logs/4.log 2>&1 &
nohup python -u run_eval.py 5 llama2-13b --dataset DA > logs/5.log 2>&1 &
nohup python -u run_eval.py 6 llama2-13b --dataset AG > logs/6.log 2>&1 &
nohup python -u run_eval.py 7 llama2-13b --dataset AB > logs/7.log 2>&1 &

nohup python -u run_eval.py 0 llama2-13b --dataset FZ  > logs/0.log 2>&1 &
nohup python -u run_eval.py 1 llama2-13b --dataset WA  > logs/1.log 2>&1 &
nohup python -u run_eval.py 2 llama2-13b --dataset DA DS IA WA > logs/2.log 2>&1 &
nohup python -u run_eval.py 3 llama2-13b --dataset DA DS IA WA  > logs/3.log 2>&1 &
nohup python -u run_eval.py 4 llama2-13b --dataset DA DS IA WA > logs/4.log 2>&1 &
nohup python -u run_eval.py 5 llama2-13b --dataset DA DS IA WA > logs/5.log 2>&1 &
nohup python -u run_eval.py 6 llama2-13b --dataset DA DS IA WA > logs/6.log 2>&1 &
nohup python -u run_eval.py 7 llama2-13b --dataset DA DS IA WA > logs/7.log 2>&1 &