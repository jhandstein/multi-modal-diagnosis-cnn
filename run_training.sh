. .venv/bin/activate
> nohup.out  # This line truncates/empties the nohup.out file
nohup python3 train_model.py &
sleep 1
tail -f nohup.out