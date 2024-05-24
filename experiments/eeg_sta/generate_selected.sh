path="/detop/temp/experiment2_force/"
eeg_path="/detop/temp/experiment2_force/processed/newCAR/"
output="./plots_selected/"

python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=4 --sess=1 --ds -o $output --type mixed
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=6 --sess=5 --ds -o $output --type mixed