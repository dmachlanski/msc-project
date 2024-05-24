path="/detop/temp/experiment2_force/"
eeg_path="/detop/temp/experiment2_force/processed/ear_ref/"
output="./plots_ear_ref/"

python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=4 --sess=1 --ds -o $output --type mixed fist mixed fist fist fist
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=4 --sess=2 --ds -o $output --type finger mixed fist fist
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=4 --sess=3 --ds -o $output --type fist fist fist fist
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=4 --sess=4 --ds -o $output --type fist fist fist
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=4 --sess=5 --ds -o $output --type fist fist fist fist

python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=5 --sess=1 --ds -o $output --type fist
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=5 --sess=2 --ds -o $output --type fist
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=5 --sess=4 --ds -o $output --type fist

python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=6 --sess=1 --ds -o $output --type fist mixed
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=6 --sess=2 --ds -o $output --type mixed finger
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=6 --sess=4 --ds -o $output --type mixed mixed
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=6 --sess=5 --ds -o $output --type mixed fist

python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=7 --sess=1 --ds -o $output --type finger mixed finger
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=7 --sess=2 --ds -o $output --type finger fist fist
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=7 --sess=3 --ds -o $output --type mixed fist
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=7 --sess=4 --ds -o $output --type fist finger
python -W ignore generate_plots.py -d $path --eeg $eeg_path -s=7 --sess=5 --ds -o $output --type finger finger fist fist