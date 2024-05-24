for subject in {4..7}
do
    for session in {1..5}
    do
        echo "Subject $subject, session $session"
        python ../../track_sources.py -d "/detop/temp/experiment2_force/decomposition/final/" -s=$subject --sess=$session -g all -a=0.001 --align -r "/detop/temp/experiment2_force/processed/" --root_path "/home/damian/project/ce901_machlanski_d/" -v=0 -o "/detop/temp/experiment2_force/decomposition/tracking/run1/"
    done
done