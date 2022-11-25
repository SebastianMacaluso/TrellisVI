#!/bin/bash
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=6-22:00:00
#SBATCH --mem=32GB
#SBATCH --job-name=paramTrellis
#SBATCH --mail-type=END
#SBATCH --mail-user=sm4511@nyu.edu
#SBATCH --output=logs/AmortizedFit_%a_%j.out
##SBATCH --gres=gpu:1

module purge

## executable
##SRCDIR=$HOME/ReclusterTreeAlgorithms/scripts
#
mainDIR=$SCRATCH/paramTrellis
cd $mainDIR
mkdir -p logs

restore_dir=None
#restore_dir=5_50_60_5_0.005_3n/

model_dir=experiments/amortized_posterior/
#model_dir=experiments/amortized_posterior_zi_fit/

#lr=1e-4

#lr=5e-3
lr=1e-2

#lr=5e-2
#lr=1e-1


#RNNsize=12
#RNNsize=20
RNNsize=50
#RNNsize=100

#bidirectional=False
bidirectional=True

#accumulation_steps=10
accumulation_steps=10

eval_Nsamples=4000
#eval_Nsamples=10000
#eval_Nsamples=50
### accumulation_steps* Nsamples = trees per epoch

#for minLeaves in 5
for minLeaves in 7
#for minLeaves in 8
#for minLeaves in 10
do
#  for Nsamples in 500
  for Nsamples in 300
  do
#    for epochs in 100
    for epochs in 1000
    do

      singularity exec --nv \
            --overlay /scratch/sm4511/pytorch1.7.0-cuda11.0.ext3:ro \
            /scratch/work/public/singularity/cuda11.0-cudnn8-devel-ubuntu18.04.sif \
            bash -c "source /ext3/env.sh; python $SCRATCH/paramTrellis/examples/run_paramTrellis.py --RNNsize=$RNNsize --bidirectional=$bidirectional --NleavesMin=$minLeaves --Nsamples=$Nsamples --epochs=$epochs --eval_Nsamples=${eval_Nsamples} --lr=$lr -accumulation_steps=${accumulation_steps} --save=False --model_dir=${model_dir} --restore_dir=${restore_dir}"


    done

  done
done


#python $SCRATCH/paramTrellis/examples/run_paramTrellis.py --NleavesMin=5 --Nsamples=4 --epochs=45 --eval_Nsamples=5 --lr=5e-3 -accumulation_steps=10 --save=False
## to submit(for 3 jobs): sbatch --array 0-2 submitHPC_paramTrellis.s

