#!/bin/bash
#SBATCH --job-name=telme-run
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=output/telme-recent.out
#SBATCH --error=output/telme-recent.err

# Activate your virtual environment
source ~/venvs/telme/bin/activate

# Paths
SRC_DATA_DIR="/home/s2751435/Work/msc/TelME/dataset"
SCRATCH_DATA_DIR="/disk/scratch/$USER/TelME_dataset"



# Copy to scratch
echo "Copying MELD dataset to scratch..."
mkdir -p $SCRATCH_DATA_DIR
cp -r $SRC_DATA_DIR/* $SCRATCH_DATA_DIR/
export SCRATCH_DATA_DIR=$SCRATCH_DATA_DIR

echo "Final dataset structure in $SCRATCH_DATA_DIR:"
tree -d "$SCRATCH_DATA_DIR"

# Navigate to your project directory
cd /home/s2751435/Work/msc/TelME




# Run the scripts
python MELD/teacher.py
python MELD/student.py
python MELD/fusion.py
