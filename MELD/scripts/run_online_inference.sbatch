#!/bin/bash
#SBATCH --job-name=telme-online
#SBATCH --partition=PGR-Standard
#SBATCH --gres=gpu:a40:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=08:00:00
#SBATCH --output=output/telme-online.log
#SBATCH --error=output/telme-online.log

# Activate your virtual environment
source ~/venvs/telme/bin/activate

# Ensure Python output is unbuffered
export PYTHONUNBUFFERED=1

# IMPORTANT: Make sure you have run MELD/download.py on a machine with internet access before submitting this job.
# The Hugging Face cache (usually ~/.cache/huggingface) must be available on the compute node.

# Paths
SRC_DATA_DIR="/home/s2751435/Work/msc/TelME/dataset"
SCRATCH_DATA_DIR="/disk/scratch/$USER/TelME_dataset"
MODEL_DIR="/home/s2751435/Work/msc/TelME/MELD/save_model"
MODEL_ARCHIVE="/home/s2751435/Work/msc/TelME/model.tar.gz"

# Check and extract model files if needed
echo "Checking model files..."
if [ ! -f "$MODEL_DIR/teacher.bin" ] || \
   [ ! -f "$MODEL_DIR/student_audio/total_student.bin" ] || \
   [ ! -f "$MODEL_DIR/student_video/total_student.bin" ] || \
   [ ! -f "$MODEL_DIR/total_fusion.bin" ]; then
    echo "Some model files are missing, extracting from archive..."
    mkdir -p $MODEL_DIR/student_audio
    mkdir -p $MODEL_DIR/student_video
    if [ -f "$MODEL_ARCHIVE" ]; then
        echo "Extracting model files from $MODEL_ARCHIVE..."
        tar -xzf $MODEL_ARCHIVE -C $MODEL_DIR
        if [ -f "$MODEL_DIR/teacher.bin" ] && \
           [ -f "$MODEL_DIR/student_audio/total_student.bin" ] && \
           [ -f "$MODEL_DIR/student_video/total_student.bin" ] && \
           [ -f "$MODEL_DIR/total_fusion.bin" ]; then
            echo "Model files extracted successfully"
        else
            echo "Error: Model files not found in archive or extraction failed"
            exit 1
        fi
    else
        echo "Error: Model archive not found at $MODEL_ARCHIVE"
        exit 1
    fi
else
    echo "All model files exist"
fi

# Check if test files exist in scratch
echo "Checking if test dataset exists in scratch..."
if [ -d "$SCRATCH_DATA_DIR/MELD.Raw" ]; then
    echo "MELD.Raw directory exists in scratch"
    if [ -f "$SCRATCH_DATA_DIR/MELD.Raw/test_meld_emo.csv" ]; then
        echo "Test dataset file exists in scratch"
    else
        echo "Test dataset file missing in scratch, copying from source..."
        mkdir -p $SCRATCH_DATA_DIR/MELD.Raw
        cp $SRC_DATA_DIR/MELD.Raw/test_meld_emo.csv $SCRATCH_DATA_DIR/MELD.Raw/
    fi
else
    echo "MELD.Raw directory not found in scratch, creating and copying test file..."
    mkdir -p $SCRATCH_DATA_DIR/MELD.Raw
    cp $SRC_DATA_DIR/MELD.Raw/test_meld_emo.csv $SCRATCH_DATA_DIR/MELD.Raw/
fi

# Copy the test split directory if not present
if [ ! -d "$SCRATCH_DATA_DIR/MELD.Raw/output_repeated_splits_test" ]; then
    echo "output_repeated_splits_test directory missing in scratch, copying from source..."
    cp -r $SRC_DATA_DIR/MELD.Raw/output_repeated_splits_test $SCRATCH_DATA_DIR/MELD.Raw/
else
    echo "output_repeated_splits_test directory already exists in scratch."
fi

export SCRATCH_DATA_DIR=$SCRATCH_DATA_DIR

echo "Final dataset structure in $SCRATCH_DATA_DIR:"
tree -d "$SCRATCH_DATA_DIR"

# Navigate to your project directory
cd /home/s2751435/Work/msc/TelME

echo "About to run python MELD/online_inference.py"
ls -l MELD/online_inference.py

# Run the online inference script
echo "Starting online inference..."
python -u MELD/online_inference.py

echo "Online inference completed!" 