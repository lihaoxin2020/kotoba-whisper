# export PREPROCESSING_ONLY=1
accelerate launch --num_processes=8 run_pseudo_labelling.py \
    --model_name_or_path "openai/whisper-large-v3" \
    --dataset_name "kotoba-speech/reazonspeech-all-v2_stage2_synth" \
    --dataset_config_name "subset_0" \
    --text_column_name "text_en_gpt3.5" \
    --id_column_name "key" \
    --per_device_eval_batch_size 32 \
    --dataloader_num_workers 32 \
    --preprocessing_num_workers 1 \
    --logging_steps 100 \
    --max_label_length 128 \
    --language "ja" \
    --return_timestamps \
    --generation_num_beams 1 \
    --overwrite_output_dir \
    --output_dir "output-pseudolabel" \
    --hub_model_id "{your-hf-org}/{your-dataset-name}"
