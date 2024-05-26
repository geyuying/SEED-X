# SEED-X

## Usage

### Dependencies
- Python >= 3.8 (Recommend to use [Anaconda](https://www.anaconda.com/download/#linux))
- [PyTorch >=2.0.1](https://pytorch.org/)
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

### Installation
Clone the repo and install dependent packages

  ```bash
  git clone this_project
  cd SEED-X
  pip install -r requirements.txt
  ```

### Model Weights
We release the pretrained De-Tokenizer, the pre-trained foundation model **SEED-X**, the general instruction-tuned model **SEED-X-I**, the editing model **SEED-X-Edit** in [Google Drive](https://drive.google.com/drive/folders/1Zbkrra8V-1fCR-PGFnjSny1ymMG8UCzO?usp=sharing)

Please download the checkpoints and save them under the folder `./pretrained`. For example, `./pretrained/seed_x`.

You also need to download [stable-diffusion-xl-base-1.0](https://huggingface.co/stabilityai/stable-diffusion-xl-base-1.0) and [Qwen-VL-Chat](https://huggingface.co/Qwen/Qwen-VL-Chat), and save them under the folder `./pretrained`. Please use the following script to extract the weights of visual encoder in Qwen-VL-Chat.
```bash
python3 src/tools/reload_qwen_vit.py
```
### Inference
#### Inference with SEED-X De-tokenizer
```bash
# For image reconstruction with ViT image features
python3 src/inference/eval_seed_x_detokenizer.py
# For image reconstruction with ViT image features and conditional image
python3 src/inference/eval_seed_x_detokenizer_with_condition.py
```

#### Inference with pre-trained model SEED-X
```bash
# For image comprehension and detection
python3 src/inference/eval_img2text_seed_x.py
# For image generation
python3 src/inference/eval_text2img_seed_x.py
```

#### Inference with the general instruction-tuned model SEED-X-I
```bash
# For image comprehension and detection
python3 src/inference/eval_img2text_seed_x_i.py
# For image generation
python3 src/inference/eval_text2img_seed_x_i.py
```

#### Inference with the editing model SEED-X-Edit
```bash
# For image editing
python3 src/inference/eval_img2edit_seed_x_edit.py
```

### Instruction Tuning
#### Training
1. Prepare the pretrained models including the pre-trained foundation model **SEED-X** and the visual encoder of Qwen-VL-Chat (See Model Weights).
2. Prepare the instruction tuning data. For example, for "build_llava_jsonl_datapipes" dataloader, each folder stores a number of jsonl files, each jsonl file contains 10K pieces of content, with an example of the content as follows:
```bash
{"image": "coco/train2017/000000033471.jpg", "data": ["What are the colors of the bus in the image?", "The bus in the image is white and red.", "What feature can be seen on the back of the bus?", "The back of the bus features an advertisement.", "Is the bus driving down the street or pulled off to the side?", "The bus is driving down the street, which is crowded with people and other vehicles."]}
```

For "build_caption_datapipes_with_pixels" dataloder, each folder stores a number of .tar files and reads image-text pairs in the form of webdataset.

For "build_single_turn_edit_datapipes" dataloder,  each folder stores a number of jsonl files, each jsonl file contains 10K pieces of content, with an example of the content as follows:
```bash
{"source_image": "source_images/f6f4d0669694df5b.jpg", "target_image": "target_images/f6f4d0669694df5b.jpg", "instruction": "Erase the car that is parked in front of the Roebuck building."}
```
3. Run the following script.

```bash
# For general instruction tuning for multimodal comprehension and generation
sh scripts/train_seed_x_sft_comp_gen.sh
```

```bash
# For training language-guided image editing
sh scripts/train_seed_x_sft_edit.sh
```
#### Inference with your own model
1. Obtain "pytorch_model.bin" with the following script.
```bash
cd train_output/seed_x_sft_comp_gen/checkpoint-xxxx
python3 zero_to_fp32.py . pytorch_model.bin
```
2. Change "pretrained_model_path" in "configs/clm_models/agent_seed_x.yaml" with the new checkpoint. For example,
```bash
pretrained_model_path: train_output/seed_x_sft_comp_gen/checkpoint-4000/pytorch_model.bin
```
3. Change the "llm_cfg_path" and "agent_cfg_path" in the inference script (See below), which will automatically load the trained LoRA weights onto the pretrained model SEED-X.
```bash
llm_cfg_path = 'configs/clm_models/llm_seed_x_lora.yaml'
agent_cfg_path = 'configs/clm_models/agent_seed_x.yaml'
```
4. Run the inference script,
```bash
# For image comprehension
python3 src/inference/eval_img2text_seed_x_i.py
# For image generation
python3 src/inference/eval_text2img_seed_x_i.py
# For image editing
python3 src/inference/eval_img2edit_seed_x_edit.py
```


