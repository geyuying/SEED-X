# SEED-X
<!--## Examples of Visual De-tokenization
![image](demos/tokenizer.jpg?raw=true)
The reconstruction results of our visual de-tokenizer. It can decode realistic images that are semantically aligned with the original images by taking the ViT features as inputs, and further recover fine-grained details by incorporating the conditional images as inputs.
## Ablation Study
### Visual De-tokenizer
![image](demos/ablation_detokenizer.jpg?raw=true)
We utilize a pre-trained ViT as the visual tokenizer and pre-train a visual de-tokenizer to decode realistic images by taking the features of the ViT as inputs.  Specifically, N visual embeddings (after average pooling) from the ViT tokenizer are fed into a learnable module as the inputs of the U-Net of the pre-trained SD-XL. We perform an ablation study on **the number of viual embeddings** and **the learnable parameters** of the SD-XL U-Net, where keys and values within the U-Net are optimized if not specified with "fully fine-tunue". The input images and the reconstructed images from the visual de-tokenizer are shown in the figure below. We can observe that more visual tokens can result in better reconstruction of the original images. For example, the decoded images from 256 visual embeddings can recover the characters' postures of the original images, while decoded images from 32 visual embeddings have already lost the original structure of the scene. We further observe that fully fine-tuning the parameters of the SD-XL U-Net can lead to distortions in image details, such as the woman's feet, compared to only training the keys and values within the U-Net. In SEED-X, we use N = 64 visual embeddings to train the visual de-tokenizer and only optimize the keys and values within the U-Net (See the ablation study below for an explanation of why we do not choose N = 256).

### MLLM for Image Generation
![image](demos/ablation_t2i.jpg?raw=true)
To enable MLLM for image generation, we employ N learnable queries to obtain the output visual representations from the LLM, which are trained to reconstruct N visual embeddings from the ViT tokenizer with a learnable module. We first perform an abation study on **the number of learnable queries**. The images generated by the MLLM based on the input caption are shown in the figure below. We can observe that using 256 learnable queries to reconstruct 256 visual embeddings can lead to distortion in the generated images compared with N = 64. This occurs because regressing more visual features is more challenging for the model, even though 256 visual embeddings from the de-tokenizer can better reconstruct images, as demonstrated in the previous ablation study. We also observe that, compared to learning a **one-layer** cross-attention for reconstructing image features, a **multi-layer** resampler (multi-layer cross-attention) yields less satisfactory performance, which can happen due to the lack of more direct regularizations on the hidden states of the LLM. We further optimize the visual de-tokenizer by using the reconstructed visual embeddings from the MLLM as input instead of ViT features, but the generated images exhibit a more monotonous appearance. It demonstrates the effectiveness of utilizing the ViT Tokenizer as the bridge to decouple the training of visual de-tokenizer and the MLLM for image generation.

## Model Performance
<table class="tg"><thead>
  <tr>
    <th class="tg-0pky"></th>
    <th class="tg-c3ow">MMB</th>
    <th class="tg-c3ow" colspan="4">SEED-Bench-2</th>
    <th class="tg-c3ow" colspan="2">MME</th>
  </tr></thead>
<tbody>
  <tr>
    <td class="tg-0pky"></td>
    <td class="tg-0pky">Single</td>
    <td class="tg-0pky">Single</td>
    <td class="tg-0pky">Multi</td>
    <td class="tg-0pky">Inter-<br>leaved</td>
    <td class="tg-0pky">Gen</td>
    <td class="tg-0pky">Single</td>
    <td class="tg-0pky">Single</td>
  </tr>
  <tr>
    <td class="tg-0pky">SEED-X</td>
    <td class="tg-0pky">65.8</td>
    <td class="tg-0pky">48.2</td>
    <td class="tg-0pky">53.8</td>
    <td class="tg-0pky">24.3</td>
    <td class="tg-0pky">57.8</td>
    <td class="tg-0pky">1250</td>
    <td class="tg-0pky">236</td>
  </tr>
  <tr>
    <td class="tg-0pky">SEED-X-I</td>
    <td class="tg-0pky">77.8</td>
    <td class="tg-0pky">66.8</td>
    <td class="tg-0pky">57.1</td>
    <td class="tg-0pky">40.5</td>
    <td class="tg-0pky">61.6</td>
    <td class="tg-0pky">1520</td>
    <td class="tg-0pky">338</td>
  </tr>
</tbody></table>

## Dataset
## Pre-training
#### Image-Caption
<table style="margin: auto">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LAION-COCO</td>
       <td>600M</td>
      <td>Web-images with synthetic captions by BLIP L/14. 30M images are re-captioned by a MLLM.</td>
    </tr>
        <tr>
      <td>SAM</td>
       <td>11M</td>
      <td>Diverse and high-resolution images, with captions generated by a MLLM.</td>
    </tr>
        <tr>
      <td>LAION-Aesthetics</td>
       <td>3M</td>
      <td>Image-text pairs with predicted aesthetics scores of 6.25 or higher.</td>
    </tr>
        <tr>
      <td>Unsplash</td>
       <td>2M</td>
      <td>Images from contributing global photographers, with captions generated by a MLLM.</td>
    </tr>
        <tr>
      <td>JourneyDB</td>
       <td>4M</td>
      <td> High-resolution Midjourney images, annotated with corresponding text prompt, image caption.</td>
    </tr>
          <tr>
      <td>CapFusion</td>
       <td>120M</td>
      <td>Images from LAION-COCO, with captions integrated from both the web and synthetic captions.</td>
    </tr>
      </tbody>
</table>

#### Grounded Image-Caption
<table style="margin: auto">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>GRIT</td>
       <td>191M</td>
      <td>Image-text pairs with noun phrases in the caption annotated with bounding boxes.</td>
    </tr>
</table>

#### Interleaved Image-Text
<table style="margin: auto">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MMC4</td>
       <td>7M</td>
      <td>A augmentation of the text-only c4 corpus with images interleaved.</td>
    </tr>
          <tr>
      <td>OBELICS</td>
       <td>141M</td>
      <td>An web-scale filtered dataset of interleaved image-text documents comprising web pages extracted from Common Crawl.</td>
    </tr>
        <tr>
      <td>OpenFlamingo</td>
       <td>400K</td>
      <td>A sequence of interleaved text and image alt-texts generated by ChatGPT, with images retrieved from LAION-5B.</td>
    </tr>
      </tbody>
</table>

#### OCR
<table style="margin: auto">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LLaVAR-Pretrain</td>
       <td>400K</td>
      <td>Text-rich images from LAION, with OCR results.</td>
    </tr>
          <tr>
      <td>Slides</td>
       <td>1M</td>
      <td>Images from slides, with OCR results.</td>
    </tr>
</table>


#### Pure Text
<table style="margin: auto">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Wikipedi</td>
       <td>66M</td>
      <td>Cleaned articles of all languages from the Wikipedia dump.</td>
    </tr>
</table>


### Instruction Tuning
#### VQA
<table style="margin: auto">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LLaVAR-sft</td>
       <td>16K</td>
      <td>High-quality instruction-following data by interacting with GPT-4 based on OCR results of text-rich images.</td>
    </tr>
        <tr>
      <td>Text-rich QA</td>
       <td>900K</td>
      <td>Instruction-following data generated by GPT-4V based on text-rich images.</td>
    </tr>
        <tr>
      <td>MIMIC-IT</td>
       <td>150K</td>
      <td>Difference spotting data with general scene difference and subtle difference.</td>
    </tr>
        <tr>
      <td>MathQA</td>
       <td>37K</td>
      <td>Math word problems that are densely annotated with operation programs.</td>
    </tr>
        <tr>
      <td>ChartQA</td>
       <td>33K</td>
      <td>Human-written questions focusing on visual and logical reasoning about charts.</td>
    </tr>
          <tr>
      <td>AI2D</td>
       <td>5K</td>
      <td>Illustrative diagrams for diagram understanding and associated question answering.</td>
    </tr>
      <tr>
      <td>ScienceQA</td>
       <td>21K</td>
      <td>Multiple-choice science questions collected from elementary and high school science curricula.</td>
    </tr>
        <tr>
      <td>KVQA</td>
       <td>183K</td>
      <td>Questions that require multi-entity, multi-relation, and multi-hop reasoning over large Knowledge Graphs.</td>
    </tr>
        <tr>
      <td>DVQA</td>
       <td>3M</td>
      <td>A synthetic question-answering dataset on images of bar-charts.</td>
    </tr>
        <tr>
      <td>Grounded QA</td>
       <td>680K</td>
      <td>Questions constructed from region captions with bounding boxes.</td>
    </tr>
        <tr>
      <td>Referencing QA</td>
       <td>630K</td>
      <td>Questions constructed from images with regions marked.</td>
    </tr>
      </tbody>
</table>


#### Conversation
<table style="margin: auto">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>LLaVA-150k</td>
       <td>150K</td>
      <td>A set of GPT-generated multimodal instruction-following data.</td>
    </tr>
        <tr>
      <td>ShareGPT</td>
       <td>1.2M</td>
      <td>100K high-quality captions collected from GPT4-V and 1.2 million data captioned by a superb caption model.</td>
    </tr>
        <tr>
      <td>LVIS-Instruct4V</td>
       <td>220K</td>
      <td>A fine-grained visual instruction dataset produced by GPT-4V with images from LVIS.</td>
    </tr>
        <tr>
      <td>VLIT</td>
       <td>770K</td>
      <td>A multi-round question answering dataset about a given image from COCO.</td>
    </tr>
        <tr>
      <td>Vision-Flan</td>
       <td>190K</td>
      <td>A visual instruction tuning dataset that consists of 200+ diverse vision-language tasks derived from 101 open-source computer vision datasets.</td>
    </tr>
          <tr>
      <td>ALLaVA-4V</td>
       <td>1.4M</td>
      <td>Images with fine-grained captions, complex instructions and detailed answers generated by GPT-4V.</td>
    </tr>
      </tbody>
</table>

#### Image Generation
<table style="margin: auto">
  <thead>
    <tr>
      <td>LAION-COCO</td>
       <td>600M</td>
      <td>Web-images with synthetic captions by BLIP L/14. 30M images are re-captioned by a MLLM.</td>
    </tr>
        <tr>
      <td>SAM</td>
       <td>11M</td>
      <td>Diverse and high-resolution images, with captions generated by a MLLM.</td>
    </tr>
        <tr>
      <td>LAION-Aesthetics</td>
       <td>3M</td>
      <td>Image-text pairs with predicted aesthetics scores of 6.25 or higher.</td>
    </tr>
        <tr>
      <td>Unsplash</td>
       <td>2M</td>
      <td>Images from contributing global photographers, with captions generated by a MLLM.</td>
    </tr>
        <tr>
      <td>JourneyDB</td>
       <td>4M</td>
      <td> High-resolution Midjourney images, annotated with corresponding text prompt, image caption.</td>
    </tr>
      </tbody>
</table>

#### Image Editing
<table style="margin: auto">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>Instructpix2pix</td>
       <td>313K</td>
      <td>Image editing examples with language instructions generated by GPT-3 and Stable Diffusion.</td>
    </tr>
        <tr>
      <td>MagicBrush</td>
       <td>10K</td>
      <td>Manually annotated triplets (source image, instruction, target image) with multi rounds.</td>
    </tr>
        <tr>
      <td>Openimages-editing</td>
       <td>1.4M</td>
      <td>Image editing examples with language instructions constructed by an automatic pipeline, with images from Openimages.</td>
    </tr>
        <tr>
      <td>Unsplash-editing</td>
       <td>1.3M</td>
      <td>Image editing examples with language instructions constructed by an automatic pipeline, with images from Unsplash.</td>
    </tr>
      </tbody>
</table>


#### Slides Generation
<table style="margin: auto">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>SlidesGen</td>
       <td>10K</td>
      <td>Slides with layout descriptions and captions generated by a slide2json tool and a MLLM.</td>
    </tr>
</table>

#### Story Telling
<table style="margin: auto">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VIST</td>
       <td>20K</td>
      <td>Inique photos in sequences, aligned to both descriptive (caption) and story language.</td>
    </tr>
</table>

#### Virtual Try-on
<table style="margin: auto">
  <thead>
    <tr>
      <th>Dataset</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>VITON-HD</td>
       <td>13K</td>
      <td>A dataset for high-resolution virtual try-on, with frontal-view woman and top clothing image pairs.</td>
    </tr>
</table>


## Benchmark
<table style="margin: auto">
  <thead>
    <tr>
      <th>Benchmark</th>
      <th>Number</th>
      <th>Description</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <td>MMBench</td>
       <td>3K</td>
      <td>Multiple-choice questions for evaluating both perception and reasoning covering 20 fine-grained ability dimensions.</td>
    </tr>
        <tr>
      <td>SEED-Bench-2</td>
       <td>24K</td>
      <td>Multiple-choice questions with accurate human annotations, which spans 27 dimensions, including the evaluation of both text and image generation.</td>
    </tr>
        <tr>
      <td>MME</td>
       <td>2K</td>
      <td>True/False questions for evaluating both perception and cognition, including a total of 14 subtasks.</td>
    </tr>
      </tbody>
</table>
-->

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


