import torch
from PIL import Image
import os.path as osp
import warnings
import string
import pandas as pd

from ...smp import *
from ...dataset import DATASET_TYPE, DATASET_MODALITY, build_dataset, infer_dataset_basename
from ..base import BaseModel

from transformers import LlavaForConditionalGeneration, AutoProcessor


class LLaVA(BaseModel):

    INSTALL_REQ = True
    INTERLEAVE = True

    def __init__(self, model_path="llava-hf/llava-1.5-7b-hf", **kwargs):
        """
        使用 transformers 官方实现，不再依赖 llava 的 builder/mm_utils。
        """
        super().__init__()

        # --- 加载官方模型 ---
        self.processor = AutoProcessor.from_pretrained(model_path)
        self.model = LlavaForConditionalGeneration.from_pretrained(
            model_path,
            torch_dtype=torch.float16,
            device_map="auto"
        )

        # 系统提示
        self.system_prompt = (
            "A chat between a curious human and an artificial intelligence assistant. "
            "The assistant gives helpful, detailed, and polite answers to the human's questions. "
        )
        self.stop_str = "</s>"

        # 生成参数
        kwargs_default = dict(
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            top_p=None,
            num_beams=1,
            use_cache=True,
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(f"Generation config: {self.kwargs}")

    # ==========================================================
    # 数据集 prompt 构造逻辑
    # ==========================================================
    def use_custom_prompt(self, dataset):
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        """
        输入 line（一个 df 行），返回：
        [
            {"type":"image","value":"/xxx.png"},
            {"type":"text","value":"Question ..."}
        ]
        """
        assert self.use_custom_prompt(dataset)

        tgt_path = self.dump_image(line, dataset)

        # question + hint
        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        # 选项
        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, val in options.items():
            question += f"\n{key}. {val}"

        # 提示语
        if len(options):
            question += (
                "\n请直接回答选项字母。" if cn_string(question)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            question += (
                "\n请直接回答问题。" if cn_string(question)
                else "\nAnswer the question directly."
            )

        # 构造 message
        msg = [dict(type="image", value=s) for s in tgt_path]
        msg.append(dict(type="text", value=question))
        return msg

    def concat_tilist(self, message):
        """
        把 message 列表拆分为文本和图片列表
        message = [
            {"type": "text", "value": "..."},
            {"type": "image", "value": "..."}
        ]
        返回 text_str, images_list
        """
        text, images = "", []
        for item in message:
            if item["type"] == "text":
                text += item["value"]
            elif item["type"] == "image":
                text += " <image> "
                images.append(item["value"])
        return text, images

    # ==========================================================
    # 官方 transformers 版本的 message 转换
    # ==========================================================
    def _convert_to_transformers_messages(self, message_list):
        """
        将你原始格式：
        [
          {"type":"image","value":"/xxx.png"},
          {"type":"text","value":"hi"}
        ]
        转换成 transformers 格式：
        [
          {
            "role":"user",
            "content":[
                {"type":"image","image":PIL},
                {"type":"text","text":"hi"}
            ]
          }
        ]
        """

        content_list = []

        for item in message_list:
            if item["type"] == "image":
                img = Image.open(item["value"]).convert("RGB")
                content_list.append({"type": "image", "image": img})
            else:
                content_list.append({"type": "text", "text": item["value"]})

        return [{
            "role": "user",
            "content": content_list
        }]

    def _convert_chat_history(self, history):
        """
        history: [
          {"role":"user","content":[...]},
          {"role":"assistant","content":[...]},
          ...
        ]
        转换成 transformers 聊天格式。
        """
        out = []
        for utter in history:
            new_content = []
            for item in utter["content"]:
                if item["type"] == "image":
                    img = Image.open(item["value"]).convert("RGB")
                    new_content.append({"type": "image", "image": img})
                else:
                    new_content.append({"type": "text", "text": item["value"]})
            out.append({
                "role": utter["role"],
                "content": new_content
            })
        return out

    # ==========================================================
    # generate_inner：单轮问答
    # ==========================================================
    def generate_inner(self, message, dataset=None):
        """
        使用官方 transformers LlavaForConditionalGeneration 生成答案
        支持 text + image 混合输入，只返回 MCQ 的选项字母
        """
        import re

        _, images = self.concat_tilist(message)
        images = [Image.open(s).convert("RGB") for s in images] if images else None

        transformers_msg = self._convert_to_transformers_messages(message)

        prompt = self.processor.apply_chat_template(
            transformers_msg,
            add_generation_prompt=True,
            tokenize=False
        )

        inputs = self.processor(
            text=prompt,
            images=images,
            return_tensors="pt",
            padding=True
        )

        inputs = {k: v.to("cuda") for k, v in inputs.items()}

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                **self.kwargs
            )

        input_token_len = inputs["input_ids"].shape[1]
        generated_ids = output_ids[:, input_token_len:]

        output_text = self.processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=False
        )[0]

        return output_text

    # ==========================================================
    # chat_inner：多轮对话
    # ==========================================================
    def chat_inner(self, message, dataset=None):
        transformers_msgs = self._convert_chat_history(message)

        inputs = self.processor.apply_chat_template(
            transformers_msgs,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt"
        )

        for k in inputs:
            inputs[k] = inputs[k].to("cuda", torch.float16)

        output = self.model.generate(
            **inputs,
            **self.kwargs
        )

        return self.processor.decode(output[0], skip_special_tokens=True)


class LLaVA_Next(BaseModel):

    INSTALL_REQ = False
    INTERLEAVE = True

    def __init__(self, model_path="llava-hf/llava-v1.6-vicuna-7b-hf", **kwargs):
        import transformers
        from transformers import (
            LlavaNextProcessor,
            LlavaNextForConditionalGeneration,
            AutoProcessor,
            LlavaForConditionalGeneration,
        )

        self.model_path = model_path
        if "34b" in model_path.lower():
            self.processor = LlavaNextProcessor.from_pretrained(
                self.model_path, use_fast=False
            )
        elif "interleave" in model_path.lower():
            self.processor = AutoProcessor.from_pretrained(self.model_path)
        else:
            self.processor = LlavaNextProcessor.from_pretrained(self.model_path)
        flash_attn_flag = False
        try:
            import flash_attn

            flash_attn_flag = True
        except ImportError:
            pass

        if flash_attn_flag:
            if "interleave" in model_path.lower():
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=True,
                )
            else:
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path,
                    torch_dtype=torch.float16,
                    low_cpu_mem_usage=True,
                    use_flash_attention_2=True,
                )
        else:
            if "interleave" in model_path.lower():
                model = LlavaForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
                )
            else:
                model = LlavaNextForConditionalGeneration.from_pretrained(
                    self.model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
                )

        model = model.eval()
        self.model = model.cuda()
        kwargs_default = dict(
            do_sample=False, temperature=0, max_new_tokens=2048, top_p=None, num_beams=1
        )
        kwargs_default.update(kwargs)
        self.kwargs = kwargs_default
        warnings.warn(
            f"Following kwargs received: {self.kwargs}, will use as generation config. "
        )

    def apply_prompt_template(self, prompt):
        model_path = self.model_path.lower()
        if "mistral" in model_path:
            template = "[INST] PLACEHOLDER [/INST]"
        elif "vicuna" in model_path:
            template = (
                "A chat between a curious human and an artificial intelligence assistant. "
                "The assistant gives helpful, detailed, and polite answers to the human's questions. "
                "USER: PLACEHOLDER ASSISTANT:"
            )
        elif "34b" in model_path:
            template = (
                "<|im_start|>system\nAnswer the questions.<|im_end|><|im_start|>user\nPLACEHOLDER<|im_end|>"
                "<|im_start|>assistant\n"
            )
        else:
            raise NotImplementedError(
                f"Prompt template for {model_path} not implemented."
            )

        prompt = template.replace("PLACEHOLDER", f"<image>\n{prompt}")
        return prompt

    def output_process(self, answer):
        if "<s>" in answer:
            answer = answer.replace("<s>", "").strip()
        if "[/INST]" in answer:
            answer = answer.split("[/INST]")[1].strip()
        elif "ASSISTANT:" in answer:
            answer = answer.split("ASSISTANT:")[1].strip()
        elif "assistant\n" in answer:
            answer = answer.split("assistant\n")[1].strip()
        elif "<|end_header_id|>\n\n" in answer:
            answer = answer.split("<|end_header_id|>\n\n")[2].strip()

        if "</s>" in answer:
            answer = answer.split("</s>")[0].strip()
        elif "<|im_end|>" in answer:
            answer = answer.split("<|im_end|>")[0].strip()
        elif "<|eot_id|>" in answer:
            answer = answer.split("<|eot_id|>")[0].strip()
        return answer

    def use_custom_prompt(self, dataset):
        assert dataset is not None
        if DATASET_TYPE(dataset) == "MCQ":
            return True
        return False

    def build_prompt(self, line, dataset=None):
        assert self.use_custom_prompt(dataset)
        assert dataset is None or isinstance(dataset, str)
        tgt_path = self.dump_image(line, dataset)

        question = line["question"]
        hint = line["hint"] if ("hint" in line and not pd.isna(line["hint"])) else None
        if hint is not None:
            question = hint + "\n" + question

        options = {
            cand: line[cand]
            for cand in string.ascii_uppercase
            if cand in line and not pd.isna(line[cand])
        }
        for key, item in options.items():
            question += f"\n{key}. {item}"
        prompt = question

        if len(options):
            prompt += (
                "\n请直接回答选项字母。"
                if cn_string(prompt)
                else "\nAnswer with the option's letter from the given choices directly."
            )
        else:
            prompt += (
                "\n请直接回答问题。"
                if cn_string(prompt)
                else "\nAnswer the question directly."
            )
        message = [dict(type="image", value=s) for s in tgt_path]
        message.append(dict(type="text", value=prompt))
        return message

    def generate_inner(self, message, dataset=None):
        content, images = [], []
        for msg in message:
            if msg["type"] == "text":
                content.append({"type": msg["type"], "text": msg["value"]})
            else:
                content.append({"type": "image"})
                images.append(Image.open(msg["value"]).convert("RGB"))
        conversation = [
            {
                "role": "user",
                "content": content,
            }
        ]
        prompt = self.processor.apply_chat_template(
            conversation, add_generation_prompt=True
        )
        inputs = self.processor(prompt, images, return_tensors="pt").to(
            "cuda", torch.float16
        )
        output = self.model.generate(**inputs, **self.kwargs)
        answer = self.processor.decode(output[0], skip_special_token=True)
        answer = self.output_process(answer)
        answer = answer.replace('<unk>', '')
        return answer


class LLaVA_Next2(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True

    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="lmms-lab/llama3-llava-next-8b", **kwargs):
        assert model_path is not None
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import (
                get_model_name_from_path,
                tokenizer_image_token,
                KeywordsStoppingCriteria,
            )
        except Exception as err:
            logging.critical(
                "Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`"
            )
            raise err

        model_name = get_model_name_from_path(model_path)
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path, None, model_name, device_map=None
        )
        model.cuda().eval()
        model.tie_weights()

        if "llama3" in model_path.lower():
            conv_mode = "llava_llama_3"
        elif "qwen" in model_path.lower():
            conv_mode = "qwen_1_5"
        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.KeywordStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle

    def generate_inner(self, message, dataset=None):
        content, images = "", []
        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                images.append(Image.open(msg["value"]).convert("RGB"))
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        preprocess = self.image_processor.preprocess
        image_tokenizer = self.tokenizer_image_token
        image_tensor = [
            preprocess(f, return_tensors="pt")["pixel_values"][0].half().cuda()
            for f in images
        ]
        image_tensor = torch.stack(image_tensor)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = image_tokenizer(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs


class LLaVA_OneVision(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="lmms-lab/llava-onevision-qwen2-7b-si", **kwargs):
        assert model_path is not None
        try:
            from llava.model.builder import load_pretrained_model
            from llava.conversation import conv_templates, SeparatorStyle
            from llava.mm_utils import (
                get_model_name_from_path,
                process_images,
                tokenizer_image_token,
                KeywordsStoppingCriteria,
            )  # noqa: E501
        except Exception as err:
            logging.critical(
                "Please `pip install git+https://github.com/LLaVA-VL/LLaVA-NeXT.git`"
            )
            raise err

        video_kwargs_default = dict(
            overwrite=True, mm_spatial_pool_mode="average", force_sample=True
        )
        video_kwargs_default.update(kwargs)
        self.video_kwargs = video_kwargs_default

        overwrite_config = None
        if "video" in model_path.lower():
            if self.video_kwargs["overwrite"]:
                overwrite_config = {}
                overwrite_config["mm_spatial_pool_mode"] = self.video_kwargs[
                    "mm_spatial_pool_mode"
                ]

        model_name = get_model_name_from_path(model_path)
        import warnings
        # filter warning align with official code
        warnings.filterwarnings("ignore")
        tokenizer, model, image_processor, _ = load_pretrained_model(
            model_path,
            None,
            model_name,
            device_map="auto",
            overwrite_config=overwrite_config,
        )
        model.eval()
        model.tie_weights()

        if "llava" in model_path.lower():
            conv_mode = "qwen_1_5"
        if 'llava-video' in model_path.lower():
            self.nframe = 64
        else:
            self.nframe = 16
            if "72b" in model_path.lower():
                self.nframe = 32

        if "video" in model_path.lower():
            self.force_sample = self.video_kwargs["force_sample"]
        else:
            self.force_sample = False

        self.conv_template = conv_mode
        self.conv_templates = conv_templates
        self.tokenizer = tokenizer
        self.model = model
        self.image_processor = image_processor
        self.tokenizer_image_token = tokenizer_image_token
        self.process_images = (
            process_images  # Store process_images as a class attribute
        )
        self.KeywordStoppingCriteria = KeywordsStoppingCriteria
        self.SeparatorStyle = SeparatorStyle

    def generate_inner_image(self, message, dataset=None):
        content, images = "", []
        image_sizes = []  # Store image sizes

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            else:
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)  # Store the size of each image
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        # Process images using the class attribute self.process_images
        image_tensor = self.process_images(
            images, self.image_processor, self.model.config
        )
        image_tensor = [
            _image.to(dtype=torch.float16, device="cuda") for _image in image_tensor
        ]

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensor,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def generate_inner_video(self, message, dataset=None):
        content, text_content, visual_content, videos = "", "", "", []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            else:
                videos.append(msg["value"])
                visual_content += self.DEFAULT_IMAGE_TOKEN + "\n"

        if len(videos) > 1:
            raise ValueError(
                "LLaVA-OneVision does not support multiple videos as input."
            )

        video_frames, frame_time, video_time = self.load_video(
            videos[0], self.nframe, 1, self.force_sample
        )

        time_instruciton = (
            f"The video lasts for {video_time:.2f} seconds,"
            f"and {len(video_frames[0])} frames are uniformly sampled from it."
            f"These frames are located at {frame_time}."
            f"Please answer the following questions related to this video.\n"
        )

        if self.force_sample:
            content = visual_content + time_instruciton + text_content
        else:
            content = visual_content + text_content

        image_tensors = []
        frames = (
            self.image_processor.preprocess(video_frames, return_tensors="pt")[
                "pixel_values"
            ]
            .half()
            .cuda()
        )
        image_tensors.append(frames)

        conv = copy.deepcopy(self.conv_templates[self.conv_template])
        conv.append_message(conv.roles[0], content)
        conv.append_message(conv.roles[1], None)
        prompt_question = conv.get_prompt()

        input_ids = self.tokenizer_image_token(
            prompt_question, self.tokenizer, self.IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        input_ids = input_ids.unsqueeze(0).cuda()
        image_sizes = [frame.size for frame in video_frames]
        modalities = ["video"] * len(video_frames)

        stop_str = conv.sep if conv.sep_style != self.SeparatorStyle.TWO else conv.sep2
        keywords = [stop_str]
        stopping_criteria = self.KeywordStoppingCriteria(
            keywords, self.tokenizer, input_ids
        )

        # Pass image sizes along with other parameters
        cont = self.model.generate(
            input_ids,
            images=image_tensors,
            image_sizes=image_sizes,  # Pass the image sizes here
            do_sample=False,
            temperature=0,
            max_new_tokens=2048,
            modalities=modalities,
            stopping_criteria=[stopping_criteria],
        )
        text_outputs = self.tokenizer.batch_decode(cont, skip_special_tokens=True)[0]
        return text_outputs

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        from decord import VideoReader, cpu
        import numpy as np

        if max_frames_num == 0:
            return np.zeros((1, 336, 336, 3))
        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        video_time = total_frame_num / vr.get_avg_fps()
        fps = round(vr.get_avg_fps() / fps)
        frame_idx = [i for i in range(0, len(vr), fps)]
        frame_time = [i / fps for i in frame_idx]
        if len(frame_idx) > max_frames_num or force_sample:
            sample_fps = max_frames_num
            uniform_sampled_frames = np.linspace(
                0, total_frame_num - 1, sample_fps, dtype=int
            )
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / vr.get_avg_fps() for i in frame_idx]
        frame_time = ",".join([f"{i:.2f}s" for i in frame_time])
        spare_frames = vr.get_batch(frame_idx).asnumpy()
        # import pdb;pdb.set_trace()
        return spare_frames, frame_time, video_time

    def generate_inner(self, message, dataset=None):
        if DATASET_MODALITY(dataset) == 'VIDEO' and 'megabench' not in dataset.lower():
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)


class LLaVA_OneVision_HF(BaseModel):
    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"
    IMAGE_TOKEN_INDEX = -200

    def __init__(self, model_path="llava-hf/llava-onevision-qwen2-0.5b-ov-hf", **kwargs):
        from transformers import AutoProcessor, LlavaOnevisionForConditionalGeneration
        assert model_path is not None, "Model path must be provided."
        self.model = LlavaOnevisionForConditionalGeneration.from_pretrained(
            model_path, torch_dtype=torch.float16, low_cpu_mem_usage=True
        ).to('cuda')
        self.processor = AutoProcessor.from_pretrained(model_path)

        self.video_kwargs = kwargs.get("video_kwargs", {})
        self.force_sample = self.video_kwargs.get("force_sample", False)
        self.nframe = kwargs.get("nframe", 8)
        self.fps = 1
        self.model_path = model_path

    def generate_inner_image(self, message, dataset=None):
        content, images = "", []
        images = []
        image_sizes = []

        for msg in message:
            if msg["type"] == "text":
                content += msg["value"]
            elif msg["type"] == "image":
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                image_sizes.append(img.size)
                content += self.DEFAULT_IMAGE_TOKEN + "\n"

        conversation = [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": content},
                ],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)
        #inputs = self.processor(images=images, text=prompt, return_tensors="pt").to('cuda', torch.float16)

        if len(images) == 0:
            inputs = self.processor(
                text=prompt,
                return_tensors="pt"
            ).to("cuda", torch.float16)
        else:
            inputs = self.processor(
                images=images,
                text=prompt,
                return_tensors="pt"
            ).to("cuda", torch.float16)

        output = self.model.generate(**inputs, max_new_tokens=2048)
        return self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def generate_inner_video(self, message, dataset=None):
        content, text_content, visual_content, videos = "", "", "", []

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            elif msg["type"] == "video":
                videos.append(msg["value"])
                visual_content += self.DEFAULT_IMAGE_TOKEN + "\n"

        if len(videos) > 1:
            raise ValueError("LLaVA-OneVision does not support multiple videos as input.")

        video_frames, frame_time, video_time = self.load_video(
            videos[0], self.nframe, fps=1, force_sample=self.force_sample
        )

        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, "
            f"and {len(video_frames)} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}. "
            f"Please answer the following questions related to this video.\n"
        )

        content = visual_content + time_instruction + text_content
        conversation = [
            {
                "role": "user",
                "content": [{"type": "text", "text": content}, {"type": "video"}],
            }
        ]
        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(videos=video_frames, text=prompt, return_tensors="pt").to('cuda', torch.float16)
        output = self.model.generate(**inputs, max_new_tokens=2048)
        return self.processor.decode(output[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)

    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        from decord import VideoReader, cpu
        import numpy as np

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()

        if avg_fps == 0:
            raise ValueError(f"Video '{video_path}' has an average FPS of 0, which is invalid.")
        if fps <= 0:
            raise ValueError("FPS argument must be greater than 0.")

        effective_fps = round(avg_fps / fps)
        frame_idx = list(range(0, total_frame_num, effective_fps))
        frame_time = [i / avg_fps for i in frame_idx]

        if len(frame_idx) > max_frames_num or force_sample:
            uniform_sampled_frames = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int)
            frame_idx = uniform_sampled_frames.tolist()
            frame_time = [i / avg_fps for i in frame_idx]

        frame_time_str = ", ".join([f"{t:.2f}s" for t in frame_time])
        video_frames = vr.get_batch(frame_idx).asnumpy()
        video_time = total_frame_num / avg_fps

        return video_frames, frame_time_str, video_time

    def generate_inner(self, message, dataset=None):
        if DATASET_MODALITY(dataset) == "VIDEO" and 'megabench' not in dataset.lower():
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)

class LLaVA_OneVision1_5_HF(BaseModel):
    """
    VLMEvalKit wrapper for:
      lmms-lab/LLaVA-OneVision-1.5-8B-Instruct

    Style aligns with existing LLaVA_OneVision_HF:
      - build content string with <image> placeholders
      - conversation for image: [{"role":"user","content":[{"type":"text","text": content}]}]
      - conversation for video: [{"role":"user","content":[{"type":"text","text": content},{"type":"video"}]}]
      - processor(images/videos=..., text=prompt, return_tensors="pt")
      - generate then slice off prompt tokens
    """

    INSTALL_REQ = True
    INTERLEAVE = True
    VIDEO_LLM = True
    DEFAULT_IMAGE_TOKEN = "<image>"

    def __init__(
        self,
        model_path="lmms-lab/LLaVA-OneVision-1.5-8B-Instruct",
        **kwargs
    ):
        super().__init__()
        from transformers import AutoModelForCausalLM, AutoProcessor, AutoTokenizer

        assert model_path is not None, "Model path must be provided."
        self.model_path = model_path

        # generation / video settings (align with your HF onevision wrapper)
        self.video_kwargs = kwargs.get("video_kwargs", {})
        self.force_sample = self.video_kwargs.get("force_sample", False)

        self.max_num_frames = int(kwargs.get("max_num_frames", 32))
        self.fps = float(kwargs.get("fps", 1))

        # pixels control (kept for compatibility; processor may ignore depending on model)
        self.min_pixels = int(kwargs.get("min_pixels", 256 * 28 * 28))
        self.max_pixels = int(kwargs.get("max_pixels", 1605632))

        # prompt controls
        self.system_prompt = kwargs.get("system_prompt", "You are a helpful assistant.")
        self.reasoning_prompt = kwargs.get("reasoning_prompt", None)
        if self.reasoning_prompt is not None:
            self.reasoning_prompt = self.reasoning_prompt.replace("\\n", "\n")

        # model / processor / tokenizer (trust_remote_code like lmms-eval)
        attn_impl = kwargs.get("attn_implementation", None)
        model_kwargs = dict(torch_dtype="auto", device_map=kwargs.get("device_map", "auto"), trust_remote_code=True)
        if attn_impl is not None:
            model_kwargs["attn_implementation"] = attn_impl

        self.model = AutoModelForCausalLM.from_pretrained(model_path, **model_kwargs).eval()

        # processor: pass min/max_pixels if supported; ignore if not
        try:
            self.processor = AutoProcessor.from_pretrained(
                model_path,
                trust_remote_code=True,
                min_pixels=self.min_pixels,
                max_pixels=self.max_pixels
            )
        except TypeError:
            self.processor = AutoProcessor.from_pretrained(model_path, trust_remote_code=True)

        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        self.max_new_tokens = int(kwargs.get("max_new_tokens", 2048))

    # -------------------------
    # video loader
    # -------------------------
    def load_video(self, video_path, max_frames_num, fps=1, force_sample=False):
        from decord import VideoReader, cpu
        import numpy as np

        vr = VideoReader(video_path, ctx=cpu(0), num_threads=1)
        total_frame_num = len(vr)
        avg_fps = vr.get_avg_fps()

        if avg_fps == 0:
            raise ValueError(f"Video '{video_path}' has an average FPS of 0, which is invalid.")
        if fps <= 0:
            raise ValueError("FPS argument must be greater than 0.")

        effective_step = max(1, round(avg_fps / fps))
        frame_idx = list(range(0, total_frame_num, effective_step))

        if len(frame_idx) > max_frames_num or force_sample:
            frame_idx = np.linspace(0, total_frame_num - 1, max_frames_num, dtype=int).tolist()
        if (total_frame_num - 1) not in frame_idx:
            frame_idx.append(total_frame_num - 1)

        frame_idx = sorted(set(frame_idx))
        frame_time = [i / avg_fps for i in frame_idx]
        frame_time_str = ", ".join([f"{t:.2f}s" for t in frame_time])

        video_frames = vr.get_batch(frame_idx).asnumpy()  # (T,H,W,C)
        video_time = total_frame_num / avg_fps
        return video_frames, frame_time_str, video_time

    # -------------------------
    # image generation
    # -------------------------
    def generate_inner_image(self, message, dataset=None):
        images = []
        user_content = []

        for msg in message:
            if msg["type"] == "text":
                user_content.append({"type": "text", "text": msg["value"]})
            elif msg["type"] == "image":
                img = Image.open(msg["value"]).convert("RGB")
                images.append(img)
                user_content.append({"type": "image"})

        if self.reasoning_prompt:
            if user_content and user_content[-1]["type"] == "text":
                user_content[-1]["text"] += self.reasoning_prompt

        conversation = [
            {"role": "system", "content": [{"type": "text", "text": self.system_prompt}]},
            {"role": "user", "content": user_content},
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        if len(images) == 0:
            inputs = self.processor(text=prompt, return_tensors="pt", padding=True)
        else:
            inputs = self.processor(images=images, text=prompt, return_tensors="pt", padding=True)

        inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        in_len = inputs["input_ids"].shape[1]
        gen_ids = output[:, in_len:]
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text.strip()

    # -------------------------
    # video generation
    # -------------------------
    def generate_inner_video(self, message, dataset=None):
        text_content = ""
        videos = []
        visual_content = ""

        for msg in message:
            if msg["type"] == "text":
                text_content += msg["value"]
            elif msg["type"] == "video":
                videos.append(msg["value"])
                visual_content += self.DEFAULT_IMAGE_TOKEN + "\n"

        if len(videos) != 1:
            raise ValueError("LLaVA-OneVision-1.5 HF wrapper supports exactly one video input.")

        video_frames, frame_time, video_time = self.load_video(
            videos[0],
            max_frames_num=self.max_num_frames,
            fps=self.fps,
            force_sample=self.force_sample
        )

        time_instruction = (
            f"The video lasts for {video_time:.2f} seconds, "
            f"and {len(video_frames)} frames are uniformly sampled from it. "
            f"These frames are located at {frame_time}. "
            f"Please answer the following questions related to this video.\n"
        )

        content = visual_content + time_instruction + text_content
        if self.reasoning_prompt:
            content = content.strip() + self.reasoning_prompt

        conversation = [
            {"role": "system", "content": self.system_prompt},
            {"role": "user", "content": [{"type": "text", "text": content}, {"type": "video"}]},
        ]

        prompt = self.processor.apply_chat_template(conversation, add_generation_prompt=True)

        inputs = self.processor(videos=video_frames, text=prompt, return_tensors="pt", padding=True)
        inputs = {k: (v.to("cuda") if torch.is_tensor(v) else v) for k, v in inputs.items()}

        with torch.inference_mode():
            output = self.model.generate(**inputs, max_new_tokens=self.max_new_tokens)

        in_len = inputs["input_ids"].shape[1]
        gen_ids = output[:, in_len:]
        text = self.processor.batch_decode(gen_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return text.strip()

    def generate_inner(self, message, dataset=None):
        if DATASET_MODALITY(dataset) == "VIDEO" and "megabench" not in dataset.lower():
            return self.generate_inner_video(message, dataset)
        else:
            return self.generate_inner_image(message, dataset)

