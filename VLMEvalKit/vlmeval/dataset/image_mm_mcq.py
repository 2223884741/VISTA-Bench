import re
import string
import pandas as pd

from .image_mcq import ImageMCQDataset

_MULTI_SPLIT_RE = re.compile(r"\s*[;,，；]\s*")


class MMImageMCQDataset(ImageMCQDataset):
    TYPE = "MM_MCQ"

    DATASET_URL = {
        'VISTA-Bench-VT':'',
    }

    DATASET_MD5 = {
        'VISTA-Bench-VT':None,
    }

    import re
    import pandas as pd

    _MULTI_SPLIT_RE = re.compile(r"\s*[;,，；]\s*")

    def build_prompt(self, line):
        if isinstance(line, int):
            line = self.data.iloc[line]

        # ---------- 工具函数 ----------
        def _get(col, default=None):
            return line[col] if (col in line) else default

        def _nonempty(v) -> bool:
            """安全地判断“是否有内容”。支持标量/字符串/list/tuple。"""
            if v is None:
                return False
            # list/tuple：只要有一个非空且不是 nan/none/null 的子项就算非空
            if isinstance(v, (list, tuple)):
                for x in v:
                    sx = str(x).strip()
                    if sx and sx.lower() not in {"nan", "none", "null", "[]"}:
                        return True
                return False
            # 其他：优先用字符串判空；再尽量用 pd.isna 判定 NaN
            s = str(v).strip()
            if s == "" or s.lower() in {"nan", "none", "null"}:
                return False
            try:
                return not pd.isna(v)
            except Exception:
                return True  # 无法判定 isna 时按有值处理

        def _to_list(v):
            """把字段安全转成 list[str]：支持 list/tuple、'[a, b]'、'a;b'、'a，b' 等。"""
            if v is None:
                return []
            if isinstance(v, (list, tuple)):
                return [str(x).strip() for x in v if str(x).strip()]
            s = str(v).strip()
            if s.startswith("[") and s.endswith("]"):
                s = s[1:-1].strip()
            if not s:
                return []
            parts = _MULTI_SPLIT_RE.split(s)
            return [p for p in (x.strip() for x in parts) if p]

        def _add_imgs(msgs, paths):
            if not paths:
                return
            if isinstance(paths, list):
                msgs.extend([dict(type='image', value=p) for p in paths])
            else:
                msgs.append(dict(type='image', value=paths))

        # ---------- 题干图 ----------
        qimg_raw = _get('question_image_path')
        if not _nonempty(qimg_raw):
            qimg_raw = _get('question_image')
        if not _nonempty(qimg_raw):
            raise ValueError(
                "MMImageMCQDataset: missing 'question_image'/'question_image_path' for the current sample.")
        qimg_paths = _to_list(qimg_raw)

        # ---------- 主图 ----------
        if self.meta_only:
            main_raw = _get('image_path', _get('image'))
            main_paths = _to_list(main_raw)
        else:
            main_paths = self.dump_image(line)

        # ---------- 可选 hint，且不再拼接 A/B/C/D 文本 ----------
        hint = _get('hint')
        prompt_lines = []
        if _nonempty(hint):
            prompt_lines.append(f"Hint: {str(hint).strip()}")
        prompt_lines.append(
            "Read the question and options shown in the image(s). "
            "Answer with only the single letter of the correct option (e.g., A, B, C, D)."
        )
        
        #short prompt
        #prompt_lines.append(
        #    "Read the question and options, then answer with only the single letter (e.g., A, B, C, D)."
        #)

        #long prompt
        #prompt_lines.append(
        #    "Please carefully read the question and options presented in the image(s). "
        #    "Ensure you understand the meaning of each option. "
        #    "Based on the question, choose the most appropriate answer and respond with only the letter of the correct option (e.g., A, B, C, or D). "
        #    "Do not include any additional text or explanations in your response."
        #)
        
        #understanding prompt
        #prompt_lines.append(
        #    "Please take a moment to carefully read the question and all available options shown in the image(s). "
        #    "Ensure you fully understand the context and meaning behind each option. "
        #    "After considering the question and options, choose the most appropriate answer. "
        #    "Respond only with the letter corresponding to the correct option (e.g., A, B, C, or D). "
        #    "Do not include any explanations or comments in your response."
        #)
        
        #CoT prompt
        #prompt_lines.append(
        #    "Analyze the image to identify the question text and the available options. "
        #    "Think step-by-step to deduce the correct answer based on the visual information provided. "
        #    "However, **discard your reasoning process** in the final output. "
        #    "Your final response must consist of **nothing but** the single letter of the correct option (e.g., A). "
        #    "Do not explain why, and do not use punctuation."
        #)
        prompt = "\n".join(prompt_lines).strip()

        # ---------- 组织消息：题干图 -> 主图 -> 文本 ----------
        msgs = []
        _add_imgs(msgs, main_paths)
        _add_imgs(msgs, qimg_paths)
        msgs.append(dict(type='text', value=prompt))

        return msgs