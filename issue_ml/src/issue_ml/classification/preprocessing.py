import re

# 清洗版本号：每次修改清洗规则时更新，training.py 和 prediction.py 均依赖此常量。
PREPROCESSING_VERSION = 'v1'

def clean(title, body):
    """清洗 Issue 标题和正文，返回供模型使用的合并文本。

    删除：Markdown 标题行、HTML 注释、URL、"No response" 占位、类别前缀标签。
    调用方应保证 title 和 body 均为 str（允许空字符串，不接受 None）。
    """
    title = str(title) if title is not None else ''
    body  = str(body)  if body  is not None else ''
    title = re.sub(r'^\s*(?:\[(?:bug|feature(?: request)?|enhancement)]\s*|(?:bug(?: report)?|feature request|enhancement)\s*:\s*)', '', title, flags=re.I)
    body = re.sub(r'<!--.*?-->', ' ', body, flags=re.S)
    body = re.sub(r'^\s*#{1,6}\s+.*$', ' ', body, flags=re.M)
    body = re.sub(r'https?://\S+', ' URL ', body)
    body = re.sub(r'^\s*_?No response_?\s*$', ' ', body, flags=re.M|re.I)
    return re.sub(r'\s+', ' ', title + ' ' + body).strip()

def normalise(text):
    """标准化文本用于去重比较（小写、折叠空白）。

    原名 key()，重命名以避免与内置词 key 和循环变量混淆。
    保留 key 作为别名以兼容现有调用方。
    """
    return re.sub(r'\s+', ' ', text).strip().lower()

# 向后兼容别名：splitting.py / data.py 仍可 from .preprocessing import key
key = normalise
