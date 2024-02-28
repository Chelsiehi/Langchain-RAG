
## 依赖

```
pip install langchain pypdf rapidocr-onnxruntime modelscope transformers faiss-cpu tiktoken -i https://mirrors.aliyun.com/pypi/simple/
```

## 用法

1、运行indexer.py，解析pdf生成向量库

```
python indexer.py
```

2、运行rag.py，开始体验RAG增强检索

```
python rag.py
```
