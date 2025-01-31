## **Welcome**  

Hello!

Welcome to my take home assessment. In this repo, there are two ways you can test my code. You'll find a .ipynb jupyter notebook that you can use to initialize each model class and generate embeddings. The answers to discussion questions are also in this notebook, and only available in this notebook. You can also generate embeddings using src/run.py. Instructions for how to use run.py are below. For either path you chose, I'd recommend creating a python venv using src/requirements.txt. Thanks!

## **Usage**  

You can run src/run.py from the commandline like this: 

```sh
python3 src/run.py <model_name> <sentence> [--task <task>]
```

### **Arguments**
- **`model_name`** (required): As part of this assessment I built two model classes you can use to generate embeddings. One class contains just a transformer backbone, and the second contains the backbone extended to support an MTL forward pass. You can call either one using the following options:
  - `"backbone"`  
  - `"mtl"`

- **`sentence`** (required): The input sentence for which embeddings will be generated. Must be enclosed in quotes if it contains spaces.  

- **`--task`** (optional, only used for `"mtl"` model): Options:
  - `"ner"` (Named Entity Recognition)  
  - `"classification"` (Sentence Classification)  
  - If `model_name` is `"backbone"`, this argument will be **ignored** even if you pass it. It will have no effect.  

---

## **Examples**  

### **1. Using the "mtl" model with a task**  
```
python3 src/run.py "mtl" "The cat is running around outside" --task "ner"
```

### **2. Using the "backbone" model**  
```
python3 src/run.py "backbone" "The cat is running around outside"
```

