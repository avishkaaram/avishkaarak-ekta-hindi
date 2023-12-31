# Avishkaarak-ekta for QA 

This is the [avishkaarak-ekta-hindi](https://huggingface.co/AVISHKAARAM/avishkaarak-ekta-hindi) model, fine-tuned using the [SQuAD2.0](https://huggingface.co/datasets/squad_v2) dataset. It's been trained on question-answer pairs, including unanswerable questions, for the task of Question Answering. 


## Overview
**Language model:** avishkaarak-ekta-hindi  
**Language:** English, Hindi(Upcoming)  
**Downstream-task:** Extractive QA  
**Training data:** SQuAD 2.0  
**Eval data:** SQuAD 2.0  
**Code:**  See [an example QA pipeline on Haystack](https://haystack.deepset.ai/tutorials/first-qa-system)  
**Infrastructure**: 4x Tesla v100

## Hyperparameters

```
batch_size = 4
n_epochs = 50
base_LM_model = "roberta-base"
max_seq_len = 512
learning_rate = 9e-5
lr_schedule = LinearWarmup
warmup_proportion = 0.2
doc_stride=128
max_query_length=64
``` 

## Usage

### In Haystack
Haystack is an NLP framework by deepset. You can use this model in a Haystack pipeline to do question answering at scale (over many documents). To load the model in [Haystack](https://github.com/deepset-ai/haystack/):
```python
reader = FARMReader(model_name_or_path="AVISHKAARAM/avishkaarak-ekta-hindi")
# or 
reader = TransformersReader(model_name_or_path="AVISHKAARAM/avishkaarak-ekta-hindi",tokenizer="AVISHKAARAM/avishkaarak-ekta-hindi")
```
For a complete example of ``AVISHKAARAM/avishkaarak-ekta-hindi`` being used for  Question Answering, check out the [Tutorials in Haystack Documentation](https://haystack.deepset.ai/tutorials/first-qa-system)

### In Transformers
```python
from transformers import AutoModelForQuestionAnswering, AutoTokenizer, pipeline

model_name = "AVISHKAARAM/avishkaarak-ekta-hindi"

# a) Get predictions
nlp = pipeline('question-answering', model=model_name, tokenizer=model_name)
QA_input = {
    'question': 'What is Bhagavadgita?',
    'context': 'Bhagavadgita, (Sanskrit: “Song of God”) an episode recorded in the great Sanskrit poem of the Hindus, the Mahabharata. It occupies chapters 23 to 40 of Book VI of the Mahabharata and is composed in the form of a dialogue between Prince Arjuna and Krishna, an avatar (incarnation) of the god Vishnu. Composed perhaps in the 1st or 2nd century CE, it is commonly known as the Gita.'
}
res = nlp(QA_input)

# b) Load model & tokenizer
model = AutoModelForQuestionAnswering.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)
```

## Performance
Evaluated on the SQuAD 2.0 dev set with the [official eval script](https://worksheets.codalab.org/rest/bundles/0x6b567e1cf2e041ec80d7098f031c5c9e/contents/blob/).

```
"exact": 79.87029394424324,
"f1": 82.91251169582613,

"total": 11873,
"HasAns_exact": 77.93522267206478,
"HasAns_f1": 84.02838248389763,
"HasAns_total": 5928,
"NoAns_exact": 81.79983179142137,
"NoAns_f1": 81.79983179142137,
"NoAns_total": 5945
```

## Authors
**Shashwat Bindal:** avishkaaram.models@gmail.com

**Sanoj:** avishkaaram.models@gmail.com
