from modelscope.hub.snapshot_download import snapshot_download
from transformers import AutoModelForCausalLM, AutoTokenizer

cache_dir = './sunsimiao/'

model_dir = snapshot_download('thomas/Sunsimiao-InternLM-01M',
                              cache_dir=cache_dir,
                              revision='v1.0.0')
model_dir_sft = snapshot_download('thomas/Sunsimiao-InternLM-01M',
                                  cache_dir=cache_dir,
                                  revision='v1.0.0')

tokenizer = AutoTokenizer.from_pretrained(cache_dir +
                                          'thomas/Sunsimiao-InternLM-01M',
                                          trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(cache_dir +
                                             'thomas/Sunsimiao-InternLM-01M',
                                             trust_remote_code=True).cuda()
model = model.eval()
response, history = model.chat(tokenizer, "晚上睡不着怎么办？", history=[])
print(response)

