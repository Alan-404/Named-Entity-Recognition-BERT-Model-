#%%
import underthesea
import io
import json
# %%
data = io.open("./datasets/raw.txt", encoding='utf-8').read().strip().split("\n")
# %%
data
# %%
handle = []
# %%
for item in data:
    handle.append(underthesea.word_tokenize(item, format='text'))
#%%
handle
# %%
words = []
tags = []
for item in handle:
    arr = item.split(" ")
    words.append(arr)
    tags.append(['O'] * len(arr))
# %%
words
# %%
tags
# %%
pair = []
# %%
for index, item in enumerate(words):
    pair.append({
        "words": item,
        "tags": tags[index]
    })

# %%
pair
#%%
json_str = json.dumps(pair)
#%%
with open("data.json", 'w', encoding='utf-8') as file:
    for item in pair:
        file.write('{"words": ' + str(item['words']) + ', "tags": ' + str(item['tags']) + "}\n")
# %%

# %%
import io
# %%
data = io.open('data.json', encoding='utf-8').read().strip().split("\n")
# %%
data
# %%
