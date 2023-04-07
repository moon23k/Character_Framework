import os, json, torch
from tqdm import tqdm
from datasets import load_dataset
from torch.utils.data import DataLoader
from transformers import (set_seed, 
						  T5TokenizerFast,
						  T5ForConditionalGeneration)



class Dataset(torch.utils.data.Dataset):
    def __init__(self, data):
        super().__init__()
        self.data = data

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):        
        return self.data[idx]['en'], self.data[idx]['de']



class Collator(object):
    def __init__(self, tokenizer):
        self.tokenizer = tokenizer

    def __call__(self, batch):
        src_batch, trg_batch = [], []

        for src, trg in batch:
            src_batch.append(src) 
            trg_batch.append(trg)

        src_batch = self.tokenizer(src_batch, padding=True, truncation=True, return_tensors='pt')

        return {'input_ids': src_batch.input_ids, 
                'attention_mask': src_batch.attention_mask,
                'labels': trg_batch}                



def generate_data(model, dataloader, device):
	orig_volumn, generated = 0, []
	
	for batch in tqdm(dataloader):    
	    input_ids = batch['input_ids'].to(device)
	    attention_mask = batch['attention_mask'].to(device)
	    labels = batch['labels']
	    
	    orig_volumn += input_ids.size(0)

	    preds = model.generate(input_ids, attention_mask)
	    preds = tokenizer.batch_decode(preds)

	    for p, l in zip(preds, labels):
	    	if p == l:
	    		continue
		    generated.append{'src': p, 'trg': l}

    print(f"{round(len(generated) / orig_volumn , 2)} percent has augmented!")
	return generated



def main():
	#setup
	set_seed(42)
	batch_size = 32
	m_name = 't5-base'
	device = torch.cuda("cuda" if torch.cuda.is_available() else "cpu")

	model = T5ForConditionalGeneration.from_pretrained(m_name).to(device)
	tokenizer = T5TokenizerFast.from_pretrained(m_name)
    data = load_dataset('wmt14', 'de-en', split='test')['translation'][:100]

    datalaoder = DataLoader(Dataset(data),
    						batch_size=batch_size,
    						shuffle=False,
    						collate_fn=Collator(tokenizer),
    						pin_memory=True, 
    						num_workers=2)

    #Generate Data
    generated = generate_data(model, dataloader, device)


  	with open('data/generated.json', 'w') as f:
  		json.dump(f, generated)
  	assert os.path.exists('data/generated.json')


if __name__ == '__main__':
	main()