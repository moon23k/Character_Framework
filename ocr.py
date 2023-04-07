import json, glob, PIL, torch
from transformers import TrOCRProcessor, VisionEncoderDecoderModel



def main():
    #setup
    generated = []
    m_name = 'microsoft/trocr-large-printed'
    processor = TrOCRProcessor.from_pretrained(m_name)
    model = VisionEncoderDecoderModel.from_pretrained(m_name)


    for file in glob.glob('orc_data/*'):
        img = PIL.Image.open(file).convert("RGB")

        pixel_values = processor(images=img, return_tensors="pt").pixel_values
        generated_ids = model.generate(pixel_values)
        generated_text = processor.batch_decode(generated_ids, skip_special_tokens=True)[0]        

        generated.append(generated_text)

    with open('data/ocr.json', 'w') as f:
        json.dump(f, generated)
    assert os.path.exists('data/ocr.json')



if __name__ == '__main__':
    assert os.path.exists('data/orc_data')
    main()