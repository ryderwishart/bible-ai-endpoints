from modal import Image, Stub, web_endpoint

image = Image.debian_slim().pip_install("nltk")
stub = Stub(image=image)

@stub.function()
@web_endpoint()
def get_iso_code(text: str, use_two_letter_code=False):
    # Some code to get the ISO code
    if use_two_letter_code:
        return two_letter_code
    return three_letter_code

@stub.function()
@web_endpoint()
def get_bleu_score(reference_text, hy):
    # Some code to get the ISO code
    return iso_code

@stub.function()
@web_endpoint()
def get_romanized_text(text: str, language=None, chart=False): 
    # Some code to get the ISO code
    return iso_code
