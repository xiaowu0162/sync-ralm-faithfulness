import openai
import backoff
import json
from vllm import LLM, SamplingParams
from nltk.tokenize import PunktSentenceTokenizer


backend = 'nltk'   # vllm, openai, nltk
print('Using backend "{}" for split_sentences.'.format(backend))

if backend == 'vllm':
    model = 'mistralai/Mistral-7B-Instruct-v0.1'
    llm = LLM(model=model, quantization='awq' if 'awq' in model.lower() else None,
              download_dir='/local2/diwu/selfrag_model_cache')
    sampling_params = SamplingParams(temperature=0, top_p=1, max_tokens=1000)
elif backend == 'openai':
    model = 'gpt-3.5-turbo'
    openai.api_key = ''   # your API key
    openai.organization = ''   # your organization ID (comment out this line to use default)
elif backend == 'nltk':
    model = PunktSentenceTokenizer()
else:
    raise NotImplementedError

@backoff.on_exception(backoff.expo, (openai.error.RateLimitError,
                                     openai.error.ServiceUnavailableError,
                                     openai.error.APIError))
def chat_completions_with_backoff(**kwargs):
    return openai.ChatCompletion.create(**kwargs)


prompt = """
    Task: Read the paragraph and separate sentences with the special identifier [sep]. Be mindful of decimals, ellipses, quotation marks, and names with periods. Consider complex sentences and various edge cases.

    Example 1:
    Input: "Yesterday, I visited Dr. Emily O. Watson. She said, 'Your health is fine.' It was a relief. Later, at 5 p.m., I went to the gym."
    Output: "Yesterday, I visited Dr. Emily O. Watson.[sep]She said, 'Your health is fine.'[sep]It was a relief.[sep]Later, at 5 p.m., I went to the gym."

    Example 2:
    Input: "In the U.S., people often celebrate Thanksgiving with their families. It's on Nov. 25th. This year, my friend, T.J. Miller, invited me to his place. He lives in Washington, D.C."
    Output: "In the U.S., people often celebrate Thanksgiving with their families.[sep]It's on Nov. 25th.[sep]This year, my friend, T.J. Miller, invited me to his place.[sep]He lives in Washington, D.C."

    Example 3:
    Input: "The total was $45.67. 'Quite expensive!' I thought. However, these tools are essential for the project we're undertaking. Aren't they?"
    Output: "The total was $45.67.[sep]'Quite expensive!' I thought.[sep]However, these tools are essential for the project we're undertaking.[sep]Aren't they?"

    Example 4:
    Input: "The meeting started at 3:00 p.m. It ended at 4:30 p.m. During the meeting, Mr. Johnson said, 'We must improve our sales by 30%. That's our goal.'"
    Output: "The meeting started at 3:00 p.m.[sep]It ended at 4:30 p.m.[sep]During the meeting, Mr. Johnson said, 'We must improve our sales by 30%. That's our goal.'"

    Example 5:
    Input: "Look at those clouds... It looks like it's going to rain. Oh, look! There's a rainbow. Isn't it beautiful?"
    Output: "Look at those clouds...[sep]It looks like it's going to rain.[sep]Oh, look! There's a rainbow.[sep]Isn't it beautiful?"

    New paragraph to process:
    Input: {}
    Output:"""


def split_sentences(text):
    if backend == 'vllm':
        out = llm.generate([prompt.format(text)], sampling_params, use_tqdm=False)[0].outputs[0].text
        out = out.lstrip(' ')
        return [x for x in out.split('[sep]') if x.strip() != ""]
    elif backend == 'openai':
        cur_prompt = {
            "model": model, 
            "messages": [{"role": "system", "content": "You are a helpful assistant that helps text processing."},
                         {"role": "user", "content": prompt.format(text)}],
            "n": 1,
            "temperature": 0,
            "max_tokens": 1000
        }
        response = chat_completions_with_backoff(**cur_prompt)
        out = response['choices'][0]['message']['content']
        return [x for x in out.split('[sep]') if x.strip() != ""]
    elif backend == 'nltk':
        out = model.tokenize(text)
        return out


if __name__ == '__main__':
    sample_input = "Last week, I attended the International Science Conference 2024 held in New York City. The event kicked off with a keynote speech by Dr. Ava R. Patel, who discussed the latest trends in renewable energy technologies. \"In the next decade, we aim to reduce carbon emissions by 5.0%,\" she stated confidently. The crowd was visibly impressed, murmuring in agreement. Following the opening ceremony, I joined a workshop titled \"The Future of AI in Environmental Science.\" It was fascinating to see how AI could predict climate change patterns with high accuracy. Lunch was at 1:30 p.m., and I had the chance to chat with Dr. Patel. She mentioned her recent trip to Greenland, \"It's alarming how fast the ice is melting.\" In the afternoon, there were multiple sessions on sustainable materials. One speaker, Prof. John T. Lee from MIT, showcased a new biodegradable plastic that decomposes in under a year. \"This could revolutionize the packaging industry,\" he exclaimed. The day ended with a panel discussion on ethical considerations in scientific research. As I left the conference hall, I couldn't help but feel optimistic about the future of science."
    sample_input = "Yesterday, I visited Dr. Emily O. Watson. She said, 'Your health is fine.' It was a relief. Later, at 5 p.m., I went to the gym."
    print(sample_input)
    print(split_sentences(sample_input))
