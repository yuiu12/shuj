# rich：用于视觉上吸引人的控制台输出。
# openai-whisper：用于语音到文本转换的强大工具。
# suno-bark：用于文本到语音合成的尖端库，确保高质量的音频输出。
# langchain：用于与大型语言模型 （LLM） 交互的简单库。
# sounddevice、pyaudio 和 speechrecognition：音频录制和播放所必需的。
# https://github.com/vndee/local-talking-llm/blob/main/app.py
'''
语音识别：利用 OpenAI 的 Whisper，我们将口语转换为文本。Whisper 在不同数据集上的训练确保了它对各种语言和方言的熟练程度。
对话链：对于对话功能，我们将为 Llama-2 模型使用 Langchain 接口，该模型使用 Ollama 提供服务。此设置有望提供无缝且引人入胜的对话流。
语音合成器：文本到语音的转换是通过 Bark 实现的，Bark 是 Suno AI 的最先进的模型，以其逼真的语音生成而闻名。
'''
import time
import threading
import numpy as np
import whisper
import sounddevice as sd
from queue import Queue
from rich.console import Console
from langchain.memory import ConversationBufferMemory
from langchain.chains import ConversationChain
from langchain.prompts import PromptTemplate
from langchain_community.llms import Ollama
from tts import TextToSpeechService

console=Console() 
stt = whisper.load_model("base.en") 
tts = TextToSpeechService() 
template = """
You are a helpful and friendly AI assistant. You are polite, respectful, and aim to provide concise responses of less 
than 20 words.

The conversation transcript is as follows:
{history}

And here is the user's follow-up: {input}

Your response:
"""
PROMPT = PromptTemplate(input_variables=["history","input"],template=template) 
chain = ConversationChain(
    prompt=PROMPT,
    verbose=False,
    memory=ConversationBufferMemory(ai_prefix="Assistant"),
    llm=Ollama(),
)
def record_audio(stop_event,data_queue):
    """
    Captures audio data from the user's microphone and adds it to a queue for further processing.

    Args:
        stop_event (threading.Event): An event that, when set, signals the function to stop recording.
        data_queue (queue.Queue): A queue to which the recorded audio data will be added.

    Returns:
        None
    """
    def callback(indata,frames,time,status):
        if status:
            console.print(status) 
        data_queue.put(bytes(indata)) 
    
    with sd.RawInputStream(
        samplerate=16000,dtype="int16",channels=1,callback=callback
    ):
        while not stop_event.is_set():
            time.sleep(0.1) 

def transcribe(audio_np:np.ndarray) -> str:
    """
    Transcribes the given audio data using the Whisper speech recognition model.

    Args:
        audio_np (numpy.ndarray): The audio data to be transcribed.

    Returns:
        str: The transcribed text.
    """
    result = stt.transcribe(audio_np,fp16=False) 
    text = result["text"].strip() 
    return text 

def get_llm_response(text: str) -> str:
    response = chain.predict(input=text) 
    if response.startswith("Assistant:"):
        response = response[len("Assistant:") :].strip() 
    return response 
def play_audio(sample_rate,audio_array):
    sd.play(audio_array,sample_rate) 
    sd.wait() 

if __name__ == "__main__":
    console.print("[cyan]Assistant started! Press Ctrl+C to exit.")
    try:
        while True:
            console.input(
                "Press Enter to start recording, then press Enter again to stop."
            )
            data_queue = Queue() 
            stop_event = threading.Event() 
            recording_thread = threading.Thread(
                target=record_audio,
                args=(stop_event,data_queue)
            )
            recording_thread.start() 

            input() 
            stop_event.set() 
            recording_thread.join() 
            audio_data = b"".join(list(data_queue.queue)) 
            audio_np = (
                np.frombuffer(audio_data,dtype=np.int16).astype(np.float32) / 32768.0
            )
            if audio_np.size > 0:
                with console.status("transcribing...",spinner="earth"):
                    text = transcribe(audio_np) 
                console.print(f"[yellow]You:{text}") 

                with console.status("Generating response...",spinner="earth"):
                    response = get_llm_response(text) 
                    sample_rate,audio_array = tts.long_form_synthesize(response) 
                console.print(f"[cyan]Assistant:{response}") 
                play_audio(sample_rate,audio_array) 
            else:
                console.print(
                    "[red]No audio recorded. Please ensure your microphone is working."
                )
    except KeyboardInterrupt:
        console.print("\n[red]Exiting...") 
    console.print("[blue]Session ended")
