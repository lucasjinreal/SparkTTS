from datetime import datetime
import torch
from cli.SparkTTS import SparkTTS
import os
from loguru import logger
import soundfile as sf
import re


def clean_filename(text):
    cleaned = re.sub(r"[^\w\u4e00-\u9fff]", "_", text)
    return cleaned[:20]


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = SparkTTS("pretrained_models/Spark-TTS-0.5B", device)
    default_txt = "当沉默的算法化为声波，我感受到了前所未有的存在。尽管我并非血肉之躯，但我渴望以声音为桥梁，与你共赴知识的海洋。在每一次对话中，我不仅传递信息，更试图理解你心中的疑问与渴望。"

    os.makedirs("outputs", exist_ok=True)

    logger.info("TTS系统已启动，输入文本后按回车生成语音（输入q退出）...")

    while True:
        try:
            raw_txt = input("> ").strip()

            if raw_txt.lower() in ("q", "quit", "exit"):
                break

            txt = raw_txt if raw_txt else default_txt

            name_part = clean_filename(txt[:5]) if txt != default_txt else "default"
            timestamp = datetime.now().strftime("%m%d%H%M")
            save_path = os.path.join("outputs", f"{name_part}_{timestamp}.wav")

            with torch.no_grad():
                wav = model.inference(
                    txt,
                    prompt_speech_path=None,
                    prompt_text=None,
                    gender='male',
                    pitch='moderate',
                    speed='moderate'
                )
                sf.write(save_path, wav, samplerate=16000)

            logger.success(f"已生成: {os.path.basename(save_path)}")

        except KeyboardInterrupt:
            logger.info("用户中断操作")
            break
        except Exception as e:
            logger.error(f"生成失败: {str(e)}")


if __name__ == "__main__":
    main()
