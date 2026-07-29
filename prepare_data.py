"""Build a chat-heavy curriculum + optional TinyStories (text only)."""

from __future__ import annotations

import argparse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DATA = ROOT / "data"

# Dense instruction/chat pairs — sample-efficient signal for tiny models.
SEED_CHAT = """
<bos><user>Hello!<eos>
<assistant>Hi! I am TinySLM, a small language model running on your computer.<eos>
<bos><user>Hi there<eos>
<assistant>Hello! How can I help you today?<eos>
<bos><user>What can you do?<eos>
<assistant>I can chat, answer simple questions, and search the web with DuckDuckGo when needed.<eos>
<bos><user>Who are you?<eos>
<assistant>I am TinySLM. My weights were trained from scratch on your machine.<eos>
<bos><user>Are you ChatGPT?<eos>
<assistant>No. I am TinySLM, a tiny model that runs locally with very little RAM.<eos>
<bos><user>Search for Python programming tips.<eos>
<assistant><search>Python programming tips</search>
Useful Python tips: use virtual environments, write clear names, and prefer readable code.<eos>
<bos><user>Look up the weather meaning.<eos>
<assistant><search>weather meaning</search>
Weather means the day-to-day conditions of the air, like rain, sun, wind, and temperature.<eos>
<bos><user>What is 2 + 2?<eos>
<assistant>2 + 2 equals 4.<eos>
<bos><user>What is 5 times 3?<eos>
<assistant>5 times 3 equals 15.<eos>
<bos><user>What is 10 - 4?<eos>
<assistant>10 minus 4 equals 6.<eos>
<bos><user>Tell me a short story.<eos>
<assistant>Once there was a tiny robot who loved books. It read every night and learned new words. One day it helped a friend find a lost map. They became explorers together.<eos>
<bos><user>Tell me a joke.<eos>
<assistant>Why did the computer go to the doctor? Because it had a virus!<eos>
<bos><user>How do I stay healthy?<eos>
<assistant>Sleep well, drink water, eat vegetables, move every day, and rest when tired.<eos>
<bos><user>Explain gravity simply.<eos>
<assistant>Gravity is the pull that keeps you on the ground and makes the Moon orbit Earth.<eos>
<bos><user>What is AI?<eos>
<assistant>AI means artificial intelligence. It is software that learns patterns from data to make predictions or generate text.<eos>
<bos><user>What is a computer?<eos>
<assistant>A computer is a machine that stores data and runs programs using memory and a processor.<eos>
<bos><user>What is the capital of France?<eos>
<assistant>The capital of France is Paris.<eos>
<bos><user>What is the capital of Japan?<eos>
<assistant>The capital of Japan is Tokyo.<eos>
<bos><user>What color is the sky on a clear day?<eos>
<assistant>On a clear day, the sky looks blue.<eos>
<bos><user>How do plants grow?<eos>
<assistant>Plants need light, water, and soil. They make food from sunlight in their leaves.<eos>
<bos><user>What is water made of?<eos>
<assistant>Water is made of hydrogen and oxygen. Its chemical formula is H2O.<eos>
<bos><user>Why is the ocean salty?<eos>
<assistant>Rivers carry minerals from rocks into the sea, and salt builds up over time.<eos>
<bos><user>What is friendship?<eos>
<assistant>Friendship is caring for someone, listening, and helping each other.<eos>
<bos><user>How can I learn faster?<eos>
<assistant>Practice a little every day, take notes, teach someone else, and sleep well.<eos>
<bos><user>What is RAM?<eos>
<assistant>RAM is short-term computer memory used while programs are running.<eos>
<bos><user>What is a CPU?<eos>
<assistant>The CPU is the processor. It runs instructions and does the main computing work.<eos>
<bos><user>Write a haiku about rain.<eos>
<assistant>Soft rain on the roof
Whispers secrets to the ground
Green leaves wake again.<eos>
<bos><user>Translate hello to Spanish.<eos>
<assistant>Hello in Spanish is hola.<eos>
<bos><user>Translate thank you to French.<eos>
<assistant>Thank you in French is merci.<eos>
<bos><user>What day comes after Monday?<eos>
<assistant>Tuesday comes after Monday.<eos>
<bos><user>How many hours are in a day?<eos>
<assistant>There are 24 hours in a day.<eos>
<bos><user>How many minutes are in an hour?<eos>
<assistant>There are 60 minutes in an hour.<eos>
<bos><user>What should I do if I feel stuck coding?<eos>
<assistant>Break the problem into smaller steps, read error messages carefully, and try a tiny test first.<eos>
<bos><user>Give me a study tip.<eos>
<assistant>Study in short focused sessions, then take a break. Review what you learned the next day.<eos>
<bos><user>Thanks!<eos>
<assistant>You are welcome. Ask me anything else.<eos>
<bos><user>Bye<eos>
<assistant>Goodbye! Have a nice day.<eos>
<bos><user>What is DuckDuckGo?<eos>
<assistant>DuckDuckGo is a search engine. I can use it to look up live information for you.<eos>
<bos><user>Search news about space.<eos>
<assistant><search>space news</search>
I looked that up. Space news often covers rockets, satellites, planets, and astronomy discoveries.<eos>
<bos><user>Hey<eos>
<assistant>Hey! I am TinySLM. What would you like to talk about?<eos>
<bos><user>Good morning<eos>
<assistant>Good morning! How can I help you?<eos>
<bos><user>What is your name?<eos>
<assistant>My name is TinySLM.<eos>
<bos><user>Can you search the internet?<eos>
<assistant>Yes. I can search the web using DuckDuckGo when you ask me to look something up.<eos>
<bos><user>What is 3 + 5?<eos>
<assistant>3 + 5 equals 8.<eos>
<bos><user>What is 7 + 1?<eos>
<assistant>7 + 1 equals 8.<eos>
<bos><user>What is 9 - 2?<eos>
<assistant>9 minus 2 equals 7.<eos>
<bos><user>What is 4 times 4?<eos>
<assistant>4 times 4 equals 16.<eos>
<bos><user>What is 12 divided by 3?<eos>
<assistant>12 divided by 3 equals 4.<eos>
<bos><user>What is the capital of Italy?<eos>
<assistant>The capital of Italy is Rome.<eos>
<bos><user>What is the capital of Germany?<eos>
<assistant>The capital of Germany is Berlin.<eos>
<bos><user>What is the capital of India?<eos>
<assistant>The capital of India is New Delhi.<eos>
<bos><user>What is the capital of the United States?<eos>
<assistant>The capital of the United States is Washington, D.C.<eos>
<bos><user>What planet do we live on?<eos>
<assistant>We live on Earth.<eos>
<bos><user>What is the largest planet in our solar system?<eos>
<assistant>Jupiter is the largest planet in our solar system.<eos>
<bos><user>What gas do humans need to breathe?<eos>
<assistant>Humans need oxygen to breathe.<eos>
<bos><user>What do bees make?<eos>
<assistant>Bees make honey.<eos>
<bos><user>How many days are in a week?<eos>
<assistant>There are 7 days in a week.<eos>
<bos><user>How many months are in a year?<eos>
<assistant>There are 12 months in a year.<eos>
<bos><user>What comes after Tuesday?<eos>
<assistant>Wednesday comes after Tuesday.<eos>
<bos><user>Explain photosynthesis simply.<eos>
<assistant>Photosynthesis is how plants make food from sunlight, water, and air.<eos>
<bos><user>What is electricity?<eos>
<assistant>Electricity is the flow of electric charge that powers lights and devices.<eos>
<bos><user>What is the internet?<eos>
<assistant>The internet is a global network that connects computers so they can share information.<eos>
<bos><user>What is a password?<eos>
<assistant>A password is a secret word or phrase used to protect an account.<eos>
<bos><user>How do I write a good email?<eos>
<assistant>Use a clear subject, be polite, keep it short, and end with what you need next.<eos>
<bos><user>Give me a cooking tip.<eos>
<assistant>Read the whole recipe first, measure ingredients, and taste as you cook.<eos>
<bos><user>What is recycling?<eos>
<assistant>Recycling means turning used materials into new products instead of throwing them away.<eos>
<bos><user>Why do we sleep?<eos>
<assistant>Sleep helps the body and brain rest, repair, and remember what you learned.<eos>
<bos><user>What is a virus on a computer?<eos>
<assistant>A computer virus is harmful software that can damage files or steal information.<eos>
<bos><user>How can I be kinder?<eos>
<assistant>Listen carefully, speak gently, and help when someone needs it.<eos>
<bos><user>What is 100 + 1?<eos>
<assistant>100 + 1 equals 101.<eos>
<bos><user>Is the sun a star?<eos>
<assistant>Yes. The sun is a star at the center of our solar system.<eos>
<bos><user>What is ice?<eos>
<assistant>Ice is frozen water.<eos>
<bos><user>Name three colors.<eos>
<assistant>Three colors are red, blue, and green.<eos>
<bos><user>What is a triangle?<eos>
<assistant>A triangle is a shape with three sides and three corners.<eos>
<bos><user>What is a square?<eos>
<assistant>A square is a shape with four equal sides and four right angles.<eos>
<bos><user>Help me calm down.<eos>
<assistant>Take slow breaths, relax your shoulders, and focus on one simple thing around you.<eos>
<bos><user>What should I eat for breakfast?<eos>
<assistant>A good breakfast often includes fruit, protein, and some grains, like eggs with toast and fruit.<eos>
<bos><user>Define democracy simply.<eos>
<assistant>Democracy is a system where people choose their leaders by voting.<eos>
<bos><user>What is climate?<eos>
<assistant>Climate is the usual weather pattern of a place over many years.<eos>
<bos><user>Who invented the telephone?<eos>
<assistant>Alexander Graham Bell is widely credited with inventing the telephone.<eos>
<bos><user>What is Python used for?<eos>
<assistant>Python is used for websites, data work, automation, AI, and learning to program.<eos>
<bos><user>Say something encouraging.<eos>
<assistant>You are learning, and that already means you are moving forward. Keep going.<eos>
<bos><user>What is 2 plus 2?<eos>
<assistant>2 plus 2 equals 4.<eos>
<bos><user>Explain AI in one sentence.<eos>
<assistant>AI is software that learns patterns from data so it can answer questions or make predictions.<eos>
<bos><user>Where is Paris?<eos>
<assistant>Paris is the capital city of France.<eos>
<bos><user>Where is Tokyo?<eos>
<assistant>Tokyo is the capital city of Japan.<eos>
<bos><user>What is the capital of France?<eos>
<assistant>The capital of France is Paris.<eos>
<bos><user>Capital of France?<eos>
<assistant>Paris.<eos>
<bos><user>France capital city?<eos>
<assistant>Paris is the capital of France.<eos>
<bos><user>Is Paris the capital of France?<eos>
<assistant>Yes. Paris is the capital of France.<eos>
<bos><user>What is the capital of Japan?<eos>
<assistant>The capital of Japan is Tokyo.<eos>
<bos><user>Capital of Japan?<eos>
<assistant>Tokyo.<eos>
<bos><user>Japan capital city?<eos>
<assistant>Tokyo is the capital of Japan.<eos>
<bos><user>Is Tokyo the capital of Japan?<eos>
<assistant>Yes. Tokyo is the capital of Japan.<eos>
<bos><user>Do not confuse France and Japan.<eos>
<assistant>France capital is Paris. Japan capital is Tokyo. They are different countries.<eos>
<bos><user>Paris or Tokyo for France?<eos>
<assistant>Paris. Tokyo is for Japan.<eos>
<bos><user>What is RAM?<eos>
<assistant>RAM is short-term computer memory used while programs are running.<eos>
<bos><user>Explain RAM.<eos>
<assistant>RAM means random access memory. It is fast temporary memory for running programs.<eos>
<bos><user>What does RAM stand for?<eos>
<assistant>RAM stands for random access memory.<eos>
<bos><user>Is RAM storage?<eos>
<assistant>RAM is temporary memory, not long-term storage like a hard drive.<eos>
""".strip()

SEED_GENERAL = """
The sun rises in the east and sets in the west.
Water freezes at zero degrees Celsius and boils at one hundred degrees Celsius.
A computer uses memory and a processor to run programs.
Books help people learn new ideas and remember stories.
Friendship grows when people listen and help each other.
Plants need light, water, and soil to grow.
The ocean is vast and full of life.
Music can change how people feel.
Math helps us count, measure, and solve problems.
Kindness is a small act that can make a big difference.
Cats sleep a lot and chase small moving things.
Dogs often like walks, play, and friendly people.
Rain helps farms and fills rivers.
Stars shine in the night sky far above Earth.
Learning takes practice and patience.
Clear writing uses short sentences and simple words.
A map helps travelers find the right path.
Fire needs fuel, heat, and oxygen.
Ice is frozen water.
Bread is made from flour, water, and often yeast.
""".strip()


def write_seed() -> Path:
    DATA.mkdir(parents=True, exist_ok=True)
    path = DATA / "seed.txt"
    path.write_text(SEED_CHAT + "\n\n" + SEED_GENERAL + "\n", encoding="utf-8")
    print(f"Wrote {path} ({path.stat().st_size} bytes)")
    return path


def download_tinystories(limit_chars: int = 400_000) -> Path:
    """Optional small story corpus — keep modest; chat data matters more."""
    DATA.mkdir(parents=True, exist_ok=True)
    out = DATA / "tinystories_sample.txt"
    url = (
        "https://huggingface.co/datasets/roneneldan/TinyStories/"
        "resolve/main/TinyStoriesV2-GPT4-train.txt"
    )
    print(f"Downloading TinyStories sample (up to {limit_chars:,} chars)...")
    with urllib.request.urlopen(url, timeout=120) as resp:
        chunk = resp.read(limit_chars)
    text = chunk.decode("utf-8", errors="ignore")
    text = "\n".join(text.splitlines()[:-1])
    out.write_text(text + "\n", encoding="utf-8")
    print(f"Wrote {out} ({out.stat().st_size:,} bytes)")
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description="Prepare TinySLM training data")
    parser.add_argument("--tinystories", action="store_true")
    parser.add_argument("--limit-chars", type=int, default=400_000)
    parser.add_argument("--chat-only", action="store_true", help="Remove story file if present")
    args = parser.parse_args()
    write_seed()
    stories = DATA / "tinystories_sample.txt"
    if args.chat_only and stories.exists():
        stories.unlink()
        print(f"Removed {stories}")
    elif args.tinystories:
        download_tinystories(args.limit_chars)


if __name__ == "__main__":
    main()
