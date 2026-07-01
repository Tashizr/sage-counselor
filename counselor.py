import random
import re
import sys
import time
import os
import json
import numpy as np
import pandas as pd
import joblib

USER = 0
BOT = 1

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "self-harm", "hurt myself", "not worth living", "better off dead",
    "end it all", "take my own life", "don't want to live",
    "wish i was dead", "end my suffering", "never wake up",
    "harm myself", "cut myself", "suicidal",
]

SLANG_MAP = {
    "fr": "for real", "frfr": "for real for real",
    "ngl": "not going to lie", "lowkey": "somewhat", "highkey": "very much",
    "rn": "right now", "atm": "at the moment",
    "idk": "i don't know", "idc": "i don't care", "imo": "in my opinion",
    "yk": "you know", "alr": "alright",
    "nah": "no", "ye": "yes", "yuh": "yes",
    "bet": "okay", "say less": "i understand",
    "no cap": "no lie", "cap": "lie",
    "deadass": "seriously",
    "im cooked": "i'm in trouble",
    "im tweaking": "i'm panicking",
    "its wraps": "it's over",
    "we ball": "we'll be fine",
    "i sold": "i failed",
    "i fumbled": "i made a mistake and lost something",
    "ghosted": "stopped contacting without explanation",
    "left me on delivered": "read my message but didn't reply",
    "left on read": "read my message and didn't reply",
    "friend zoned": "rejected as a romantic interest",
    "down bad": "in a difficult situation",
    "spiraling": "losing emotional control",
    "crashing out": "losing control completely",
    "bro": "friend", "bruh": "friend", "dawg": "friend",
    "tf": "what the", "wtf": "what the",
    "lmao": "laughing", "lol": "laughing",
}

SLANG_EMOJI_MAP = {
    "😭": "crying", "💀": "dead", "🙏": "please",
    "❤️": "love", "💔": "heartbreak",
    "🥲": "tearful smile", "😔": "sad", "😕": "confused",
    "🙂": "okay", "😊": "happy",
}

SHORT_ANSWER_SET = {
    "yeah", "nah", "idk", "idc", "maybe", "sure", "kinda",
    "not really", "probably", "alr", "ok", "okay", "ye",
    "yuh", "ig", "i guess", "not sure", "i dunno",
}

SOFT_ENDINGS = {
    "anxiety": ["I'm here with you.", "We can sit with this for a while.", "Take your time."],
    "sadness": ["I'm here with you.", "We can stay with this for a bit.", "Take your time."],
    "anger": ["I'm here with you.", "Let that sit for a moment.", "Take your time."],
    "stress": ["We can slow down here.", "Take your time.", "I'm here with you."],
    "sleep": ["Take your time.", "We can let that sit.", "I'm here with you."],
    "relationships": ["I'm here with you.", "Take your time with this.", "We can sit with this."],
    "work": ["Take your time.", "I'm here with you.", "We can slow down here."],
    "confusion": ["Take your time.", "We can sit with the question.", "I'm here with you."],
    "positive": ["That's really nice.", "I'm glad for you.", "That's good to hear."],
    "general": ["Take your time.", "I'm here with you.", "We can sit with this."],
}

STOP_WORDS = {
    "i", "me", "my", "myself", "we", "our", "you", "your", "yourself",
    "he", "him", "she", "her", "it", "they", "them", "their",
    "a", "an", "the", "and", "or", "but", "if", "because", "so",
    "is", "am", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "do", "does", "did", "will", "would",
    "can", "could", "shall", "should", "may", "might",
    "not", "no", "nor", "just", "very", "too", "really", "also",
    "about", "with", "without", "for", "of", "to", "in", "into",
    "on", "at", "by", "from", "up", "down", "out", "off", "now",
    "over", "under", "again", "further", "then", "once",
    "here", "there", "when", "where", "why", "how",
    "all", "each", "every", "both", "few", "more", "most", "much",
    "some", "any", "thing", "things", "like", "get", "got",
    "feel", "feels", "feeling", "felt", "make", "makes",
    "go", "went", "come", "came", "take", "took", "know",
    "think", "thought", "want", "needs", "need", "says", "say",
    "cannot", "cannot", "dont", "doesnt", "wont", "cant",
    "always", "never", "ever", "even", "still", "already",
    "much", "many", "lot", "lots", "some", "something",
    "nothing", "everything", "anything", "everyone", "someone",
    "maybe", "perhaps", "probably", "actually", "basically",
    "definitely", "certainly", "absolutely", "quite", "pretty",
    "ago", "since", "until", "while", "during", "before", "after",
    "today", "yesterday", "tomorrow", "tonight", "morning",
    "also", "though", "although", "however", "therefore",
    "well", "oh", "ah", "okay", "ok", "yeah", "yes", "no",
    "sure", "right", "fine", "good", "bad",
    "im", "dont", "cant", "wont", "didnt", "wasnt", "isnt",
    "arent", "couldnt", "wouldnt", "shouldnt", "havent",
    "hasnt", "hadnt", "doesnt", "thats", "theres",
    "its", "ive", "youve", "weve", "theyve",
    "youre", "theyre", "were",
}

REFLECTION_STARTS = {
    "anxiety": [
        "It sounds like there's a lot of weight in what you're describing.",
        "I wonder if this has been sitting with you for a while now.",
        "That worry sounds really present for you right now.",
        "It seems like your mind is working hard to keep you alert.",
        "I imagine that kind of unease is hard to shake off.",
    ],
    "sadness": [
        "It sounds like there's a heaviness that's hard to put into words.",
        "I wonder if this has been weighing on you more than you let on.",
        "It seems like there's a deep ache in what you're sharing.",
        "That kind of sadness sounds like it runs deep.",
        "I imagine that leaves you feeling pretty drained.",
    ],
    "anger": [
        "It sounds like something really got to you.",
        "I wonder if there's more underneath that frustration.",
        "It seems like this touched something important.",
        "That kind of anger usually points to something that matters deeply.",
        "I imagine that left you feeling pretty unheard.",
    ],
    "stress": [
        "It sounds like you're carrying a lot right now.",
        "I wonder if this has been building for a while.",
        "It seems like there's a lot pulling at your attention.",
        "That kind of pressure sounds exhausting.",
        "I imagine it's hard to find a moment to breathe.",
    ],
    "sleep": [
        "It sounds like your mind doesn't quiet down when you need it to.",
        "I wonder if there's something your mind is trying to process.",
        "It seems like rest has been hard to come by.",
        "That struggle to sleep sounds really draining.",
        "I imagine lying awake with your thoughts is exhausting.",
    ],
    "relationships": [
        "It sounds like there's something shifting in a relationship that matters to you.",
        "I wonder if this has been on your mind for a while.",
        "It seems like this connection means a lot to you.",
        "That kind of tension in a relationship is hard to carry.",
        "I imagine that leaves you feeling pretty torn.",
    ],
    "work": [
        "It sounds like work has been taking a lot out of you.",
        "I wonder if this has been building up over time.",
        "It seems like there's a mismatch between what you give and what you get.",
        "That kind of work stress can affect everything else too.",
        "I imagine that leaves you questioning things.",
    ],
    "confusion": [
        "It sounds like you're at a place where things aren't clear yet.",
        "I wonder if this uncertainty has been unsettling.",
        "It seems like you're weighing something important.",
        "That kind of not-knowing is genuinely uncomfortable.",
        "I imagine it's hard to move forward when things feel unclear.",
    ],
    "positive": [
        "It sounds like something good is happening for you.",
        "I wonder if this has been a long time coming.",
        "It seems like things are moving in a good direction.",
        "That kind of positive energy is really nice to hear.",
        "I imagine that feels pretty good.",
    ],
    "general": [
        "It sounds like there's something on your mind.",
        "I wonder if this has been sitting with you for a while.",
        "It seems like this is important to you.",
        "I appreciate you sharing that with me.",
        "I'm glad you're talking about this.",
    ],
}

VALIDATIONS = {
    "anxiety": [
        "It makes sense you'd feel that way.",
        "Anxiety is your body's way of trying to protect you, even when it overdoes it.",
    ],
    "sadness": [
        "It makes sense you'd feel that way.",
        "Sadness is a natural response when something matters to us.",
    ],
    "anger": [
        "Anyone might feel that way in your position.",
        "Anger often shows us when something we care about has been crossed.",
    ],
    "stress": [
        "Anyone would feel stretched under that load.",
        "It makes sense to feel overwhelmed with everything on your plate.",
    ],
    "sleep": [
        "It makes sense that would leave you feeling drained.",
        "Sleep struggles often reflect how full our minds are during the day.",
    ],
    "relationships": [
        "It makes sense that would hurt.",
        "Relationships touch such a deep part of us, so it's natural for this to affect you.",
    ],
    "work": [
        "It makes sense that would weigh on you.",
        "So much of our identity gets tied up in work, so this kind of frustration is natural.",
    ],
    "confusion": [
        "It makes sense to feel uncertain right now.",
        "Confusion often means something important is shifting.",
    ],
    "positive": [
        "That's really nice to hear.",
        "It's good you're noticing that.",
    ],
    "general": [
        "",
    ],
}

QUESTIONS = {
    "anxiety": [
        "What does that anxiety feel like in your body right now?",
        "When does that worry tend to show up most?",
        "What's the first thing that comes to mind when you think about what's worrying you?",
        "What part of this feels the hardest to sit with?",
    ],
    "sadness": [
        "When did that heavy feeling first start showing up?",
        "What does that sadness need you to understand about it?",
        "What's the situation that feels most connected to this sadness?",
        "Is there something specific that triggered this, or has it been building?",
    ],
    "anger": [
        "What happened that sparked this frustration?",
        "What does this anger want you to protect?",
        "What would need to happen for you to feel heard about this?",
        "When did you first notice this building up?",
    ],
    "stress": [
        "What part of this feels the heaviest right now?",
        "If you could set one thing down today, what would it be?",
        "What's pulling at your attention most urgently?",
        "What does your body feel like when you think about everything on your plate?",
    ],
    "sleep": [
        "What's going through your mind when you're lying awake?",
        "How long has this pattern been going on?",
        "What's your evening like before you try to sleep?",
        "Is it hard to fall asleep, or hard to stay asleep?",
    ],
    "relationships": [
        "What do you find yourself needing from them that you're not getting right now?",
        "What kind of conversation have you been wanting to have but haven't yet?",
        "What does your gut tell you about this situation?",
        "How long has this dynamic been building?",
    ],
    "work": [
        "What part of your work drains you the most?",
        "What would a better version of your work life look like?",
        "Is it the work itself, or the environment around it?",
        "What's been the hardest thing to navigate?",
    ],
    "confusion": [
        "What feels most unclear when you sit with it?",
        "What's the heart of what you're trying to figure out?",
        "What options are you weighing right now?",
        "What part of this feels the most important to resolve?",
    ],
    "positive": [
        "What's contributing to that good feeling?",
        "What helped bring this about?",
        "What does that feeling make you want to do?",
        "How does it feel in your body when you experience this?",
    ],
    "general": [
        "What's coming up for you as you share that?",
        "What feels most important about this to you?",
        "How long have you been thinking about this?",
        "What made you decide to bring this up today?",
    ],
}


class KnowledgeBase:
    def __init__(self, kb_dir=None):
        if kb_dir is None:
            kb_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
        self.kb_dir = kb_dir
        self.emotions = {}
        self.life_events = {}
        self.cognitive_patterns = {}
        self.behavior_patterns = {}
        self.conversations = []
        self.crisis_data = {}
        self.coping_strategies = {}
        self.language_patterns = {}
        self._loaded = False

    def load(self):
        if self._loaded:
            return
        if not os.path.isdir(self.kb_dir):
            print("Warning: knowledge_base directory not found.")
            return
        self._load_json("01_emotions.json", "emotions", "name")
        self._load_json("02_life_events.json", "life_events", "name")
        self._load_json("03_cognitive_patterns.json", "cognitive_patterns", "name")
        self._load_json("04_behavior_patterns.json", "behavior_patterns", "name")
        self._load_jsonl("06_conversation_examples.jsonl")
        self._load_json("07_crisis_detection.json", "crisis_data", "name")
        self._load_json("08_coping_strategies.json", "coping_strategies", "name")
        self._load_json("09_language_patterns.json", "language_patterns", "pattern")
        self._loaded = True

    def _load_json(self, fname, attr, key_field):
        fpath = os.path.join(self.kb_dir, fname)
        if not os.path.exists(fpath):
            return
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                data = json.load(f)
            items = data.get(list(data.keys())[1] if len(data.keys()) > 1 else list(data.keys())[0], [])
            if isinstance(items, list):
                store = {}
                for item in items:
                    k = item.get(key_field, "").lower()
                    if k:
                        store[k] = item
                setattr(self, attr, store)
        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}")

    def _load_jsonl(self, fname):
        fpath = os.path.join(self.kb_dir, fname)
        if not os.path.exists(fpath):
            return
        try:
            with open(fpath, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line:
                        self.conversations.append(json.loads(line))
        except Exception as e:
            print(f"  Warning: could not load {fname}: {e}")

    def get_coping_strategies(self, topic):
        self.load()
        strategies = []
        for name, strat in self.coping_strategies.items():
            targets = [t.lower() for t in strat.get("target", [])]
            if topic.lower() in targets or any(t in topic.lower() for t in targets):
                strategies.append(strat)
        return strategies[:3]

    def get_language_insight(self, phrase):
        self.load()
        phrase_lower = phrase.lower().strip(",.!?")
        for pat_name, pat in self.language_patterns.items():
            if pat_name in phrase_lower or phrase_lower in pat_name:
                return pat
        return None

    def get_emotion_info(self, emotion_name):
        self.load()
        return self.emotions.get(emotion_name.lower())

    def get_crisis_info(self, crisis_type):
        self.load()
        return self.crisis_data.get(crisis_type.lower())

    def get_cognitive_pattern(self, pattern_name):
        self.load()
        return self.cognitive_patterns.get(pattern_name.lower())


def detect_crisis(text):
    return any(kw in text.lower() for kw in CRISIS_KEYWORDS)


class Counselor:
    COMMON_WORDS = {
        "i", "me", "my", "myself", "we", "our", "you", "your", "he", "she", "it",
        "a", "an", "the", "and", "or", "but", "is", "am", "are", "was", "were",
        "have", "has", "do", "does", "did", "be", "been", "being",
        "not", "no", "so", "if", "all", "just", "can", "will",
    }

    def __init__(self, model_dir=None):
        self.history = []
        self.user_name = None
        self.session_count = 0
        self.kb = KnowledgeBase()
        if model_dir is None:
            model_dir = os.path.dirname(__file__)
        model_path = os.path.join(model_dir, "counselor_model.joblib")
        if os.path.exists(model_path):
            self.model = joblib.load(model_path)
            self.classes_ = self.model.named_steps["clf"].classes_
        else:
            print("Warning: no trained model found. Run train_model.py first.")
            self.model = None

    def _is_counseling_content(self, text):
        lower = text.lower().strip(",.!? ")
        words = lower.split()
        if not words:
            return False
        first_word = words[0].lower().split("'")[0]
        if first_word in self.COMMON_WORDS:
            return True
        if self.model is not None:
            topic, conf = self._predict_topic(text)
            if conf > 0.25:
                return True
        return False

    def _predict_topic(self, text):
        if self.model is None:
            return "general", 0.0
        df = pd.DataFrame({"text": [text]})
        probs = self.model.predict_proba(df["text"])
        idx = np.argmax(probs, axis=1)[0]
        confidence = np.max(probs, axis=1)[0]
        topic = self.classes_[idx]
        return topic, confidence

    def greet(self):
        return ("Hi, I'm here to listen. I help people talk through what's on their mind. "
                "I'm not a replacement for a licensed therapist. "
                "What's your name?")

    def respond(self, user_input):
        text = user_input.strip()
        if not text:
            self.history.append((USER, ""))
            return "I'm here. You can take your time."

        self.history.append((USER, text))

        if detect_crisis(text):
            return ("I'm really concerned about what you're saying. Please reach out to a crisis line now:\n"
                    "- National Suicide Prevention Lifeline: 988\n"
                    "- Crisis Text Line: Text HOME to 741741\n"
                    "There are people ready to help you through this.")

        if self.user_name is None:
            words = text.split()
            if len(words) <= 3 and not self._is_counseling_content(text):
                self.user_name = text.split()[0].strip(",.!?")
                self.session_count += 1
                return f"Nice to meet you, {self.user_name}. What's been on your mind lately?"
            self.user_name = "Friend"
            self.session_count += 1

        return self._generate(text)

    def _translate_slang(self, text):
        lower = text.lower()
        translated = lower
        for phrase, meaning in sorted(SLANG_MAP.items(), key=lambda x: -len(x[0])):
            if phrase in translated:
                translated = translated.replace(phrase, meaning)
        for emoji, meaning in SLANG_EMOJI_MAP.items():
            if emoji in text:
                translated += f" {meaning}"
        return translated, text if translated == lower else translated

    def _is_short_answer(self, text):
        lower = text.lower().strip(",.!? ")
        if lower in SHORT_ANSWER_SET:
            return True
        words = lower.split()
        if len(words) > 2:
            return False
        meaningful_keywords = {"anxious", "sad", "angry", "scared", "hurt", "tired",
                               "lonely", "stressed", "depressed", "worried", "lost",
                               "hopeless", "numb", "heartbroken", "grateful", "happy"}
        if any(w.strip(",.!?") in meaningful_keywords for w in words):
            return False
        if lower in ("i don't know", "i dunno", "not sure", "i'm not sure"):
            return True
        return True

    def _detect_secondary_emotion(self, text):
        emotion_words = {
            "anxiety": ["anxious", "worried", "nervous", "panic", "scared", "afraid", "dread"],
            "sadness": ["sad", "sorrow", "grief", "depressed", "lonely", "empty"],
            "anger": ["angry", "mad", "furious", "annoyed", "irritated"],
            "stress": ["stressed", "overwhelmed", "burned out", "pressure"],
            "confusion": ["confused", "lost", "unsure", "torn", "uncertain"],
        }
        lower = text.lower()
        found = []
        for topic, words in emotion_words.items():
            for w in words:
                if w in lower:
                    found.append(topic)
                    break
        return found[:2]

    def _build_soft_ending(self, topic):
        opts = SOFT_ENDINGS.get(topic, SOFT_ENDINGS["general"])
        return random.choice(opts)

    def _extract_key_phrases(self, text):
        words = text.lower().split()
        meaningful = [w.strip(",.!?;:'\"") for w in words
                      if w.strip(",.!?;:'\"") not in STOP_WORDS
                      and len(w.strip(",.!?;:'\"")) > 2]
        random.shuffle(meaningful)
        return meaningful[:3]

    def _build_reflection(self, text, key_words, topic):
        start = random.choice(REFLECTION_STARTS.get(topic, REFLECTION_STARTS["general"]))
        if key_words and topic != "general":
            kw = random.choice(key_words)
            patterns = [
                f"{start} Especially with {kw} on your mind.",
                f"{start} And it sounds like {kw} is at the center of it.",
                f"{start} I can hear how much {kw} is affecting you.",
            ]
            return random.choice(patterns)
        return start

    def _build_validation(self, topic):
        opts = VALIDATIONS.get(topic, VALIDATIONS["general"])
        if not opts or not opts[0]:
            return ""
        val = random.choice(opts)
        if val:
            return val[0].upper() + val[1:]
        return ""

    def _build_question(self, topic):
        opts = QUESTIONS.get(topic, QUESTIONS["general"])
        return random.choice(opts)

    def _check_history(self):
        if len(self.history) < 3:
            return None
        for i in range(len(self.history) - 2, 0, -2):
            prev_text = self.history[i][1]
            if len(prev_text.split()) > 3:
                return prev_text[:80]
        return None

    def _generate(self, text):
        raw_text = text
        translated_text, _ = self._translate_slang(text)

        topic, confidence = self._predict_topic(translated_text)
        key_words = self._extract_key_phrases(translated_text)

        is_short = self._is_short_answer(raw_text)

        if is_short:
            prev = self._check_history()
            if prev:
                return self._handle_short_answer(topic, prev)

        secondary = self._detect_secondary_emotion(translated_text)

        reflection = self._build_reflection(translated_text, key_words, topic)
        validation = self._build_validation(topic)
        question = self._build_question(topic)
        soft = self._build_soft_ending(topic)

        parts = [reflection]
        if validation:
            parts.append(validation)
        if secondary and len(secondary) > 0 and secondary[0] != topic:
            if random.random() < 0.3:
                parts.append(f"And it sounds like there's some of that too.")

        if random.random() < 0.25:
            parts.append(soft)
            return " ".join(parts)

        parts.append(question)
        return " ".join(parts)

    def _handle_short_answer(self, topic, prev_text):
        prev_key = self._extract_key_phrases(prev_text)
        reflection = self._build_reflection(prev_text, prev_key, topic)
        soft = self._build_soft_ending(topic)
        short_prompts = [
            f"{reflection} You don't have to find the perfect words. Take your time.",
            f"It's okay not to know exactly what to say. {soft}",
            f"{reflection} We can sit here with it. {soft}",
            f"That's alright. Sometimes it's hard to put a name to what we feel. What's at the surface right now?",
        ]
        return random.choice(short_prompts)

    def goodbye(self):
        return (f"{self.user_name or 'Friend'}, thank you for talking with me. "
                "If things get difficult, please reach out to a counselor or crisis line. "
                "Take care.")


def typing_effect(text, delay=0.02):
    for char in text:
        print(char, end="", flush=True)
        time.sleep(delay)
    print()


def main():
    bot = Counselor()
    typing_effect(bot.greet())
    print()

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.lower() in ("quit", "exit", "bye", "goodbye"):
            typing_effect(bot.goodbye())
            break

        response = bot.respond(user_input)
        typing_effect(f"Bot: {response}")
        print()

    sys.exit(0)


if __name__ == "__main__":
    main()
