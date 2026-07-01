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

TOPIC_RESPONSES = {
    "anxiety": [
        "Anxiety can feel really heavy. What physical sensations do you notice when it shows up?",
        "Let's try something: take a slow breath in for 4 counts, hold for 4, out for 4. What do you notice?",
        "Anxiety often comes from trying to control the future. What's one small thing you can control right now?",
        "It sounds like your mind is stuck in a worry loop. What helps you press pause on those thoughts?",
        "Anxiety wants you to believe the worst will happen. What's a more balanced way to look at this?",
        "I hear how hard this is. When you feel anxious, where do you feel it most in your body?",
        "You don't have to fight anxiety alone. What would support look like for you right now?",
        "Sometimes naming your anxiety helps. If it had a shape or color, what would it be?",
        "What's one tiny thing you can do right now that feels manageable?",
        "Your anxiety is valid, but it's also lying to you about how dangerous this is.",
        "Let's try grounding: name 5 things you can see, 4 you can touch, 3 you can hear. What do you notice?",
        "Anxiety thrives on avoidance. What's one small step you could take toward what scares you?",
        "What would you tell a friend who was feeling this same anxiety?",
        "I see how much this is affecting you. Have you tried progressive muscle relaxation?",
        "Panic attacks are terrifying, but they can't hurt you. They pass. What helps you ride the wave?",
        "Your nervous system is trying to protect you, but you're not in danger right now.",
    ],
    "sadness": [
        "It's okay to be sad. Sadness isn't weakness — it's a sign you care about something. What started this feeling?",
        "I'm sorry you're going through that. Have you been able to let yourself feel it, or are you pushing it away?",
        "Sadness can be your mind's way of telling you something matters. What do you think it's trying to say?",
        "It takes courage to sit with sadness. What do you need most right now?",
        "You don't have to fix it today. Just being here with it is enough.",
        "Sadness has a way of making everything feel heavier. What's one comfort you can give yourself?",
        "I see your pain. You don't have to go through it alone.",
        "What would it look like to be gentle with yourself right now?",
        "Sadness isn't permanent, even though it feels endless. What helped you feel better in the past?",
        "Your feelings are valid. There's no timeline for healing.",
        "Grief is love with nowhere to go. What did this person or thing mean to you?",
        "When we lose something important, it's natural to feel empty. What would honoring that loss look like?",
        "Sometimes the heaviest thing is pretending to be okay. You don't have to do that here.",
        "Have you tried behavioral activation? Even a small activity like a short walk can shift things.",
        "I hear your hopelessness. When you've felt this low before, what got you through?",
        "It's exhausting to carry this much sadness. What's one tiny act of self-compassion you could try?",
    ],
    "anger": [
        "Anger is often a protector for deeper feelings like hurt or fear. What's underneath it?",
        "It makes sense you'd feel angry about that. What would you need to feel heard?",
        "Anger has a lot of energy. Before you act on it, what does it need you to understand?",
        "You have every right to be angry. The question is: what do you want to do with that anger?",
        "Sometimes anger is just grief in disguise. Is there something you're mourning?",
        "I can feel how passionate you are about this. That matters.",
        "What would a healthy release for this anger look like?",
        "Anger can be a signal that a boundary has been crossed. What boundary might need attention?",
        "Let's sit with the anger for a moment. What does it need you to know?",
        "You're not bad for being angry. What happened that made you feel this way?",
        "Try naming the temperature of your anger on a scale of 1-10. What would bring it down by 1 point?",
        "Anger wants action. What's a constructive action you can take with this energy?",
        "It sounds like you've been holding this in for a long time. That must be exhausting.",
        "What would it look like to set a firm boundary around what's making you angry?",
    ],
    "stress": [
        "Stress usually means you're carrying a lot. What's one thing you could set down, even for today?",
        "When you're stressed, your nervous system is working overtime. What helps you feel safe?",
        "You can't pour from an empty cup. What's one small act of care you can give yourself?",
        "Let's break it down. What's the single most important thing on your plate right now?",
        "Your best is enough, even if it doesn't feel like it.",
        "What would it feel like to give yourself permission to rest?",
        "I hear how overwhelmed you are. What can wait until tomorrow?",
        "Stress thrives on urgency. Take a breath — what actually needs to happen right this second?",
        "You're human, not a machine. It's okay to be at your limit.",
        "What's one boundary you can set to protect your peace?",
        "Try the 5-4-3-2-1 grounding technique: name 5 things you see, 4 you feel, 3 you hear, 2 you smell, 1 you taste.",
        "Stress often comes from trying to control too much. What can you actually control right now?",
        "What would you tell a friend who was carrying the same load?",
        "Let's try a quick body scan - what part of your body is holding the most tension right now?",
        "You've been running on empty. What would refuel you even 5 percent?",
    ],
    "sleep": [
        "Sleep struggles often reflect what's happening in our waking hours. What's on your mind when you lie down?",
        "A consistent wind-down routine can help. What does your evening look like right now?",
        "It's hard to rest when your mind is racing. Would writing down your thoughts before bed help?",
        "Your body wants to sleep — it might be your mind keeping you awake. What's it trying to process?",
        "Have you tried creating a sleep sanctuary? Dark, cool, quiet — what would that look like for you?",
        "I know how frustrating sleeplessness is. What have you tried so far?",
        "Sometimes the pressure to sleep makes it worse. What if you just rested without trying to sleep?",
        "Our brains process the day at night. Is there something you haven't had a chance to process?",
        "A warm drink, dim lights, no screens — small rituals can signal safety to your brain.",
        "Sleep will come. For now, let's focus on calming your nervous system.",
        "CBT for insomnia suggests getting out of bed if you can't sleep for 20 minutes. Have you tried that?",
        "What's your caffeine and screen time like before bed? Small adjustments can make a big difference.",
        "Sleep anxiety is real - the fear of not sleeping keeps you awake. What if you took the pressure off?",
    ],
    "relationships": [
        "Relationships can be beautiful and hard. What do you need from this person that you're not getting?",
        "It sounds like this relationship matters to you. What would a healthy version of it look like?",
        "Communication is tough. What would you say if you knew they'd really listen?",
        "You deserve to feel safe and valued in your relationships. Are you feeling that?",
        "What's one small step you could take to improve things?",
        "Relationships require two people trying. Are you both showing up?",
        "It's okay to outgrow relationships. What does your heart tell you?",
        "Boundaries aren't walls — they're guidelines for healthy connection. Do you have the boundaries you need?",
        "I hear how much this is hurting you. What would you need to feel at peace?",
        "You can't control the other person, but you can control how you show up. What feels right for you?",
        "Breakups hurt because we're not just losing a person, we're losing the future we imagined.",
        "Trust takes years to build and seconds to break. What would rebuilding trust look like for you?",
        "It sounds like you've been carrying this relationship alone. That's not fair to you.",
        "What would your ideal resolution look like? Sometimes naming it helps clarify what you want.",
        "Heartbreak is a form of grief. It deserves the same compassion as any other loss.",
    ],
    "work": [
        "Work takes up so much of our lives. What part of it drains you the most?",
        "It's tough when work feels overwhelming. What's one boundary you could set this week?",
        "Your worth is not your productivity. What would you do if you weren't afraid of failing?",
        "It sounds like there's a mismatch somewhere. Is it the work, the environment, or something else?",
        "What would your ideal work life look like? Even if it feels far away.",
        "Burnout is real. What's one thing you can do to protect your energy?",
        "You spend a third of your life working. It's okay to want more from it.",
        "What skills or passions are you not using that you wish you could?",
        "A job is what you do, not who you are. Remembering that can take some pressure off.",
        "What would it take to feel fulfilled in your work?",
        "Imposter syndrome is incredibly common. You're not alone in feeling this way.",
        "Being laid off says nothing about your worth as a person. The job market is brutal right now.",
        "What parts of your work used to bring you joy? Can you reconnect with any of them?",
        "Career confusion is normal, especially in your 20s and 30s. You don't need to have it all figured out.",
    ],
    "confusion": [
        "Not knowing is uncomfortable, but it's also honest. What would bring you clarity?",
        "Sometimes confusion means you're growing. What options are you weighing?",
        "You don't need to have it all figured out. What's one thing you're sure about?",
        "Confusion is a sign you're at a threshold. What feels most important right now?",
        "Let's untangle this together. What's the heart of what you're unsure about?",
        "It's okay not to know. The answers often come when we stop forcing them.",
        "What would you do if you weren't afraid of making the wrong choice?",
        "Sometimes writing down both sides helps. Want to talk through the options?",
        "You don't have to decide today. Give yourself the space to sit with the question.",
        "What does your gut say, even if your mind is complicated?",
        "Identity crises are a normal part of growth. You're not broken - you're evolving.",
        "What values matter most to you? Sometimes starting with values helps clarify decisions.",
        "It's okay to change your mind. You're allowed to grow and shift.",
        "Think about your past self - what would they tell you? What would your future self want you to know?",
    ],
    "positive": [
        "That's wonderful to hear! What's contributing to that positive feeling?",
        "It's great that you're noticing the good. How can you nurture that?",
        "Savor this moment. You deserve to feel good. What does it feel like in your body?",
        "I love that for you. What helped you get to this place?",
        "Holding onto positive moments is like building a resilience bank. What are you most grateful for?",
        "That's beautiful. How can you carry this feeling with you through harder days?",
        "You're doing something right. What's working well in your life?",
        "Let's celebrate that. You've earned it.",
        "What would it look like to invite more of this into your life?",
        "Joy is worth protecting. What helps you stay connected to this feeling?",
        "Gratitude isn't about ignoring the hard stuff - it's about noticing the good alongside it.",
        "What's one thing you did today that future you will be grateful for?",
        "Positive moments build resilience. Store this one up for the tough days.",
    ],
    "general": [
        "Tell me more about that. How does it make you feel?",
        "I hear you. Can you share what's been going on?",
        "That sounds important to you. What comes to mind when you think about it?",
        "I'm listening. What do you wish was different?",
        "Thank you for sharing that with me. How does it feel to talk about it?",
        "I want to understand better. What's the hardest part?",
        "That's really brave of you to open up. What do you need right now?",
        "I appreciate you trusting me with this. What's been on your heart?",
        "There's no pressure to get it perfect. Just share what feels true.",
        "You're not alone in this. What feels most important to talk about?",
        "Take your time. I'm here and I'm listening.",
        "What would you say if there were no wrong answers?",
        "Sometimes just saying things out loud helps. I'm here to listen.",
        "How long have you been feeling this way?",
        "What do you think triggered these feelings?",
        "What's the biggest thing on your mind right now?",
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
        return ("Hi, I'm here to listen. I'm an AI counselor — I can help you work through "
                "what's on your mind, but I'm not a replacement for a licensed professional. "
                "What's your name?")

    def respond(self, user_input):
        text = user_input.strip()
        if not text:
            return "Take your time. I'm here whenever you're ready."

        self.history.append((USER, text))

        if detect_crisis(text):
            return ("I'm really concerned about what you're saying. Please reach out to a crisis line now:\n"
                    "- National Suicide Prevention Lifeline: 988\n"
                    "- Crisis Text Line: Text HOME to 741741\n"
                    "You matter, and there are people who want to help.")

        if self.user_name is None:
            words = text.split()
            if len(words) <= 3 and not self._is_counseling_content(text):
                self.user_name = text.split()[0].strip(",.!?")
                self.session_count += 1
                return f"Nice to meet you, {self.user_name}. What's been on your mind lately?"
            self.user_name = "Friend"
            self.session_count += 1

        return self._generate(text)

    def _generate(self, text):
        topic, confidence = self._predict_topic(text)

        if topic == "general" and confidence < 0.3:
            return random.choice(TOPIC_RESPONSES["general"])

        strategies = self.kb.get_coping_strategies(topic)
        insight = self.kb.get_language_insight(text)

        pool = TOPIC_RESPONSES.get(topic, TOPIC_RESPONSES["general"])
        chosen = random.choice(pool)

        if insight and random.random() < 0.15:
            follow_up = insight.get("follow_up", "")
            if follow_up:
                return f"{chosen}\n\n{follow_up}"

        if strategies and random.random() < 0.2:
            strat = random.choice(strategies)
            return f"{chosen}\n\nOne thing that may help: {strat.get('instructions', '')[:200]}"

        if confidence > 0.7:
            tag = f"[{topic}] "
        else:
            tag = ""
        return tag + chosen

    def goodbye(self):
        return (f"{self.user_name or 'Friend'}, thank you for trusting me today. "
                "Remember: you don't have to figure everything out at once. "
                "Be kind to yourself. And if things get heavy, please reach out to a real counselor or crisis line. "
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
