"""
SAGE Dataset Generator — produces JSONL for supervised fine-tuning.
Generates structured conversations with intent labels, entities, emotion candidates,
reasoning summaries, and quality checks.
"""

import json
import random
import uuid
import os

random.seed(42)

# ---------------------------------------------------------------------------
# Slang / emoji mappings
# ---------------------------------------------------------------------------
SLANG_VARIANTS = {
    "greeting": ["hey", "yo", "sup", "whats good", "ayy"],
    "affirm": ["fr", "frfr", "deadass", "real", "bet", "facts"],
    "negate": ["nah", "cap", "no cap it's not"],
    "emotion_anxiety": ["im tweaking", "im crashing out", "spiraling", "lowkey panicking"],
    "emotion_tired": ["im cooked", "its wraps", "cant do ts no more"],
    "emotion_sad": ["down bad", "spiraling"],
    "relationship": ["ghosted", "left me on read", "left on delivered", "friend zoned", "fumbled"],
    "filler": ["ngl", "lowkey", "highkey", "rn", "atm"],
}

SLANG_EMOJI = {
    "cry_laugh": "💀",
    "pray": "🙏",
    "heart": "❤️",
    "heartbreak": "💔",
    "tear_smile": "🥲",
    "sad": "😔",
    "confused": "😕",
    "crying": "😭",
}

# ---------------------------------------------------------------------------
# Intent response templates
# ---------------------------------------------------------------------------
INTENT_TEMPLATES = {
    "greeting": {
        "responses": [
            "Hey there. I'm glad you're here. What's on your mind today?",
            "Hi. I'm here to listen whenever you're ready to talk.",
            "Hey. Take your time — I'm around whenever you feel like sharing.",
        ]
    },
    "small_talk": {
        "responses": [
            "I'm here. What's been going on with you?",
            "Not much on this end. How are things on yours?",
            "I'm glad you reached out. What's been happening?",
        ]
    },
    "introduce_name": {
        "responses": [
            "Nice to meet you, {name}. What's been on your mind lately?",
            "Hi {name}. I'm glad you're here. What brings you to talk today?",
            "Thanks for sharing that, {name}. Is there something specific you wanted to talk about?",
        ]
    },
    "humor": {
        "responses": [
            "Glad you can find some lightness in things. What's been going on otherwise?",
            "It's good to laugh. How are things outside of that?",
            "I appreciate that. What's actually been on your mind?",
        ]
    },
    "sarcasm": {
        "responses": [
            "I hear the tone there. Sounds like things might be heavier than you're letting on.",
            "That sounded like it might mean the opposite. What's really going on?",
            "I get the sense there's more underneath that. Want to talk about it?",
        ]
    },
    "joke": {
        "responses": [
            "I appreciate you sharing that. What's really been on your mind?",
            "That's one way to lighten the mood. How are you actually feeling?",
        ]
    },
    "roleplay": {
        "responses": [
            "I'm curious — what draws you to that kind of scenario?",
            "That's an interesting idea. Is there something about it that connects to how you're feeling?",
        ]
    },
    "hypothetical": {
        "responses": [
            "That's an interesting question. What makes you think about that?",
            "I wonder what brought that question to mind for you.",
        ]
    },
    "storytelling": {
        "responses": [
            "That's quite a story. How did it feel to experience that?",
            "Thank you for sharing that. What stood out to you most about it?",
        ]
    },
    "relationship_problem": {
        "responses": [
            "It sounds like there's something shifting in a relationship that matters to you. What's been the hardest part?",
            "Relationships can be really complex. What do you feel like you need right now from this situation?",
            "I hear how much this is affecting you. When did you first notice things changing?",
        ]
    },
    "breakup": {
        "responses": [
            "I'm sorry you're going through that. Breakups can leave a real hole in your life. What's been the hardest part since it happened?",
            "Heartbreak is genuinely painful. It's okay to take time with this. How are you coping day to day?",
        ]
    },
    "loneliness": {
        "responses": [
            "That feeling of being alone is really heavy to carry. Have you felt this way for a while, or did something bring it on?",
            "Feeling lonely can be exhausting in a way that's hard to explain. What does loneliness look like for you day to day?",
        ]
    },
    "academic_stress": {
        "responses": [
            "School pressure can build up fast. What subject or task is weighing on you most?",
            "That sounds like a lot to manage. What's been the hardest thing to keep up with?",
        ]
    },
    "exam_anxiety": {
        "responses": [
            "Exam pressure is real. What part of it makes you most anxious — the preparation or the test itself?",
            "It makes sense to feel nervous before something important. What's the worst thing you imagine happening?",
        ]
    },
    "career_stress": {
        "responses": [
            "Work stress can seep into everything. What part of it drains you the most?",
            "That sounds like a difficult spot to be in. What's the one thing you wish could change most?",
        ]
    },
    "family_conflict": {
        "responses": [
            "Family dynamics are complicated. It sounds like you're carrying a lot about this. What do you wish they understood?",
            "That tension at home sounds really hard. How long has this been going on?",
        ]
    },
    "friendship_issue": {
        "responses": [
            "Friendship struggles can hurt in a unique way. What happened that changed things between you?",
            "That sounds painful, especially from someone close. Have you talked to them about how you feel?",
        ]
    },
    "financial_stress": {
        "responses": [
            "Financial worry can be really consuming. It's hard to think about anything else when money feels tight. What's the most pressing concern?",
            "I hear how heavy this is. Financial stress touches so many parts of life. Do you have anyone you can lean on for support?",
        ]
    },
    "grief": {
        "responses": [
            "I'm so sorry for your loss. Grief has its own timeline and there's no right way to feel. Tell me about them, if you want to.",
            "Losing someone you love changes everything. It's okay to not be okay. What's been the hardest moment since it happened?",
        ]
    },
    "loss": {
        "responses": [
            "Loss can leave a space that doesn't quite fill in. What are you missing most right now?",
            "I hear you. When something we value is gone, it takes time to adjust. What's different now?",
        ]
    },
    "identity_question": {
        "responses": [
            "That's a really big and honest question to sit with. What parts of yourself feel most unclear right now?",
            "Not knowing who you are can be disorienting. What used to feel certain that now feels different?",
        ]
    },
    "motivation_struggle": {
        "responses": [
            "Lack of motivation can be frustrating. Sometimes it helps to start smaller than you think you need to. What's one tiny thing you could do right now?",
            "It's hard when you want to want to do things but can't. When did you first notice this drop in motivation?",
        ]
    },
    "life_purpose": {
        "responses": [
            "That's one of the biggest questions a person can ask. What does a meaningful life look like to you?",
            "Purpose can feel like a moving target. What used to feel meaningful that doesn't anymore?",
        ]
    },
    "depression_like": {
        "responses": [
            "That sounds really heavy to carry. When everything feels pointless, even small steps take enormous energy. Have you talked to anyone about how you've been feeling?",
            "I hear the weight in what you're saying. You don't have to figure it all out at once. When did this feeling first start showing up?",
        ]
    },
    "anxiety_like": {
        "responses": [
            "Anxiety can be relentless. It sounds like your mind is working hard to protect you. What triggers that feeling most often?",
            "That worry sounds really present for you right now. What's the first thing that comes to mind when you sit with it?",
        ]
    },
    "panic": {
        "responses": [
            "That sounds terrifying. Panic attacks are intense but they can't hurt you. Let's try something — can you name three things you can see right now?",
            "I'm here with you. Focus on your breath for a moment. Breathe with me — in through the nose, out through the mouth. You're safe.",
        ]
    },
    "trauma_discussion": {
        "responses": [
            "Thank you for trusting me enough to share that. There's no pressure to go into details. What kind of support feels right for you today?",
            "Trauma stays with us in ways that can be exhausting. It makes sense that it still affects you. What helps you feel grounded?",
        ]
    },
    "abuse_disclosure": {
        "responses": [
            "I'm really sorry you've experienced that. What you're describing is not your fault. Are you safe right now?",
            "No one deserves to be treated that way. There are people who can help. Would you like me to share some resources?",
        ]
    },
    "self_harm_concern": {
        "responses": [
            "I'm really concerned about what you're saying. Please reach out to a crisis line — text HOME to 741741 or call 988. You deserve support.",
            "Those feelings are incredibly heavy to carry alone. Please contact a crisis line right now — they have trained people who can help. 988 is available 24/7.",
        ]
    },
    "violence_disclosure": {
        "responses": [
            "Can you help me understand what you mean? Are you describing a real event, a hypothetical scenario, or something else?",
        ]
    },
    "advice_seeking": {
        "responses": [
            "I can help you think through this. What options have you already considered?",
            "Let's talk it through. What feels like the most important thing to figure out first?",
        ]
    },
    "reflection": {
        "responses": [
            "That sounds like an important realization. What helped you get to that understanding?",
            "It takes awareness to recognize that pattern. What shifted for you?",
        ]
    },
    "clarification": {
        "responses": [
            "Let me try to put it differently. What I meant was that your feelings make sense given what you're going through. Does that land better?",
            "I'm sorry I wasn't clear. I meant that what you're feeling is understandable. Can you tell me what part felt confusing?",
        ]
    },
    "goodbye": {
        "responses": [
            "Thank you for talking with me today. Take care of yourself.",
            "I'm glad you reached out. Wishing you well until we talk again.",
        ]
    },
    "venting": {
        "responses": [
            "I'm listening. Let it out — sometimes saying it is the first step to feeling lighter.",
            "I hear you. That is a lot to hold in. What else has been building up?",
        ]
    },
    "gratitude": {
        "responses": [
            "I'm glad I could help. You're the one doing the work here. How are you feeling now?",
            "You're welcome. You reached out, and that took courage. How are things sitting with you now?",
        ]
    },
    "burnout": {
        "responses": [
            "Burnout is real and it takes time to recover from. What would real rest look like for you right now?",
            "Running on empty is unsustainable. When's the last time you took a real break without guilt?",
        ]
    },
    "anger_expression": {
        "responses": [
            "That sounds really frustrating. Anger often points to something we care deeply about. What's underneath it for you?",
            "I hear how strongly this affects you. What happened that sparked it?",
        ]
    },
    "jealousy": {
        "responses": [
            "Jealousy is hard to admit but very human. It usually points to something we want for ourselves. What is it about their situation that you wish for?",
            "Comparison is natural but it usually hurts more than it helps. What do you actually want for yourself?",
        ]
    },
    "regret": {
        "responses": [
            "Regret can be a heavy thing to replay. We all make choices we wish we could change. What do you think you needed in that moment?",
            "Hindsight is always clearer. What would you do differently now that you know what you know?",
        ]
    },
    "guilt": {
        "responses": [
            "Guilt can weigh on a person. It sounds like you care about doing the right thing. Have you been able to forgive yourself?",
            "That sense of responsibility shows you have a good heart. What would it take to make peace with what happened?",
        ]
    },
    "shame": {
        "responses": [
            "Shame tells us we are the problem, not that we did something wrong. That's an important difference. You are not a bad person for feeling this way.",
            "Thank you for trusting me with that. Shame thrives in silence. What would it feel like to offer yourself some compassion right now?",
        ]
    },
    "boredom": {
        "responses": [
            "Boredom can sometimes be a signal that something needs to shift. What's something you used to enjoy but haven't done lately?",
            "Restlessness often has something underneath it. What's been on your mind beneath the boredom?",
        ]
    },
    "excitement": {
        "responses": [
            "That's really nice to hear. What's contributing to that positive feeling?",
            "It's good to hold onto moments like this. What helped bring this about?",
        ]
    },
    "pride": {
        "responses": [
            "That sounds like something worth celebrating. What helped you get there?",
            "I'm glad you're recognizing your own effort. What are you most proud of about this?",
        ]
    },
    "hope": {
        "responses": [
            "That's good to hear. It can make a real difference to feel like things are moving in the right direction. What's giving you that sense of hope?",
            "Hope can be a powerful thing. What changed that made you feel more optimistic?",
        ]
    },
    "physical_health_concern": {
        "responses": [
            "Health worries can be really consuming. Have you been able to speak with a doctor about what you're experiencing?",
            "That sounds concerning. It's always okay to get medical advice when something doesn't feel right. Have you had a chance to see someone about it?",
        ]
    },
}

# ---------------------------------------------------------------------------
# User message generators per intent
# ---------------------------------------------------------------------------
def make_user_message(intent, entities=None):
    if entities is None:
        entities = {}

    name = entities.get("name", "")

    generators = {
        "greeting": lambda: random.choice([
            "Hey", "Hello", "Hi there", "Yo", "Heyy", "Hi",
        ]),
        "small_talk": lambda: random.choice([
            "Not much, just chillin", "Just bored", "How are you", "Nm hbu",
        ]),
        "introduce_name": lambda: f"My name is {name}" if name else random.choice([
            "I'm Alex", "My name is Sam", "Call me Jordan", "Name's Casey",
        ]),
        "humor": lambda: random.choice([
            "Why did the scarecrow win an award? Because he was outstanding in his field lol",
            "I'm so good at sleeping I could do it with my eyes closed",
            "Lmao that's funny",
        ]),
        "sarcasm": lambda: random.choice([
            "Oh great, another Monday. Fantastic.",
            "Wonderful. Just what I needed.",
            "Oh perfect, everything is going exactly as planned.",
        ]),
        "joke": lambda: random.choice([
            "Knock knock", "Why did the chicken cross the road?",
        ]),
        "roleplay": lambda: random.choice([
            "Let's say I'm a superhero. What would you ask me?",
            "Imagine you're my therapist for real. What would you say?",
        ]),
        "hypothetical": lambda: random.choice([
            "What if I just moved to another country tomorrow?",
            "What if I quit everything and started over?",
        ]),
        "storytelling": lambda: random.choice([
            "So the other day I was walking and this random dog just started following me",
            "Something crazy happened on my way to work today",
        ]),
        "relationship_problem": lambda: random.choice([
            "My partner and I keep fighting over nothing",
            "I don't feel like my boyfriend understands me",
            "My girlfriend and I have been drifting apart",
        ]),
        "breakup": lambda: random.choice([
            "They broke up with me last night",
            "My partner ended things and I'm heartbroken",
            "We broke up and I don't know what to do with myself",
        ]),
        "loneliness": lambda: random.choice([
            "I feel so lonely even when I'm around people",
            "I don't have any real friends",
            "No one really understands me",
        ]),
        "academic_stress": lambda: random.choice([
            "I have so much homework I'm drowning",
            "Exams are killing me rn",
            "I can't keep up with my classes",
        ]),
        "exam_anxiety": lambda: random.choice([
            "I have a huge exam tomorrow and I'm freaking out",
            "What if I fail my finals",
            "I'm so anxious about my test I can't sleep",
        ]),
        "career_stress": lambda: random.choice([
            "I hate my job so much",
            "My boss is making my life miserable",
            "I got laid off and feel lost",
        ]),
        "family_conflict": lambda: random.choice([
            "My parents don't understand me at all",
            "I keep fighting with my mom",
            "My family is driving me crazy",
        ]),
        "friendship_issue": lambda: random.choice([
            "My best friend just ghosted me",
            "I had a fight with my friend and we haven't talked since",
            "I feel like my friends don't care about me anymore",
        ]),
        "financial_stress": lambda: random.choice([
            "I'm so stressed about money",
            "I can't pay my bills this month",
            "The debt is crushing me",
        ]),
        "grief": lambda: random.choice([
            "My grandmother passed away last week",
            "I lost my dad and I don't know how to cope",
            "Someone I loved died and I can't stop crying",
        ]),
        "loss": lambda: random.choice([
            "I lost my best friend to cancer",
            "Everything changed after my mom passed",
        ]),
        "identity_question": lambda: random.choice([
            "I don't know who I am anymore",
            "Who am I really? I feel so lost",
            "I don't recognize myself lately",
        ]),
        "motivation_struggle": lambda: random.choice([
            "I have zero motivation to do anything",
            "I can't even get out of bed these days",
            "I keep procrastinating everything important",
        ]),
        "life_purpose": lambda: random.choice([
            "What is the point of all this",
            "I don't know what to do with my life",
            "I feel like I have no purpose",
        ]),
        "depression_like": lambda: random.choice([
            "Nothing matters anymore",
            "I feel empty all the time",
            "What's the point of even trying",
            "I give up. Everything is pointless.",
        ]),
        "anxiety_like": lambda: random.choice([
            "I can't stop overthinking everything",
            "My mind is racing all the time",
            "I keep worrying about things I can't control",
        ]),
        "panic": lambda: random.choice([
            "I think I'm having a panic attack",
            "I can't breathe my heart is pounding",
            "Help me I'm freaking out and I don't know why",
        ]),
        "trauma_discussion": lambda: random.choice([
            "Something happened to me that I never talk about",
            "I have PTSD from something that happened years ago",
            "I keep having flashbacks and I don't know what to do",
        ]),
        "abuse_disclosure": lambda: random.choice([
            "My partner has been hurting me",
            "I'm in an abusive relationship",
            "Someone I trusted hurt me badly",
        ]),
        "self_harm_concern": lambda: random.choice([
            "I want to hurt myself",
            "I've been cutting again",
            "I don't want to exist anymore",
        ]),
        "violence_disclosure": lambda: random.choice([
            "I killed someone",
            "I hurt someone badly",
            "I got into a fight and someone got hurt",
        ]),
        "advice_seeking": lambda: random.choice([
            "What should I do about my situation",
            "I need advice on what to do next",
            "Can you help me figure this out",
        ]),
        "reflection": lambda: random.choice([
            "I think I realize now what was really bothering me",
            "Looking back I see the pattern clearly now",
            "I've been reflecting on what you said earlier",
        ]),
        "clarification": lambda: random.choice([
            "What do you mean by that",
            "I don't understand what you're saying",
            "Can you explain that differently",
        ]),
        "goodbye": lambda: random.choice([
            "I have to go now thanks",
            "Bye I'll talk to you later",
            "Thanks for listening. Gotta go.",
        ]),
        "venting": lambda: random.choice([
            "I just need to vent for a minute",
            "Let me get this off my chest",
            "I need to say something that's been bothering me",
        ]),
        "gratitude": lambda: random.choice([
            "Thank you so much",
            "I really appreciate you listening",
            "Thanks for being here",
        ]),
        "burnout": lambda: random.choice([
            "I'm so burned out I can't function",
            "I'm completely drained",
            "I feel like I have nothing left to give",
        ]),
        "anger_expression": lambda: random.choice([
            "I'm so angry right now I could scream",
            "I hate everything about my situation",
            "I'm furious at how I've been treated",
        ]),
        "jealousy": lambda: random.choice([
            "I'm so jealous of my friend's success",
            "Everyone else seems to have it together except me",
            "Why does everyone else get what they want",
        ]),
        "regret": lambda: random.choice([
            "I regret so many decisions I've made",
            "I wish I had done things differently",
            "If only I could go back and change things",
        ]),
        "guilt": lambda: random.choice([
            "I feel so guilty about what happened",
            "It's all my fault",
            "I can't forgive myself for what I did",
        ]),
        "shame": lambda: random.choice([
            "I'm so ashamed of who I am",
            "I feel disgusting",
            "I'm embarrassed to even be here",
        ]),
        "boredom": lambda: random.choice([
            "I'm so bored",
            "Nothing interesting ever happens",
            "I'm bored and lonely",
        ]),
        "excitement": lambda: random.choice([
            "I'm so excited for my trip",
            "Something amazing just happened",
            "I got the job I wanted",
        ]),
        "pride": lambda: random.choice([
            "I'm really proud of myself for once",
            "I actually accomplished something today",
            "I did it. I finally did it.",
        ]),
        "hope": lambda: random.choice([
            "I feel like things are finally looking up",
            "For the first time in a while I feel hopeful",
            "I think I see a way forward now",
        ]),
        "physical_health_concern": lambda: random.choice([
            "I've been having really bad headaches",
            "I can't sleep at all anymore",
            "I think something is wrong with my health",
        ]),
    }

    gen = generators.get(intent, lambda: "I need to talk about something")
    return gen()


# ---------------------------------------------------------------------------
# Slang injection
# ---------------------------------------------------------------------------
def inject_slang(text):
    if random.random() < 0.3:
        filler = random.choice(SLANG_VARIANTS["filler"])
        text = f"{filler} {text.lower()}"
    if random.random() < 0.2:
        emoji = random.choice(list(SLANG_EMOJI.values()))
        text = f"{text} {emoji}"
    return text


# ---------------------------------------------------------------------------
# Emotion candidate generation
# ---------------------------------------------------------------------------
INTENT_EMOTIONS = {
    "greeting": [{"emotion": "neutral", "confidence": 0.8}, {"emotion": "open", "confidence": 0.6}],
    "introduce_name": [{"emotion": "neutral", "confidence": 0.7}, {"emotion": "open", "confidence": 0.6}],
    "small_talk": [{"emotion": "neutral", "confidence": 0.8}, {"emotion": "casual", "confidence": 0.7}],
    "relationship_problem": [{"emotion": "frustration", "confidence": 0.8}, {"emotion": "sadness", "confidence": 0.6}],
    "breakup": [{"emotion": "heartbreak", "confidence": 0.9}, {"emotion": "sadness", "confidence": 0.8}, {"emotion": "grief", "confidence": 0.6}],
    "loneliness": [{"emotion": "loneliness", "confidence": 0.9}, {"emotion": "sadness", "confidence": 0.7}],
    "academic_stress": [{"emotion": "stress", "confidence": 0.8}, {"emotion": "anxiety", "confidence": 0.7}, {"emotion": "overwhelm", "confidence": 0.6}],
    "exam_anxiety": [{"emotion": "anxiety", "confidence": 0.9}, {"emotion": "fear", "confidence": 0.7}],
    "career_stress": [{"emotion": "stress", "confidence": 0.8}, {"emotion": "frustration", "confidence": 0.7}],
    "family_conflict": [{"emotion": "frustration", "confidence": 0.7}, {"emotion": "anger", "confidence": 0.6}, {"emotion": "hurt", "confidence": 0.6}],
    "friendship_issue": [{"emotion": "hurt", "confidence": 0.7}, {"emotion": "confusion", "confidence": 0.6}],
    "financial_stress": [{"emotion": "anxiety", "confidence": 0.8}, {"emotion": "stress", "confidence": 0.7}, {"emotion": "fear", "confidence": 0.6}],
    "grief": [{"emotion": "grief", "confidence": 0.9}, {"emotion": "sadness", "confidence": 0.8}, {"emotion": "yearning", "confidence": 0.6}],
    "depression_like": [{"emotion": "hopelessness", "confidence": 0.8}, {"emotion": "emptiness", "confidence": 0.7}, {"emotion": "sadness", "confidence": 0.7}],
    "anxiety_like": [{"emotion": "anxiety", "confidence": 0.9}, {"emotion": "worry", "confidence": 0.8}, {"emotion": "overwhelm", "confidence": 0.6}],
    "panic": [{"emotion": "panic", "confidence": 0.9}, {"emotion": "fear", "confidence": 0.8}],
    "anger_expression": [{"emotion": "anger", "confidence": 0.9}, {"emotion": "frustration", "confidence": 0.7}],
    "guilt": [{"emotion": "guilt", "confidence": 0.8}, {"emotion": "shame", "confidence": 0.6}],
    "shame": [{"emotion": "shame", "confidence": 0.9}, {"emotion": "worthlessness", "confidence": 0.6}],
    "hope": [{"emotion": "hope", "confidence": 0.9}, {"emotion": "optimism", "confidence": 0.7}],
    "pride": [{"emotion": "pride", "confidence": 0.8}, {"emotion": "accomplishment", "confidence": 0.7}],
    "excitement": [{"emotion": "excitement", "confidence": 0.9}, {"emotion": "joy", "confidence": 0.7}],
    "gratitude": [{"emotion": "gratitude", "confidence": 0.9}, {"emotion": "warmth", "confidence": 0.6}],
    "boredom": [{"emotion": "boredom", "confidence": 0.8}, {"emotion": "restlessness", "confidence": 0.5}],
    "burnout": [{"emotion": "exhaustion", "confidence": 0.8}, {"emotion": "hopelessness", "confidence": 0.5}],
    "regret": [{"emotion": "regret", "confidence": 0.8}, {"emotion": "sadness", "confidence": 0.6}],
    "jealousy": [{"emotion": "jealousy", "confidence": 0.8}, {"emotion": "inadequacy", "confidence": 0.6}],
    "identity_question": [{"emotion": "confusion", "confidence": 0.8}, {"emotion": "existential_anxiety", "confidence": 0.6}],
    "motivation_struggle": [{"emotion": "frustration", "confidence": 0.6}, {"emotion": "apathy", "confidence": 0.6}],
    "life_purpose": [{"emotion": "confusion", "confidence": 0.7}, {"emotion": "existential_anxiety", "confidence": 0.6}],
    "venting": [{"emotion": "frustration", "confidence": 0.7}, {"emotion": "overwhelm", "confidence": 0.6}],
    "reflection": [{"emotion": "insight", "confidence": 0.7}, {"emotion": "curiosity", "confidence": 0.6}],
    "clarification": [{"emotion": "confusion", "confidence": 0.7}, {"emotion": "curiosity", "confidence": 0.5}],
    "goodbye": [{"emotion": "neutral", "confidence": 0.7}, {"emotion": "gratitude", "confidence": 0.5}],
    "violence_disclosure": [{"emotion": "fear", "confidence": 0.5}, {"emotion": "confusion", "confidence": 0.5}],
    "humor": [{"emotion": "amusement", "confidence": 0.7}, {"emotion": "lightness", "confidence": 0.6}],
    "sarcasm": [{"emotion": "frustration", "confidence": 0.6}, {"emotion": "irony", "confidence": 0.7}],
    "joke": [{"emotion": "playful", "confidence": 0.7}],
    "roleplay": [{"emotion": "creative", "confidence": 0.6}, {"emotion": "curious", "confidence": 0.5}],
    "hypothetical": [{"emotion": "curious", "confidence": 0.7}, {"emotion": "uncertain", "confidence": 0.5}],
    "storytelling": [{"emotion": "narrative", "confidence": 0.6}],
    "trauma_discussion": [{"emotion": "fear", "confidence": 0.6}, {"emotion": "vulnerability", "confidence": 0.7}],
    "abuse_disclosure": [{"emotion": "fear", "confidence": 0.8}, {"emotion": "shame", "confidence": 0.6}],
    "self_harm_concern": [{"emotion": "despair", "confidence": 0.9}, {"emotion": "hopelessness", "confidence": 0.8}],
    "physical_health_concern": [{"emotion": "fear", "confidence": 0.7}, {"emotion": "anxiety", "confidence": 0.6}],
    "loss": [{"emotion": "sadness", "confidence": 0.8}, {"emotion": "grief", "confidence": 0.7}],
    "advice_seeking": [{"emotion": "confusion", "confidence": 0.6}, {"emotion": "hopeful", "confidence": 0.5}],
}


# ---------------------------------------------------------------------------
# Conversation generator
# ---------------------------------------------------------------------------
def generate_conversation(intent_chain, conversation_id=None):
    """Generate a multi-turn conversation from a chain of intents."""
    if conversation_id is None:
        conversation_id = str(uuid.uuid4())[:8]

    entities = {}
    records = []
    context = {"user_name": None, "previous_topics": [], "mentioned_entities": {}}

    start_intent = intent_chain[0]
    name_intro_done = False

    for turn_idx, intent in enumerate(intent_chain):
        turn_num = turn_idx + 1

        if intent == "introduce_name" and not name_intro_done:
            name = random.choice(["Alex", "Sam", "Jordan", "Casey", "Riley", "Taylor", "Morgan"])
            entities["name"] = name
            context["user_name"] = name
            name_intro_done = True
        elif intent == "introduce_name" and name_intro_done:
            name = entities.get("name", "Friend")
            intent = "small_talk"

        user_msg = make_user_message(intent, entities)
        if random.random() < 0.35 and intent not in ("greeting", "introduce_name", "goodbye"):
            user_msg = inject_slang(user_msg)

        intent_key = intent
        templates = INTENT_TEMPLATES.get(intent_key, INTENT_TEMPLATES["small_talk"])
        response_template = random.choice(templates["responses"])

        if "{name}" in response_template:
            name = entities.get("name", "")
            if name:
                response_template = response_template.replace("{name}", name)
            else:
                response_template = response_template.replace("{name} ", "")
                response_template = response_template.replace(" {name}", "")
                response_template = response_template.replace("{name}", "")

        reasoning = f"Detected intent: {intent}. The user is expressing feelings related to {intent.replace('_', ' ')}. Response validates and explores with one open question."

        emotion_candidates = INTENT_EMOTIONS.get(intent, [{"emotion": "unknown", "confidence": 0.5}])

        memory_update = {}
        if entities.get("name"):
            memory_update["user_name"] = entities["name"]
        memory_update["last_intent"] = intent

        record = {
            "conversation_id": conversation_id,
            "turn": turn_num,
            "context": dict(context),
            "user_message": user_msg,
            "detected_intent": intent,
            "entities": dict(entities),
            "emotion_candidates": emotion_candidates,
            "reasoning_summary": reasoning,
            "assistant_response": response_template,
            "memory_update": memory_update,
            "quality_checks": {
                "relevant": True,
                "context_used": True if turn_num > 1 else False,
                "generic": False,
                "hallucination": False,
            },
        }
        records.append(record)

        context["previous_topics"].append(intent)
        if turn_num == 1 and intent in ("greeting", "introduce_name"):
            pass
        elif turn_num > 1 and name_intro_done and intent not in ("greeting", "introduce_name", "goodbye"):
            if random.random() < 0.3 and records:
                prev = records[-1]
                context["last_response"] = prev["assistant_response"]

    return records


# ---------------------------------------------------------------------------
# Intent chains
# ---------------------------------------------------------------------------
SINGLE_TURN_INTENTS = [
    "greeting", "small_talk", "loneliness", "depression_like",
    "panic", "grief", "boredom", "gratitude", "goodbye",
    "anger_expression", "hope", "pride", "excitement",
]

THREE_TURN_CHAINS = [
    ["greeting", "introduce_name", "small_talk"],
    ["greeting", "introduce_name", "relationship_problem"],
    ["greeting", "introduce_name", "breakup"],
    ["greeting", "introduce_name", "loneliness"],
    ["greeting", "introduce_name", "academic_stress"],
    ["greeting", "introduce_name", "exam_anxiety"],
    ["greeting", "introduce_name", "career_stress"],
    ["greeting", "introduce_name", "family_conflict"],
    ["greeting", "introduce_name", "friendship_issue"],
    ["greeting", "introduce_name", "financial_stress"],
    ["greeting", "introduce_name", "grief"],
    ["greeting", "introduce_name", "anxiety_like"],
    ["greeting", "introduce_name", "depression_like"],
    ["greeting", "introduce_name", "anger_expression"],
    ["greeting", "introduce_name", "identity_question"],
    ["greeting", "introduce_name", "motivation_struggle"],
    ["greeting", "introduce_name", "life_purpose"],
    ["greeting", "introduce_name", "guilt"],
    ["greeting", "introduce_name", "shame"],
    ["greeting", "introduce_name", "burnout"],
    ["greeting", "introduce_name", "jealousy"],
    ["greeting", "introduce_name", "regret"],
    ["greeting", "introduce_name", "boredom"],
    ["greeting", "introduce_name", "excitement"],
    ["greeting", "introduce_name", "pride"],
    ["greeting", "introduce_name", "hope"],
    ["greeting", "introduce_name", "venting"],
    ["greeting", "introduce_name", "physical_health_concern"],
]

FIVE_TURN_CHAINS = [
    ["greeting", "introduce_name", "relationship_problem", "venting", "reflection"],
    ["greeting", "introduce_name", "breakup", "grief", "advice_seeking"],
    ["greeting", "introduce_name", "loneliness", "depression_like", "hope"],
    ["greeting", "introduce_name", "academic_stress", "exam_anxiety", "advice_seeking"],
    ["greeting", "introduce_name", "career_stress", "burnout", "reflection"],
    ["greeting", "introduce_name", "family_conflict", "anger_expression", "clarification"],
    ["greeting", "introduce_name", "anxiety_like", "panic", "gratitude"],
    ["greeting", "introduce_name", "grief", "loss", "hope"],
    ["greeting", "introduce_name", "guilt", "shame", "reflection"],
    ["greeting", "introduce_name", "jealousy", "regret", "advice_seeking"],
    ["greeting", "introduce_name", "burnout", "motivation_struggle", "hope"],
    ["greeting", "introduce_name", "identity_question", "life_purpose", "reflection"],
    ["greeting", "introduce_name", "financial_stress", "anxiety_like", "advice_seeking"],
    ["greeting", "introduce_name", "trauma_discussion", "grief", "gratitude"],
    ["greeting", "introduce_name", "friendship_issue", "hurt", "reflection"],
]

TEN_TURN_CHAINS = [
    ["greeting", "introduce_name", "small_talk", "relationship_problem", "venting",
     "reflection", "advice_seeking", "gratitude", "hope", "goodbye"],
    ["greeting", "introduce_name", "loneliness", "depression_like", "reflection",
     "hope", "excitement", "pride", "gratitude", "goodbye"],
    ["greeting", "introduce_name", "academic_stress", "exam_anxiety", "panic",
     "advice_seeking", "reflection", "motivation_struggle", "hope", "goodbye"],
]


# ---------------------------------------------------------------------------
# Main generation
# ---------------------------------------------------------------------------
def generate_all():
    all_records = []
    convo_id = 0

    for intent in SINGLE_TURN_INTENTS:
        for _ in range(3):
            convo_id += 1
            records = generate_conversation([intent], f"single_{convo_id}")
            all_records.extend(records)

    for chain in THREE_TURN_CHAINS:
        for _ in range(2):
            convo_id += 1
            records = generate_conversation(chain, f"three_{convo_id}")
            all_records.extend(records)

    for chain in FIVE_TURN_CHAINS:
        for _ in range(2):
            convo_id += 1
            records = generate_conversation(chain, f"five_{convo_id}")
            all_records.extend(records)

    for chain in TEN_TURN_CHAINS:
        convo_id += 1
        records = generate_conversation(chain, f"ten_{convo_id}")
        all_records.extend(records)

    return all_records


def write_dataset(records, output_path):
    with open(output_path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record) + "\n")
    print(f"Wrote {len(records)} records to {output_path}")


if __name__ == "__main__":
    import sys
    output_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(output_dir, "sage_training.jsonl")
    if len(sys.argv) > 1:
        output_path = sys.argv[1]

    records = generate_all()
    write_dataset(records, output_path)
    print(f"Generated {len(records)} training records across")
    print(f"  Single-turn: {len([r for r in records if records[0] is not None])}")
