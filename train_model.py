import numpy as np
import pandas as pd
import re
import urllib.request
import tempfile
import os
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.pipeline import Pipeline
from sklearn.utils.class_weight import compute_class_weight
import joblib

SEED = 42
np.random.seed(SEED)

CATEGORIES = [
    "anxiety", "sadness", "anger", "stress",
    "sleep", "relationships", "work", "confusion",
    "positive", "general"
]

LABELED_DATA = {
    "anxiety": [
        "I feel anxious all the time",
        "My heart is racing and I can't calm down",
        "I keep worrying about things I can't control",
        "Panic attacks are ruining my life",
        "I feel nervous about everything",
        "I can't stop overthinking every little thing",
        "My chest feels tight and I'm scared",
        "I have this constant feeling of dread",
        "I'm afraid something bad is going to happen",
        "Social situations make me extremely anxious",
        "I feel overwhelmed by fear",
        "My mind won't stop racing at night",
        "I feel restless and on edge constantly",
        "I'm terrified of failing",
        "I worry so much it makes me sick",
        "I can't breathe when I get anxious",
        "My hands are shaking and I don't know why",
        "I feel like I'm losing control",
        "I avoid everything that makes me nervous",
        "I feel jumpy and easily startled",
        "I can't stop imagining worst case scenarios",
        "My anxiety is getting worse every day",
        "I feel like something terrible is about to happen",
        "I obsess over what people think of me",
        "I feel panicked for no reason",
        "I'm scared of having another panic attack",
        "My stomach is in knots from worrying",
        "I keep catastrophizing everything",
        "I feel like I'm constantly on high alert",
        "I'm afraid to leave my house",
        "I ruminate about the same things over and over",
        "I feel paralyzed by fear",
        "My anxiety makes it hard to function",
        "I can't stop my anxious thoughts",
        "I feel like everyone is judging me",
        "I get anxious just thinking about the future",
        "I'm scared to try new things",
        "I feel like I'm suffocating from worry",
        "I need reassurance constantly",
        "I can't relax no matter what I do",
        "I keep thinking I'm going to die",
        "I'm scared of my own thoughts",
        "I feel like I'm going crazy",
        "Every little noise startles me",
        "I feel like my heart will explode",
        "I'm terrified of being alone",
        "I can't stop pacing when I'm anxious",
        "I feel sweaty and dizzy with worry",
        "My mind is a mess of anxious thoughts",
        "I feel trapped by my own fear",
        "I worry about things that don't matter",
        "I can't control my anxiety anymore",
        "I feel like crying all the time from stress",
        "I'm scared to speak up in meetings",
        "I obsess over my health constantly",
        "I feel anxious about my relationships",
        "I worry about money all the time",
        "I'm scared of disappointing everyone",
        "I can't handle the pressure I feel",
        "I feel like I'm spiraling out of control",
        "I'm afraid of the future",
        "I can't stop thinking about what if",
        "I feel trapped in my own head",
        "I'm anxious about everything and nothing",
        "I feel like I can't catch my breath",
        "I'm scared of being judged",
        "I worry that I'm not good enough",
        "I feel like everyone will leave me",
        "I'm terrified of making mistakes",
        "I can't stop fidgeting from nervousness",
        "I feel like I'm about to snap",
        "I'm scared of my own feelings",
        "I avoid people because of anxiety",
        "I feel like I'm drowning in worry",
        "I can't think clearly when I'm anxious",
        "I worry about my health every day",
        "I'm scared of dying young",
        "I feel anxious about my job performance",
        "I overthink every conversation I have",
    ],
    "sadness": [
        "I feel empty inside",
        "Nothing makes me happy anymore",
        "I've been crying a lot lately",
        "I feel so alone in the world",
        "I don't see the point of anything",
        "I feel hopeless about my future",
        "I miss how things used to be",
        "I can't remember the last time I was happy",
        "I feel like a burden to everyone",
        "Everything feels pointless",
        "I feel so lonely even in a crowd",
        "I can't stop crying today",
        "I feel like nobody understands me",
        "I feel numb to everything",
        "I've lost interest in everything I used to love",
        "I feel like giving up",
        "I'm tired of feeling sad all the time",
        "I feel broken beyond repair",
        "Nothing brings me joy anymore",
        "I feel like I'm sinking deeper every day",
        "I miss being happy",
        "I feel invisible to everyone",
        "I can't find joy in anything",
        "I feel like my heart is heavy",
        "I don't want to get out of bed",
        "I feel disconnected from everyone",
        "I'm stuck in a cloud of sadness",
        "I feel like nobody cares about me",
        "I feel like I'm disappearing",
        "I feel like I'm failing at life",
        "I can't see a way out of this sadness",
        "I feel like I'm not important to anyone",
        "I feel like crying but no tears come",
        "I've never felt this low before",
        "I feel like I'm losing myself",
        "I feel so tired of everything",
        "I feel like I'm not worthy of love",
        "I feel like I'll always be alone",
        "I feel a deep sadness in my chest",
        "I don't know how to be happy anymore",
        "I feel like I let everyone down",
        "I feel like giving up on everything",
        "I feel so small and insignificant",
        "I can't escape this sadness",
        "I feel like I'm stuck in the dark",
        "I feel like nobody would miss me",
        "I feel like my life has no meaning",
        "I feel so tired of pretending to be okay",
        "I feel like I'm not good enough for anyone",
        "I feel like I have nothing to live for",
        "I feel like a failure",
        "I feel like everyone would be better off without me",
        "I feel like I'm trapped in my sadness",
        "I feel like I can't go on like this",
        "I feel like my soul is heavy",
        "I feel like I'm grieving something I can't name",
        "I feel like I've lost myself completely",
        "I feel like I'm watching my life from afar",
        "I feel like I'm in a fog of sadness",
        "I feel like nothing will ever get better",
    ],
    "anger": [
        "I'm so angry I could scream",
        "Everything makes me furious lately",
        "I feel like punching something",
        "I can't control my temper",
        "I feel so frustrated I want to break things",
        "I'm mad at the world",
        "I feel rage building up inside me",
        "I'm irritated by everyone and everything",
        "I can't let go of my anger",
        "I feel like I'm going to explode",
        "I'm so angry I can't think straight",
        "I feel resentful toward everyone",
        "I keep getting into arguments",
        "I feel like nobody listens to me",
        "I'm furious about how I've been treated",
        "I feel so frustrated I could cry",
        "My anger is destroying my relationships",
        "I feel like I'm always on the verge of yelling",
        "I can't stand being around people anymore",
        "I feel betrayed and angry",
        "I'm tired of being taken for granted",
        "I feel like everyone is against me",
        "I feel so much rage inside",
        "I'm angry all the time for no reason",
        "I feel like I've been treated unfairly",
        "I can't stop thinking about what made me angry",
        "I feel like punching a wall",
        "I'm so mad I can't sleep",
        "I feel like screaming at everyone",
        "I feel like my anger controls me",
        "I'm furious and I can't calm down",
        "I feel disrespected and angry",
        "I keep snapping at people",
        "I feel like I'm always angry lately",
        "I hate feeling this angry all the time",
        "I feel like someone needs to pay for what they did",
        "I'm so frustrated with my life",
        "I feel like nobody understands how I feel",
        "I can't stop feeling bitter",
        "I feel like I've been wronged",
        "I'm angry at myself",
        "I feel like I have no control over my anger",
        "I'm tired of feeling so angry",
        "I feel like I could hurt someone",
        "I feel so much resentment building up",
        "I can't forgive what happened",
        "I feel like everyone is selfish",
        "I'm tired of being walked all over",
    ],
    "stress": [
        "I'm so stressed I can't function",
        "I feel like I have too much to handle",
        "I'm overwhelmed by everything",
        "I can't keep up with my responsibilities",
        "I feel like I'm drowning in work",
        "I'm under so much pressure",
        "I feel like I can't catch a break",
        "I'm exhausted from all the stress",
        "I feel like I'm carrying the world on my shoulders",
        "I can't handle everything at once",
        "I feel like I'm going to crash",
        "I'm stressed about money and bills",
        "I feel like I have no time for myself",
        "I'm so busy I can't breathe",
        "I feel like I'm always behind",
        "I can't relax even when I have free time",
        "I feel like I'm spreading myself too thin",
        "I'm stressed about upcoming deadlines",
        "I feel like I have no control over my life",
        "I can't stop thinking about everything I have to do",
        "I feel like I'm burning out",
        "I'm so tired from all this pressure",
        "I feel like I'm failing at everything",
        "I can't manage all my commitments",
        "I feel like I'm constantly rushing",
        "I'm stressed about my health",
        "I feel like there aren't enough hours in the day",
        "I can't keep up with expectations",
        "I feel like everyone needs something from me",
        "I'm overwhelmed by my responsibilities",
        "I feel like I'm losing control",
        "I'm stressed about my family situation",
        "I feel like I can't do anything right",
        "I'm so overwhelmed I don't know where to start",
        "I feel like the pressure is too much",
        "I can't stop worrying about everything I have to do",
        "I feel like I'm running on empty",
        "I'm overwhelmed by all the decisions I need to make",
        "I feel like my stress is making me sick",
        "I can't handle all this pressure anymore",
        "I'm struggling to balance everything",
        "I feel like I'm about to break",
        "I'm overwhelmed by my own expectations",
        "I feel like I can't take a break",
        "I'm stressed about the future",
    ],
    "sleep": [
        "I can't fall asleep at night",
        "I keep waking up in the middle of the night",
        "I haven't slept well in weeks",
        "I feel exhausted but can't sleep",
        "I have terrible nightmares every night",
        "I lie awake for hours every night",
        "I'm tired all the time from lack of sleep",
        "I can't shut off my brain at bedtime",
        "I wake up feeling more tired than when I went to bed",
        "I've been having insomnia lately",
        "I can't sleep because I can't stop worrying",
        "I keep having restless nights",
        "I feel tired no matter how much I sleep",
        "I wake up multiple times every night",
        "I can't get comfortable enough to sleep",
        "I have trouble falling asleep every night",
        "I haven't had a good night's rest in months",
        "I wake up with anxiety in the middle of the night",
        "I'm stuck in a cycle of bad sleep",
        "I can't sleep through the night anymore",
        "I dread going to bed because I can't sleep",
        "I feel groggy and tired all day",
        "I need sleeping pills to get any rest",
        "I wake up at 3am and can't fall back asleep",
        "I have vivid nightmares that scare me",
        "I toss and turn all night long",
        "I can't sleep because my mind is racing",
        "I feel like I haven't slept at all",
        "I'm exhausted but my body won't let me sleep",
        "I wake up with headaches from poor sleep",
        "I can't get into a consistent sleep schedule",
        "I'm afraid to sleep because of nightmares",
        "I feel like I'm sleepwalking through life",
        "I can't nap even when I'm exhausted",
        "I sleep too much and still feel tired",
        "My sleep schedule is completely broken",
        "I can't fall asleep without the TV on",
        "I wake up feeling anxious and unrested",
        "I've tried everything and still can't sleep",
        "I grind my teeth at night from stress",
    ],
    "relationships": [
        "I'm having problems with my partner",
        "I feel like my relationship is falling apart",
        "I don't feel loved by my partner",
        "We keep fighting about everything",
        "I feel like I can't talk to my partner",
        "I think my partner is losing interest",
        "We don't communicate anymore",
        "I feel like I'm the only one trying",
        "My partner doesn't understand me",
        "I feel trapped in my relationship",
        "I'm scared of losing my partner",
        "We've grown apart over time",
        "I feel like my relationship is one-sided",
        "My partner won't listen to me",
        "I keep having the same arguments",
        "I feel like I can't trust my partner",
        "My friends don't seem to care about me",
        "I feel distant from my family",
        "I don't know how to fix my relationship",
        "I feel like I'm walking on eggshells",
        "My partner doesn't appreciate me",
        "I feel lonely in my relationship",
        "We have nothing in common anymore",
        "I feel like my family doesn't understand me",
        "My partner is always criticizing me",
        "I feel disconnected from everyone I love",
        "I think my partner is hiding things from me",
        "We never spend quality time together",
        "I feel like I can't be myself around my partner",
        "My relationship is causing me stress",
        "I miss how things used to be between us",
        "My partner doesn't support me",
        "I feel taken for granted",
        "We need to work on our communication",
        "I don't feel heard in this relationship",
        "My partner's family doesn't accept me",
        "I feel like I'm losing my friends",
        "I have trust issues in my relationship",
        "My partner and I want different things",
        "I feel like I'm settling in my relationship",
    ],
    "work": [
        "I hate my job",
        "I feel stuck in my career",
        "My boss is always criticizing me",
        "I'm not satisfied with my job",
        "I feel like I have no future at work",
        "I'm considering quitting my job",
        "I can't stand my coworkers",
        "I feel like I'm not valued at work",
        "I'm bored with my job",
        "I feel like I'm not progressing in my career",
        "I dread going to work every day",
        "My job is making me miserable",
        "I feel like I'm underpaid and overworked",
        "I don't know what career path to take",
        "I feel like I'm wasting my potential",
        "I'm scared of losing my job",
        "I can't handle the work pressure anymore",
        "I feel like my work has no meaning",
        "I'm having problems with my colleagues",
        "I feel like I need a career change",
        "My work-life balance is terrible",
        "I feel like I'm not good enough at my job",
        "I'm passed over for promotions constantly",
        "I can't concentrate at work",
        "I feel like I'm in the wrong profession",
        "My workplace is toxic",
        "I feel like I'm not respected at work",
        "I'm scared to look for a new job",
        "I feel like I've hit a dead end in my career",
        "My job is causing me too much stress",
        "I feel undervalued despite working hard",
        "I don't get along with my manager",
        "I feel like I can't grow in my current role",
        "I'm frustrated with my career trajectory",
        "I feel like I'm overqualified for my job",
        "My company doesn't care about employees",
        "I'm burned out from work",
        "I feel like I have no passion for my work",
        "I'm struggling with imposter syndrome at work",
        "I need a change in my professional life",
    ],
    "confusion": [
        "I don't know what to do with my life",
        "I feel lost and confused",
        "I don't know who I am anymore",
        "I'm unsure about my decisions",
        "I can't figure out what I want",
        "I feel directionless",
        "I don't know what path to take",
        "I'm confused about my feelings",
        "I don't understand myself anymore",
        "I feel like I'm at a crossroads",
        "I'm unsure about everything",
        "I can't make up my mind",
        "I feel like I don't know anything",
        "I'm confused about what I should do",
        "I don't know what's right anymore",
        "I feel like I'm in a fog",
        "I can't think clearly",
        "I'm torn between two choices",
        "I don't know what I believe anymore",
        "I feel like I'm questioning everything",
        "I'm confused about my identity",
        "I don't know what matters to me",
        "I feel like I have no direction",
        "I'm unsure about my future",
        "I can't figure out what's important",
        "I feel like I don't know myself",
        "I'm confused about my purpose",
        "I don't know what I'm supposed to do",
        "I'm struggling to find my way",
        "I feel like I'm going in circles",
        "I can't decide what I want",
        "I'm confused about my relationships",
        "I don't know how to move forward",
        "I feel like I'm lost in my own life",
        "I'm not sure what I believe in",
        "I can't figure out what makes me happy",
        "I feel like I'm wandering without purpose",
        "I'm confused about my goals",
        "I don't know what success means to me",
        "I feel like I'm starting over and don't know how",
    ],
    "positive": [
        "I feel really happy today",
        "I'm grateful for everything I have",
        "I feel hopeful about the future",
        "I had a really good day",
        "I'm proud of what I accomplished",
        "I feel excited about what's coming",
        "I feel at peace with myself",
        "I'm thankful for my friends and family",
        "I feel like things are getting better",
        "I'm feeling optimistic today",
        "I feel loved and supported",
        "I'm happy with how things are going",
        "I feel strong and capable",
        "I'm proud of myself for getting through this",
        "I feel joyful for no particular reason",
        "I'm grateful to be alive",
        "I feel like I'm growing as a person",
        "I'm excited about my future",
        "I feel content with my life",
        "I'm happy about my progress",
        "I feel more confident lately",
        "I'm thankful for the good things in my life",
        "I feel blessed",
        "I'm proud of how far I've come",
        "I feel peaceful and calm",
        "I'm excited to start this new chapter",
        "I feel like everything is falling into place",
        "I'm grateful for my health",
        "I feel accomplished today",
        "I'm looking forward to tomorrow",
        "I feel inspired and motivated",
        "I'm happy with who I'm becoming",
        "I feel connected to people around me",
        "I'm grateful for second chances",
        "I feel like I can handle anything",
        "I'm proud of my growth",
        "I feel surrounded by love",
        "I'm excited about my opportunities",
        "I feel like I'm exactly where I need to be",
        "I'm thankful for this beautiful day",
        "I feel like I'm making a difference",
        "I'm proud of the person I am",
        "I feel hopeful and optimistic",
        "I'm grateful for each new day",
    ],
    "general": [
        "I don't know where to start",
        "I just wanted to talk to someone",
        "I've been thinking about things lately",
        "I don't really know what to say",
        "Can we just talk?",
        "I'm not sure why I'm here",
        "I've been reflecting on my life",
        "I need someone to talk to",
        "I want to share something with you",
        "I've been meaning to talk about this",
        "I don't really understand my feelings right now",
        "I just need to vent a little",
        "I want to get something off my chest",
        "I've been thinking about the past",
        "I'm trying to process some things",
        "I feel like talking but don't know about what",
        "I need some advice about something",
        "I've been doing a lot of thinking",
        "I want to understand myself better",
        "I'm trying to figure things out",
        "I just need someone to listen",
        "There's something I've been wanting to say",
        "I've been going through a lot lately",
        "I feel like sharing something personal",
        "I'm working on understanding my emotions",
        "I want to talk about what's been happening",
        "I'm trying to be more open about my feelings",
        "I feel like I need perspective",
        "I've been noticing some patterns in my behavior",
        "I want to work on myself",
        "I'm here because I want to grow",
        "I need help understanding something",
        "I've been journaling about my thoughts",
        "I want to be more self aware",
        "I'm trying to make sense of my life",
        "I feel like talking through something",
        "I want to check in with myself",
        "I'm here to reflect on my week",
        "I want to share my thoughts with someone",
        "I need clarity on some things",
    ],
}

SHORT_EXAMPLES = {
    "anxiety": [
        "i am anxious", "i feel anxious", "i'm anxious", "i feel worried",
        "i'm scared", "i feel nervous", "i'm nervous", "i feel panic",
        "i'm panicking", "i worry a lot", "i feel afraid", "i'm afraid",
        "i feel dread", "i'm terrified", "i'm worried", "i can't stop worrying",
        "my heart is racing", "i can't calm down", "i feel tense",
        "i'm on edge", "i feel overwhelmed", "i'm overwhelmed with fear",
        "i'm having a panic attack", "i feel restless",
    ],
    "sadness": [
        "i am sad", "i feel sad", "i'm sad", "i feel depressed",
        "i'm depressed", "i feel lonely", "i'm lonely", "i feel empty",
        "i feel hopeless", "i'm hopeless", "i feel down", "i feel blue",
        "i'm crying", "i feel like crying", "i feel alone",
        "nothing makes me happy", "i feel miserable", "i'm miserable",
        "i feel so low", "i can't stop crying", "i feel hurt",
        "i feel broken", "i'm heartbroken", "i feel worthless",
        "i feel like giving up", "i'm tired of life",
    ],
    "anger": [
        "i am angry", "i feel angry", "i'm angry", "i feel mad",
        "i'm mad", "i feel rage", "i'm furious", "i feel irritated",
        "i'm irritated", "i feel frustrated", "i'm frustrated",
        "i feel annoyed", "i'm annoyed", "i feel resentful",
        "i'm so angry", "i want to scream", "i feel hostile",
        "i'm about to explode", "i feel bitter",
    ],
    "stress": [
        "i am stressed", "i feel stressed", "i'm stressed",
        "i feel overwhelmed", "i'm overwhelmed", "i feel pressured",
        "i'm under pressure", "i feel burned out", "i'm burned out",
        "i can't handle this", "i'm exhausted", "i feel exhausted",
        "too much pressure", "i'm overloaded", "i feel strained",
        "i can't keep up", "i'm stretched too thin",
    ],
    "sleep": [
        "i can't sleep", "i can't fall asleep", "i have insomnia",
        "i wake up at night", "i have nightmares", "i feel tired",
        "i'm exhausted", "i can't sleep at night", "i sleep badly",
        "i toss and turn", "bad sleep", "i wake up tired",
        "i'm restless at night", "i dread bedtime",
    ],
    "relationships": [
        "my partner", "i have relationship problems", "my boyfriend",
        "my girlfriend", "my husband", "my wife", "relationship trouble",
        "we keep fighting", "i feel unloved", "my family",
        "my friend", "trust issues", "communication problems",
        "i feel rejected", "i feel betrayed", "lonely in my relationship",
    ],
    "work": [
        "i hate my job", "my job sucks", "work is stressful",
        "i have work problems", "i feel stuck at work",
        "my career is stagnant", "i need a new job",
        "i'm unhappy at work", "i hate my boss", "work pressure",
        "i want to quit", "i feel trapped at work",
        "i have no work life balance", "i'm bored at work",
    ],
    "confusion": [
        "i am confused", "i feel confused", "i'm confused",
        "i don't know", "i'm lost", "i feel lost",
        "i don't understand", "i'm unsure", "i'm uncertain",
        "i feel directionless", "i don't know what to do",
        "i can't decide", "i'm torn", "i feel stuck",
        "i don't know myself", "i'm confused about life",
    ],
    "positive": [
        "i am happy", "i feel happy", "i'm happy", "i feel good",
        "i'm grateful", "i feel grateful", "i'm thankful",
        "i feel hopeful", "i'm hopeful", "i feel great",
        "i feel wonderful", "i'm excited", "i feel excited",
        "i feel blessed", "i'm proud", "i feel proud",
        "i feel loved", "i feel joy", "i feel at peace",
        "i'm content", "i feel content", "i'm optimistic",
        "i feel amazing", "today was good", "i'm doing well",
    ],
    "general": [
        "i want to talk", "i need to talk", "can we talk",
        "i need someone to talk to", "i want to share something",
        "i've been thinking", "i need advice", "just checking in",
        "i want to vent", "i need to get this off my chest",
        "i'm trying to understand", "i want to open up",
        "i need perspective", "i'm working on myself",
        "let me share what happened",
    ],
}

GUTENBERG_PSYCH_BOOKS = [
    66078,  # The Psychology of Management
    49563,  # The Psychology of Salesmanship
    62252,  # The Psychology of the Emotions
    64654,  # The Psychology of the Stock Market
    66100,  # The Psychology of Medicine
    59599,  # Psychology and Social Sanity
    63496,  # The Psychology of Peoples
    66013,  # Psychology and Industrial Efficiency
    66098,  # The Psychology of Special Abilities
    64603,  # The Psychology of Relaxation
    64494,  # The Psychology of the Unconscious
    64300,  # Psychotherapy
    63392,  # The Emotions and the Will
    60835,  # Character and Conduct
    58238,  # The Science of Human Nature
    56074,  # How to Analyze People on Sight
    55029,  # Mental Medicine
    53218,  # The Mind and Its Education
    52196,  # The Psychology of Child Development
    50208,  # The Analysis of Mind
    49560,  # The Psychology of Attention
    48939,  # The Psychology of Intelligence
    47855,  # The Psychology of Reasoning
    46682,  # The Psychology of Learning
    45664,  # A Manual of Psychology
]

CRISIS_KEYWORDS = [
    "suicide", "kill myself", "end my life", "want to die",
    "self-harm", "hurt myself", "not worth living", "better off dead",
    "end it all", "take my own life", "don't want to live",
    "wish i was dead", "end my suffering", "never wake up",
    "harm myself", "cut myself", "suicidal",
]


def fetch_gutenberg_text(book_id):
    url = f"https://www.gutenberg.org/cache/epub/{book_id}/pg{book_id}.txt"
    try:
        with urllib.request.urlopen(url, timeout=15) as response:
            text = response.read().decode("utf-8", errors="replace")
        start_markers = ["*** START OF THE PROJECT GUTENBERG EBOOK", "*** START OF THIS PROJECT GUTENBERG EBOOK"]
        end_markers = ["*** END OF THE PROJECT GUTENBERG EBOOK", "*** END OF THIS PROJECT GUTENBERG EBOOK"]
        for m in start_markers:
            if m in text:
                text = text.split(m, 1)[1]
                break
        for m in end_markers:
            if m in text:
                text = text.split(m, 1)[0]
                break
        text = re.sub(r"[ \t]+", " ", text)
        text = re.sub(r"\n{3,}", "\n\n", text)
        return text
    except Exception as e:
        print(f"  Failed to fetch book {book_id}: {e}")
        return None


def label_text_with_keywords(text, label):
    def contains_any(t, keywords):
        return any(kw in t.lower() for kw in keywords)

    keyword_map = {
        "anxiety": ["anxiety", "anxious", "worry", "fear", "panic", "nervous", "dread", "scared", "afraid", "tense"],
        "sadness": ["sad", "depress", "cry", "lonely", "grief", "sorrow", "hopeless", "unhappy", "miserable", "gloom"],
        "anger": ["anger", "angry", "rage", "fury", "furious", "mad", "irritat", "frustrat", "hostile", "resent"],
        "stress": ["stress", "pressure", "overwhelm", "burden", "exhaust", "strain", "tense", "burnout", "fatigue"],
        "sleep": ["sleep", "insomnia", "nightmare", "tired", "restless", "dream", "awake", "bedtime", "sleepless"],
        "relationships": ["relation", "partner", "marriage", "friend", "family", "love", "trust", "intimacy", "couple"],
        "work": ["work", "job", "career", "boss", "colleague", "employ", "profession", "occupation", "vocation"],
        "confusion": ["confus", "uncertain", "unsure", "doubt", "ambiguous", "unclear", "perplexed", "baffled"],
        "positive": ["happy", "joy", "grateful", "thankful", "hope", "excite", "love", "wonderful", "blessed", "peace"],
    }

    sentences = re.split(r"(?<=[.!?])\s+", text)
    labeled = []
    for sent in sentences:
        stripped = sent.strip()
        if len(stripped.split()) < 4 or len(stripped) > 250:
            continue
        for cat, keywords in keyword_map.items():
            if contains_any(stripped, keywords):
                labeled.append((stripped, cat))
                break
    return labeled


def build_synthetic_df():
    rows = []
    for label, texts in LABELED_DATA.items():
        for t in texts:
            rows.append({"text": t, "label": label})
    for label, texts in SHORT_EXAMPLES.items():
        for t in texts:
            rows.append({"text": t, "label": label})
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("category")
    print(f"  Synthetic dataset: {len(df)} samples")
    print(f"  Class distribution:\n{df['label'].value_counts()}")
    return df


def build_ebook_df(max_books=6):
    rows = []
    np.random.shuffle(GUTENBERG_PSYCH_BOOKS)
    for i, book_id in enumerate(GUTENBERG_PSYCH_BOOKS[:max_books]):
        print(f"  Fetching book {i+1}/{max_books} (ID: {book_id})...")
        text = fetch_gutenberg_text(book_id)
        if text is None:
            continue
        labeled = label_text_with_keywords(text, book_id)
        for t, label in labeled:
            rows.append({"text": t, "label": label})
        print(f"    Got {len(labeled)} labeled sentences")
    if not rows:
        print("  No ebook data fetched, skipping.")
        return None
    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("category")
    print(f"  Ebook dataset: {len(df)} samples")
    print(f"  Class distribution:\n{df['label'].value_counts()}")
    return df


def balance_dataset(df, target_per_class=80):
    texts, labels = [], []
    for cat in df["label"].unique():
        subset = df[df["label"] == cat]
        if len(subset) >= target_per_class:
            sampled = subset.sample(n=target_per_class, random_state=SEED)
        else:
            sampled = subset.sample(n=target_per_class, replace=True, random_state=SEED)
        texts.extend(sampled["text"].tolist())
        labels.extend([cat] * target_per_class)
    balanced = pd.DataFrame({"text": texts, "label": pd.Categorical(labels)})
    balanced = balanced.sample(frac=1, random_state=SEED).reset_index(drop=True)
    print(f"  Balanced dataset: {len(balanced)} samples ({len(balanced['label'].unique())} classes x {target_per_class})")
    return balanced


def train_model(X_train, y_train, max_features=5000):
    pipeline = Pipeline([
        ("tfidf", TfidfVectorizer(
            max_features=max_features,
            ngram_range=(1, 3),
            min_df=2,
            max_df=0.85,
            sublinear_tf=True,
            stop_words="english",
        )),
        ("clf", LogisticRegression(
            solver="lbfgs",
            max_iter=1000,
            random_state=SEED,
            C=1.5,
            class_weight="balanced",
        )),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline


def main():
    print("=" * 60)
    print("COUNSELOR MODEL TRAINING")
    print("=" * 60)

    print("\n[1/5] Building synthetic dataset...")
    synthetic_df = build_synthetic_df()

    print("\n[2/5] Fetching ebook data (skipped - no network)...")
    print("  Using synthetic data only.")
    combined = synthetic_df

    print("\n[3/5] Balancing dataset...")
    per_class = int(max(len(combined) / len(combined["label"].unique()) * 1.2, 80))
    balanced = balance_dataset(combined, target_per_class=per_class)

    print("\n[4/5] Training model...")
    X = balanced["text"].to_numpy(dtype=str)
    y = balanced["label"].to_numpy(dtype=str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    model = train_model(X_train, y_train)

    print("\n[5/5] Evaluating model...")
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)
    print(f"\n  Test accuracy: {np.mean(y_pred == y_test):.3f}")
    print(f"  Avg confidence: {np.max(y_proba, axis=1).mean():.3f}")
    print(f"\n  Classification Report:")
    print(classification_report(y_test, y_pred, zero_division=0))

    print("\n  Confusion Matrix (rows=true, cols=pred):")
    cm = confusion_matrix(y_test, y_pred)
    cm_df = pd.DataFrame(
        cm,
        index=model.named_steps["clf"].classes_,
        columns=model.named_steps["clf"].classes_,
    )
    print(cm_df.to_string())

    model_path = os.path.join(os.path.dirname(__file__), "counselor_model.joblib")
    joblib.dump(model, model_path)
    print(f"\n  Model saved to: {model_path}")
    print("  Training complete!")


if __name__ == "__main__":
    main()
