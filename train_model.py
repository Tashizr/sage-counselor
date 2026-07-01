import numpy as np
import pandas as pd
import re
import json
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
        "I'm terrified of failing",
        "I can't breathe when I get anxious",
        "I'm scared of having another panic attack",
        "I feel like I'm constantly on high alert",
        "I'm afraid to leave my house",
        "I feel paralyzed by fear",
        "I need reassurance constantly",
        "I keep thinking I'm going to die",
        "I'm terrified of being alone",
        "I worry about things that don't matter",
        "I'm scared to speak up in meetings",
        "I obsess over my health constantly",
        "I'm scared of disappointing everyone",
        "I'm afraid of the future",
        "I'm scared of being judged",
        "I worry that I'm not good enough",
        "I'm terrified of making mistakes",
        "I'm scared of my own feelings",
        "I avoid people because of anxiety",
        "I can't think clearly when I'm anxious",
        "I worry about my health every day",
        "I'm scared of dying young",
        "I overthink every conversation I have",
        "I have social anxiety and fear being judged",
        "I feel panicked for no reason at all",
        "My anxiety makes it hard to leave the house",
        "I feel restless and on edge constantly",
        "I can't stop my anxious thoughts",
        "I feel like something terrible is about to happen",
        "I obsess over what people think of me",
        "My stomach is in knots from worrying",
        "I keep catastrophizing everything",
        "I've been having panic attacks and I'm scared",
        "I worry about my relationships constantly",
        "I ruminate about the same things over and over",
        "I feel like I'm losing my mind with worry",
        "Every little noise makes me jump",
        "I can't relax no matter what I do",
        "Dating gives me extreme anxiety",
        "I feel anxious before every social event",
        "My heart pounds when my phone rings",
        "I avoid checking my email because of anxiety",
        "I feel like everyone is watching me",
        "I can't stop worrying about what might happen",
        "I have a constant knot in my stomach",
        "I feel like I can't breathe when I'm anxious",
        "My hands shake when I'm nervous",
        "I feel dizzy and lightheaded from anxiety",
        "I overthink every text message I send",
        "I'm afraid of public speaking",
        "I worry about things that haven't even happened yet",
        "I have trouble breathing when I get anxious",
        "I feel trapped by my own anxious thoughts",
        "I can't stop imagining worst case scenarios",
        "My mind races with anxious thoughts at night",
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
        "I've lost interest in everything I used to love",
        "I feel like giving up",
        "I'm tired of feeling sad all the time",
        "I feel broken beyond repair",
        "Nothing brings me joy anymore",
        "I can't find joy in anything",
        "I feel disconnected from everyone",
        "I'm stuck in a cloud of sadness",
        "I feel like nobody cares about me",
        "I've never felt this low before",
        "I feel like I'm losing myself",
        "I feel like I'm not worthy of love",
        "I feel a deep sadness in my chest",
        "I don't know how to be happy anymore",
        "I feel like I let everyone down",
        "I feel so small and insignificant",
        "I can't escape this sadness",
        "I'm grieving someone I lost",
        "I lost someone I love and I don't know how to cope",
        "I feel guilty for being sad when others have it worse",
        "I feel numb to everything around me",
        "I miss being happy so much it hurts",
        "I feel like my heart is heavy with grief",
        "I feel like crying but no tears come anymore",
        "I feel so tired of pretending to be okay",
        "I feel like my life has no meaning",
        "I feel like I have nothing to live for",
        "I feel like I'm watching my life from afar",
        "I feel like nothing will ever get better",
        "I feel a deep sense of grief I can't explain",
        "I lost my parent and I don't know how to move on",
        "I'm grieving a relationship that ended",
        "I feel heartbroken and lost",
        "I feel like I'm drowning in sorrow",
        "I miss them so much it physically hurts",
        "I feel like I'll never be happy again",
        "I feel so tired of everything",
        "I feel like I can't go on like this",
        "I feel invisible to everyone around me",
        "I feel like nobody would notice if I disappeared",
        "I'm mourning the life I thought I would have",
        "I feel a profound sense of loss",
        "I feel like my soul is empty",
        "I've been isolating myself from everyone",
        "I feel like I don't belong anywhere",
    ],
    "anger": [
        "I'm so angry I could scream",
        "Everything makes me furious lately",
        "I feel like punching something",
        "I can't control my temper",
        "I'm mad at the world",
        "I feel rage building up inside me",
        "I'm irritated by everyone and everything",
        "I can't let go of my anger",
        "I feel like I'm going to explode",
        "I'm so angry I can't think straight",
        "I feel resentful toward everyone",
        "I keep getting into arguments",
        "I'm furious about how I've been treated",
        "I feel betrayed and angry",
        "I'm tired of being taken for granted",
        "I feel like everyone is against me",
        "I'm angry all the time for no reason",
        "I feel like I've been treated unfairly",
        "I can't stop thinking about what made me angry",
        "I feel like punching a wall",
        "I'm so mad I can't sleep",
        "I feel like screaming at everyone",
        "I feel like my anger controls me",
        "I keep snapping at people",
        "I hate feeling this angry all the time",
        "I'm so frustrated with my life",
        "I can't stop feeling bitter about what happened",
        "I feel like I've been wronged and I can't let go",
        "I'm angry at myself for making bad choices",
        "I feel like I have no control over my anger",
        "I'm tired of feeling so angry all the time",
        "I can't forgive what they did to me",
        "I feel like everyone is selfish and doesn't care",
        "I'm tired of being walked all over",
        "I feel disrespected every single day",
        "I feel so much hatred inside and I hate it",
        "I'm angry because nobody listens to me",
        "I feel like my anger is destroying everything",
        "I was treated unfairly and I can't move past it",
        "I feel frustrated that nothing ever goes my way",
        "I keep getting into fights with people I love",
        "I feel hostile toward everyone for no reason",
        "Small things set me off and I don't know why",
        "I feel like I've been holding in anger for years",
        "I'm furious that they lied to me",
        "I feel enraged when I think about what happened",
        "I can't stop feeling irritated by everything",
        "I feel like hurting someone who hurt me",
        "I've been holding onto this grudge for too long",
        "I feel like my anger is eating me alive",
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
        "I can't manage all my commitments",
        "I'm overwhelmed by my responsibilities",
        "I'm stressed about my family situation",
        "I feel like I'm running on empty",
        "I can't handle all this pressure anymore",
        "I'm struggling to balance everything",
        "I feel like I'm about to break",
        "I'm stressed about the future",
        "I'm overwhelmed by all the decisions I need to make",
        "I feel like I'm constantly rushing everywhere",
        "I have too many people depending on me",
        "I feel stressed about my health problems",
        "I'm overwhelmed by work and home responsibilities",
        "I feel like there aren't enough hours in the day",
        "I'm stressed about taking care of my parents",
        "I feel like everyone needs something from me",
        "I'm overwhelmed by the pressure to succeed",
        "I can't keep up with everyone's expectations",
        "I feel like I'm failing at everything I do",
        "I'm tired of being pulled in every direction",
        "I feel like I have to do everything myself",
        "I'm stressed about my financial situation",
        "I'm overwhelmed by my caregiving responsibilities",
        "I feel like I'm stretched too thin to function",
        "I can't stop worrying about all my responsibilities",
        "I feel like my stress is making me physically sick",
        "I'm overwhelmed by my partner's expectations",
        "I feel like I'm one step away from a breakdown",
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
        "I have trouble falling asleep every night",
        "I haven't had a good night's rest in months",
        "I wake up with anxiety in the middle of the night",
        "I'm stuck in a cycle of bad sleep",
        "I can't sleep through the night anymore",
        "I dread going to bed because I can't sleep",
        "I feel groggy and tired all day",
        "I need sleeping pills to get any rest",
        "I wake up at 3am and can't fall back asleep",
        "I toss and turn all night long",
        "I can't sleep because my mind is racing",
        "I feel like I haven't slept at all",
        "I'm exhausted but my body won't let me sleep",
        "I wake up with headaches from poor sleep",
        "I'm afraid to sleep because of nightmares",
        "I feel like I'm sleepwalking through life",
        "I sleep too much and still feel tired",
        "My sleep schedule is completely broken",
        "I've tried everything and still can't sleep",
        "I grind my teeth at night from stress",
        "I have chronic insomnia and it's affecting everything",
        "My sleep anxiety keeps me awake every night",
        "I feel like I haven't slept properly in years",
        "I wake up feeling anxious and unrested",
        "I have vivid nightmares that leave me scared",
        "I can't nap even when I'm exhausted",
        "I feel trapped in a cycle of bad sleep",
        "I wake up gasping for air at night",
        "My partner says I thrash around while sleeping",
        "I can't sleep when I'm away from home",
        "I stay up late even though I know I need rest",
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
        "My partner is always criticizing me",
        "I think my partner is hiding things from me",
        "We never spend quality time together",
        "I feel like I can't be myself around my partner",
        "My relationship is causing me stress",
        "My partner doesn't support me",
        "I feel taken for granted",
        "I don't feel heard in this relationship",
        "My partner and I want different things",
        "I feel like I'm settling in my relationship",
        "I caught my partner cheating",
        "I'm going through a breakup and it hurts so much",
        "My partner and I are getting divorced",
        "I feel like I've been manipulated in my relationship",
        "I feel gaslit by my partner",
        "My family doesn't approve of my partner",
        "I have trust issues from past relationships",
        "I feel rejected by someone I love",
        "My heart is broken after the breakup",
        "I feel like I'll never find love again",
        "I'm scared to open up after being hurt",
        "My partner invalidates my feelings constantly",
        "I feel like I'm being controlled in my relationship",
        "I feel lonely even when I'm with my partner",
        "My partner and I have different life goals",
        "I feel like my friends are drifting away",
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
        "I don't get along with my manager",
        "I feel like I can't grow in my current role",
        "I'm frustrated with my career trajectory",
        "I'm burned out from work",
        "I'm struggling with imposter syndrome at work",
        "I was laid off from my job and feel lost",
        "I lost my job and don't know what to do next",
        "I feel like I have no passion for my work",
        "My workplace is full of bullying and gossip",
        "I feel anxious every Sunday night before work",
        "I have to work overtime every day and I'm exhausted",
        "I'm underqualified for my job and terrified of failing",
        "I feel like my career has no direction",
        "I'm embarrassed about being unemployed",
        "I feel like I'll never find a job I enjoy",
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
        "I don't know how to move forward",
        "I feel like I'm lost in my own life",
        "I'm not sure what I believe in anymore",
        "I can't figure out what makes me happy",
        "I feel like I'm wandering without purpose",
        "I'm confused about my goals",
        "I don't know what success means to me",
        "I feel like I'm starting over and don't know how",
        "I'm questioning every decision I've ever made",
        "I feel like I don't know what I want anymore",
        "I'm confused about my sexuality",
        "I'm struggling with my faith and beliefs",
        "I don't know if I'm making the right choice",
        "I feel like I'm having an identity crisis",
        "I'm unsure about my values and what I stand for",
        "I feel like everything I believed was wrong",
        "I don't know if I should stay or leave",
        "I'm confused about what I want in a relationship",
        "I feel like I'm reinventing myself and it's scary",
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
        "I'm proud of my personal growth",
        "I feel surrounded by love and support",
        "I'm excited about new opportunities",
        "I feel like I'm exactly where I need to be",
        "I'm thankful for this beautiful day",
        "I feel like I'm making a difference",
        "I'm proud of the person I am becoming",
        "I feel hopeful and optimistic about life",
        "I'm grateful for each new day I get",
        "I feel inspired to pursue my dreams",
        "I feel deeply connected to the people I love",
        "I feel a sense of peace I haven't felt in years",
        "I'm proud of myself for asking for help",
        "I feel grateful for my support system",
        "I feel excited about the possibilities ahead",
        "I feel a deep sense of gratitude today",
        "I'm proud that I'm healing and growing",
        "I feel joyful and lighthearted today",
        "I feel like I'm finally becoming myself",
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
        "I feel like I need perspective on something",
        "I've been noticing some patterns in my behavior",
        "I want to work on myself and grow",
        "I'm here because I want to change",
        "I need help understanding something about myself",
        "I've been journaling about my thoughts lately",
        "I want to be more self aware",
        "I'm trying to make sense of my life",
        "I want to check in with myself",
        "I'm here to reflect on my week",
        "I want to share my thoughts with someone safe",
        "I need clarity on some things in my life",
        "I feel like I need to talk through something",
        "I'm trying to be more honest about my feelings",
        "I want to learn how to cope better",
        "I'm here to work on my mental health",
        "I've been wanting to reach out for help",
        "I'm trying to process something that happened",
        "I need help sorting out my thoughts",
        "I want to be a better version of myself",
        "I'm here to talk through a decision I need to make",
        "I want to share what's been on my mind lately",
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
        "i feel on edge", "i'm scared of everything", "i can't relax",
        "i feel jumpy", "i'm full of worry", "i feel fearful",
        "i'm having anxiety", "i'm really nervous", "i feel terrified",
        "everything scares me", "i feel unsafe",
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
        "i have no hope", "i feel grief", "i'm grieving",
        "i feel heartbroken", "i feel numb", "i'm numb",
        "i feel terrible", "i feel awful", "life feels pointless",
    ],
    "anger": [
        "i am angry", "i feel angry", "i'm angry", "i feel mad",
        "i'm mad", "i feel rage", "i'm furious", "i feel irritated",
        "i'm irritated", "i feel frustrated", "i'm frustrated",
        "i feel annoyed", "i'm annoyed", "i feel resentful",
        "i'm so angry", "i want to scream", "i feel hostile",
        "i'm about to explode", "i feel bitter",
        "i'm furious", "i'm enraged", "i hate everyone",
        "i feel hatred", "everything makes me mad",
    ],
    "stress": [
        "i am stressed", "i feel stressed", "i'm stressed",
        "i feel overwhelmed", "i'm overwhelmed", "i feel pressured",
        "i'm under pressure", "i feel burned out", "i'm burned out",
        "i can't handle this", "i'm exhausted", "i feel exhausted",
        "too much pressure", "i'm overloaded", "i feel strained",
        "i can't keep up", "i'm stretched too thin",
        "i'm swamped", "i have too much to do", "i'm drowning",
    ],
    "sleep": [
        "i can't sleep", "i can't fall asleep", "i have insomnia",
        "i wake up at night", "i have nightmares", "i feel tired",
        "i'm exhausted", "i can't sleep at night", "i sleep badly",
        "i toss and turn", "bad sleep", "i wake up tired",
        "i'm restless at night", "i dread bedtime",
        "i'm sleep deprived", "i have trouble sleeping",
    ],
    "relationships": [
        "my partner", "i have relationship problems", "my boyfriend",
        "my girlfriend", "my husband", "my wife", "relationship trouble",
        "we keep fighting", "i feel unloved", "my family",
        "my friend", "trust issues", "communication problems",
        "i feel rejected", "i feel betrayed", "lonely in my relationship",
        "my marriage is struggling", "i'm going through a breakup",
        "i feel heartbroken", "we don't communicate",
    ],
    "work": [
        "i hate my job", "my job sucks", "work is stressful",
        "i have work problems", "i feel stuck at work",
        "my career is stagnant", "i need a new job",
        "i'm unhappy at work", "i hate my boss", "work pressure",
        "i want to quit", "i feel trapped at work",
        "i have no work life balance", "i'm bored at work",
        "i lost my job", "i'm unemployed", "career confusion",
    ],
    "confusion": [
        "i am confused", "i feel confused", "i'm confused",
        "i don't know", "i'm lost", "i feel lost",
        "i don't understand", "i'm unsure", "i'm uncertain",
        "i feel directionless", "i don't know what to do",
        "i can't decide", "i'm torn", "i feel stuck",
        "i don't know myself", "i'm confused about life",
        "i have no idea what i want", "i'm questioning everything",
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
        "i'm feeling great", "life is good", "i feel fantastic",
    ],
    "general": [
        "i want to talk", "i need to talk", "can we talk",
        "i need someone to talk to", "i want to share something",
        "i've been thinking", "i need advice", "just checking in",
        "i want to vent", "i need to get this off my chest",
        "i'm trying to understand", "i want to open up",
        "i need perspective", "i'm working on myself",
        "let me share what happened",
        "i need to process something", "i want to be honest",
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


EMOTION_TOPIC_MAP = {
    "anxiety": "anxiety", "panic": "anxiety", "fear": "anxiety", "dread": "anxiety",
    "sadness": "sadness", "grief": "sadness", "loneliness": "sadness", "hopelessness": "sadness",
    "anger": "anger", "frustration": "anger", "resentment": "anger", "rage": "anger",
    "stress": "stress", "overwhelm": "stress", "burnout": "stress", "pressure": "stress",
    "sleep": "sleep", "insomnia": "sleep", "exhaustion": "sleep",
    "relationships": "relationships", "love": "relationships", "trust": "relationships",
    "work": "work", "career": "work", "job": "work",
    "confusion": "confusion", "ambivalence": "confusion", "uncertainty": "confusion",
    "gratitude": "positive", "pride": "positive", "hope": "positive", "calm": "positive",
    "contentment": "positive", "joy": "positive", "excitement": "positive",
}


def build_from_knowledge_base(kb_dir=None):
    if kb_dir is None:
        kb_dir = os.path.join(os.path.dirname(__file__), "knowledge_base")
    if not os.path.isdir(kb_dir):
        print("  Knowledge base directory not found, skipping.")
        return None

    rows = []

    files_to_process = [
        "01_emotions.json", "02_life_events.json", "03_cognitive_patterns.json",
        "04_behavior_patterns.json", "06_conversation_examples.jsonl",
        "07_crisis_detection.json", "08_coping_strategies.json", "09_language_patterns.json",
    ]

    for fname in files_to_process:
        fpath = os.path.join(kb_dir, fname)
        if not os.path.exists(fpath):
            continue
        try:
            if fname.endswith(".jsonl"):
                with open(fpath, "r", encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        entry = json.loads(line)
                        text = entry.get("user", "")
                        emos = entry.get("emotions", [])
                        if text and emos:
                            for emo in emos:
                                topic = EMOTION_TOPIC_MAP.get(emo.lower(), "general")
                                rows.append({"text": text, "label": topic, "source": fname})
            elif fname == "01_emotions.json":
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for emo in data.get("emotions", []):
                    t = EMOTION_TOPIC_MAP.get(emo["name"].lower(), "general")
                    for phrase in emo.get("example_phrases", []):
                        if phrase:
                            rows.append({"text": phrase, "label": t, "source": fname})
                    for thought in emo.get("common_thoughts", []):
                        if thought:
                            rows.append({"text": thought, "label": t, "source": fname})
            elif fname == "02_life_events.json":
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for ev in data.get("events", []):
                    for resp in ev.get("helpful_responses", []):
                        if resp:
                            rows.append({"text": resp, "label": "general", "source": fname})
            elif fname == "03_cognitive_patterns.json":
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for pat in data.get("patterns", []):
                    for thought in pat.get("typical_thoughts", []):
                        if thought:
                            t = EMOTION_TOPIC_MAP.get(pat.get("emotion", "").split(",")[0].strip().lower(), "general")
                            rows.append({"text": thought, "label": t, "source": fname})
            elif fname == "04_behavior_patterns.json":
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for pat in data.get("patterns", []):
                    for resp in pat.get("supportive_responses", []):
                        if resp:
                            rows.append({"text": resp, "label": "general", "source": fname})
            elif fname == "07_crisis_detection.json":
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for ct in data.get("crisis_types", []):
                    for phrase in ct.get("warning_phrases", []):
                        if phrase:
                            rows.append({"text": phrase, "label": "anxiety", "source": fname})
            elif fname == "08_coping_strategies.json":
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for strat in data.get("strategies", []):
                    for target in strat.get("target", []):
                        t = EMOTION_TOPIC_MAP.get(target.lower(), "general")
                        rows.append({"text": strat["instructions"], "label": t, "source": fname})
            elif fname == "09_language_patterns.json":
                with open(fpath, "r", encoding="utf-8") as f:
                    data = json.load(f)
                for pat in data.get("patterns", []):
                    phrase = pat.get("phrase", "")
                    emo = pat.get("likely_emotion", "")
                    if phrase and emo:
                        t = EMOTION_TOPIC_MAP.get(emo.lower(), "general")
                        rows.append({"text": phrase, "label": t, "source": fname})
        except Exception as e:
            print(f"  Error processing {fname}: {e}")
            continue

    if not rows:
        print("  No training data extracted from knowledge base.")
        return None

    df = pd.DataFrame(rows)
    df["label"] = df["label"].astype("category")
    print(f"  Knowledge base training samples: {len(df)}")
    print(f"  KB class distribution:\n{df['label'].value_counts()}")
    return df


def main():
    print("=" * 60)
    print("COUNSELOR MODEL TRAINING")
    print("=" * 60)

    print("\n[1/6] Building synthetic dataset...")
    synthetic_df = build_synthetic_df()

    print("\n[2/6] Building knowledge base dataset...")
    kb_df = build_from_knowledge_base()

    print("\n[3/6] Combining datasets...")
    dfs = [synthetic_df]
    if kb_df is not None:
        dfs.append(kb_df)
    combined = pd.concat(dfs, ignore_index=True)
    print(f"  Combined dataset: {len(combined)} samples")

    print("\n[4/6] Balancing dataset...")
    per_class = int(max(len(combined) / len(combined["label"].unique()) * 1.2, 80))
    balanced = balance_dataset(combined, target_per_class=per_class)

    print("\n[5/6] Training model...")
    X = balanced["text"].to_numpy(dtype=str)
    y = balanced["label"].to_numpy(dtype=str)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=SEED, stratify=y
    )
    print(f"  Train: {len(X_train)} samples, Test: {len(X_test)} samples")
    model = train_model(X_train, y_train)

    print("\n[6/6] Evaluating model...")
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
