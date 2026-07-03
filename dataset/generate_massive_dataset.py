#!/usr/bin/env python3
"""
SAGE Massive Training Dataset Generator
Generates 250,000+ diverse conversation examples for fine-tuning
an AI emotional support assistant.

Covers: greetings, introductions, identity, small talk, relationships,
school, work, family, health, emotions, slang, ambiguous statements,
context memory, crisis awareness, and more.
"""

import json
import random
import uuid
import os
import itertools

random.seed(42)

# ============================================================
# CONFIGURATION
# ============================================================

TARGET_COUNT = 260_000
OUTPUT_PATH = os.path.join(os.path.dirname(__file__), "sage_massive_training.jsonl")
BATCH_SIZE = 50_000

# ============================================================
# NAME POOLS (diverse cultures)
# ============================================================

FIRST_NAMES_M = [
    "James", "John", "Robert", "Michael", "David", "William", "Richard", "Joseph",
    "Thomas", "Christopher", "Daniel", "Matthew", "Anthony", "Mark", "Donald",
    "Steven", "Paul", "Andrew", "Joshua", "Kenneth", "Kevin", "Brian", "George",
    "Timothy", "Ronald", "Edward", "Jason", "Jeffrey", "Ryan", "Jacob", "Gary",
    "Nicholas", "Eric", "Jonathan", "Stephen", "Larry", "Justin", "Scott", "Brandon",
    "Benjamin", "Samuel", "Raymond", "Gregory", "Frank", "Alexander", "Patrick",
    "Jack", "Dennis", "Jerry", "Tyler", "Aaron", "Jose", "Nathan", "Henry",
    "Peter", "Douglas", "Adam", "Zachary", "Walter", "Kyle", "Harold", "Carl",
    "Jeremy", "Gerald", "Keith", "Roger", "Arthur", "Terry", "Lawrence", "Jordan",
    "Jesse", "Bryan", "Billy", "Bruce", "Gabriel", "Joe", "Logan", "Albert",
    "Willie", "Alan", "Eugene", "Russell", "Vincent", "Philip", "Bobby", "Harry",
    "Noah", "Liam", "Mason", "Ethan", "Lucas", "Oliver", "Aiden", "Elijah",
    "Mateo", "Jayden", "Owen", "Samuel", "Ryan", "Isaiah", "Caleb", "Nathan",
    "Muhammad", "Ahmed", "Omar", "Hassan", "Raj", "Arjun", "Wei", "Hiroshi",
    "Carlos", "Miguel", "Diego", "Santiago", "Bruno", "Andre", "Rafael", "Luis",
    "Anuj", "Vikram", "Rohan", "Aditya", "Karthik", "Siddharth", "Nikhil", "Priyansh",
    "Kwame", "Kofi", "Amir", "Yusuf", "Ibrahim", "Tariq", "Rashid", "Malik",
    "Alejandro", "Javier", "Ricardo", "Sergio", "Fernando", "Eduardo", "Pablo", "Manuel",
    "Erik", "Sven", "Lars", "Ole", "Mikael", "Jan", "Pavel", "Dimitri",
    "Seung", "Joon", "Min", "Hyun", "Tae", "Soo", "Ji", "Chul",
]

FIRST_NAMES_F = [
    "Mary", "Patricia", "Jennifer", "Linda", "Barbara", "Elizabeth", "Susan",
    "Jessica", "Sarah", "Karen", "Lisa", "Nancy", "Betty", "Margaret", "Sandra",
    "Ashley", "Dorothy", "Kimberly", "Emily", "Donna", "Michelle", "Carol",
    "Amanda", "Melissa", "Deborah", "Stephanie", "Rebecca", "Sharon", "Laura",
    "Cynthia", "Kathleen", "Amy", "Angela", "Shirley", "Anna", "Brenda",
    "Pamela", "Emma", "Nicole", "Helen", "Samantha", "Katherine", "Christine",
    "Debra", "Rachel", "Carolyn", "Janet", "Catherine", "Maria", "Heather",
    "Diane", "Ruth", "Julie", "Olivia", "Joyce", "Virginia", "Victoria",
    "Kelly", "Lauren", "Christina", "Joan", "Evelyn", "Judith", "Megan",
    "Andrea", "Cheryl", "Hannah", "Jacqueline", "Martha", "Gloria", "Teresa",
    "Ann", "Sara", "Madison", "Frances", "Kathryn", "Janice", "Jean",
    "Abigail", "Alice", "Judy", "Sophia", "Grace", "Denise", "Amber",
    "Doris", "Marilyn", "Danielle", "Beverly", "Isabella", "Theresa", "Diana",
    "Natalie", "Brittany", "Charlotte", "Marie", "Kayla", "Alexis", "Lori",
    "Aisha", "Fatima", "Zara", "Noor", "Priya", "Ananya", "Meera", "Lakshmi",
    "Yuki", "Sakura", "Hana", "Kimiko", "Mei", "Lin", "Xiao", "Fang",
    "Sofia", "Valentina", "Camila", "Isabella", "Mariana", "Gabriela", "Luciana", "Ana",
    "Olga", "Nadia", "Tatiana", "Irina", "Yelena", "Katya", "Marina", "Darya",
    "Hyejin", "Soyeon", "Minji", "Jiwon", "Seoyeon", "Dayoung", "Eunji", "Hayoung",
]

NONBINARY_NAMES = [
    "Alex", "Jordan", "Casey", "Riley", "Morgan", "Taylor", "Quinn", "Avery",
    "Charlie", "Sage", "Drew", "Robin", "Skyler", "Finley", "Hayden", "Emerson",
    "Rowan", "Blair", "Reese", "Dakota", "Kai", "River", "Phoenix", "Seren",
]

ALL_NAMES = FIRST_NAMES_M + FIRST_NAMES_F + NONBINARY_NAMES

# ============================================================
# INTENT CATEGORIES AND TEMPLATES
# ============================================================

INTENTS = {
    "greeting": {
        "weight": 8,
        "messages": [
            "hi", "hello", "hey", "yo", "sup", "wassup", "good morning",
            "good evening", "good night", "howdy", "hiya", "heyy", "heyyyy",
            "hiiii", "yo yo", "ayy", "what's good", "what's up", "how goes it",
            "top of the morning", "evening", "morning", "afternoon",
            "hey there", "hi there", "hello there", "sup dude", "yo what's up",
            "heyy what's up", "hii", "hi!!", "hey!!", "helloo", "heyyy",
            "wsg", "wa gwan", "wagwan", "yooo", "ayooo",
        ],
        "responses": [
            "Hey. I'm glad you're here. What's on your mind?",
            "Hi there. I'm here whenever you're ready to talk.",
            "Hey. Thanks for reaching out. What brings you here today?",
            "Hello. I'm here to listen. What's going on?",
            "Hey. Take your time — I'm around whenever you feel like sharing.",
            "Hi. I'm here. What would you like to talk about?",
            "Hey there. How are you doing right now?",
            "Hello. It's good to see you. What's been happening?",
            "Hey. I'm here for you. What's on your mind?",
            "Hi. Welcome. What's going on with you today?",
        ],
    },
    "introduction": {
        "weight": 6,
        "messages": [
            "my name is {name}",
            "i'm {name}",
            "call me {name}",
            "you can call me {name}",
            "i go by {name}",
            "the name's {name}",
            "it's {name}",
            "i'm {name}, nice to meet you",
            "hey i'm {name}",
            "just so you know my name is {name}",
            "hi im {name}",
            "hello my name is {name}",
            "hey my name's {name}",
            "sup im {name}",
            "yo im {name}",
            "my name's {name}",
            "im {name} btw",
            "btw my name is {name}",
            "just wanted to say hi im {name}",
            "hi there im {name}",
        ],
        "responses": [
            "Hi, {name}! It's nice to meet you. I'm SAGE. I'm glad you reached out today. What's been on your mind?",
            "Hey, {name}! Welcome. I'm glad you're here. What's on your mind today?",
            "Hi, {name}! It's great to meet you. Thanks for introducing yourself. How can I support you today?",
            "Of course, {name}. It's nice to meet you. What would you like to talk about today?",
            "Nice to meet you, {name}! I'm SAGE. I'm here to listen. What brings you here today?",
            "Hey {name}, welcome! I'm glad you're here. What's been going on with you?",
            "Hi {name}! It's nice to put a name to this conversation. What's on your mind?",
            "Hello, {name}! I'm SAGE. Thanks for sharing your name. What would you like to talk about?",
        ],
    },
    "age_sharing": {
        "weight": 4,
        "messages": [
            "i'm {age} years old",
            "i'm {age}",
            "im {age}",
            "i just turned {age}",
            "i'm {age} btw",
            "fyi i'm {age}",
            "im {age} rn",
            "im only {age}",
            "im already {age}",
            "just turned {age} last week",
        ],
        "responses": [
            "Thanks for sharing that, {name}. {age} is a meaningful age. What's been on your mind?",
            "Got it, {name}. Being {age} comes with its own challenges. What's going on?",
            "I appreciate you telling me, {name}. What brings you here at {age}?",
            "Okay {name}, {age} — that's a lot of life experience already. What would you like to talk about?",
        ],
    },
    "identity": {
        "weight": 4,
        "messages": [
            "i'm a boy", "i'm a girl", "i'm nonbinary", "i'm non-binary",
            "i use they/them pronouns", "i use she/her pronouns", "i use he/him pronouns",
            "i'm a guy", "i'm a dude", "i'm a woman", "i'm a man",
            "i'm nonbinary and use they/them", "i identify as nonbinary",
            "i'm genderfluid", "i'm agender", "i use ze/zir pronouns",
            "i'm trans", "i'm transgender", "i'm a trans guy", "i'm a trans girl",
        ],
        "responses": [
            "Thank you for sharing that with me, {name}. I'll keep that in mind. What's been on your mind?",
            "I appreciate you telling me. What would you like to talk about today?",
            "Thanks for sharing. I'm here to listen — what's going on?",
            "Got it. I'm here for you. What's been happening?",
        ],
    },
    "occupation": {
        "weight": 3,
        "messages": [
            "i'm a student", "i'm a college student", "i'm in high school",
            "i'm a teacher", "i'm a nurse", "i'm a programmer", "i'm a lawyer",
            "i work at a cafe", "i'm unemployed", "i'm a stay at home parent",
            "i'm a freelancer", "i'm a musician", "i'm an artist", "i'm a writer",
            "i'm in middle school", "i'm in grad school", "i'm doing my masters",
            "i just started a new job", "i work in tech", "i'm a barista",
        ],
        "responses": [
            "That's interesting, {name}. How's that going for you?",
            "Thanks for sharing. What's that been like for you?",
            "I see. How are you feeling about your work/school life?",
            "Got it. What's been happening with that?",
        ],
    },
    "small_talk": {
        "weight": 7,
        "messages": [
            "how's your day", "what's up", "thanks", "thank you", "bye",
            "see you", "how are you doing", "what's new", "not much",
            "just chilling", "just relaxing", "same old", "not a lot",
            "how's it going", "how have you been", "what's been up",
            "how's life", "how's everything", "what are you up to",
            "nothing really", "just hanging out", "bored tbh",
            "just me being me", "living the dream", "surviving",
            "hanging in there", "could be better", "can't complain",
            "same stuff different day", "you know how it is",
        ],
        "responses": [
            "I'm here. What's been going on with you?",
            "Good to hear from you. What's been on your mind?",
            "Thanks for checking in. How are you really doing?",
            "I'm doing well, thanks. But I'm more interested in how you're doing.",
            "Glad you're here. What would you like to talk about?",
            "I'm here whenever you're ready. What's happening?",
        ],
    },
    "casual_question": {
        "weight": 5,
        "messages": [
            "tell me a joke", "what's your favorite color", "do you like music",
            "what's your purpose", "who made you", "are you a real person",
            "are you human", "what do you think about", "do you have feelings",
            "what's your name", "how old are you", "where are you from",
            "what's the weather like", "can you help me", "what can you do",
            "do you sleep", "do you eat", "what's your favorite food",
            "do you watch tv", "what's your favorite movie", "sing me a song",
            "tell me a story", "what's the meaning of life", "do you believe in god",
            "are you smarter than chatgpt", "what ai are you", "how do you work",
        ],
        "responses": [
            "That's a fun question! But I'm more curious about how you're doing. What's on your mind?",
            "Ha, good question. But let's talk about you — what's been going on?",
            "I appreciate the curiosity! But I'm here for you. What's happening in your life?",
            "Interesting question. I'm here to listen though — what's on your heart?",
            "I'm flattered you asked! But I'd rather hear about you. What's going on?",
            "That's a creative question! What really matters though is how you're feeling. What's up?",
        ],
    },
    "anxiety": {
        "weight": 10,
        "messages": [
            "i'm feeling anxious", "i have anxiety", "i'm so worried",
            "my heart is racing", "i can't stop worrying", "i feel nervous",
            "i'm panicking", "i feel scared for no reason", "i have a panic attack",
            "i can't breathe", "i feel like something bad is going to happen",
            "i'm terrified", "i feel dread", "i can't stop overthinking",
            "my chest feels tight", "i feel on edge", "i'm restless",
            "i feel like the walls are closing in", "i'm hyperventilating",
            "i feel like i'm losing control", "i'm shaking",
            "i can't calm down", "i feel so anxious rn",
            "anxiety is killing me", "i'm freaking out", "i'm spiraling",
            "my mind won't stop racing", "i feel so tense",
            "i'm scared of everything", "i feel unsafe",
            "i'm having intrusive thoughts", "i can't stop worrying about everything",
            "i feel like something terrible is about to happen",
            "i'm anxious about everything", "i feel paralyzed by fear",
            "i'm afraid to leave my house", "social situations terrify me",
            "i'm scared of people judging me", "i overthink every conversation",
            "my stomach is in knots from worry", "i feel butterflies but bad ones",
        ],
        "responses": [
            "It sounds like there's a lot of weight in what you're describing. What does that anxiety feel like in your body right now?",
            "I hear you. That kind of worry can feel overwhelming. When does it tend to show up most?",
            "That sounds really tough. I wonder if this has been sitting with you for a while now.",
            "I'm here with you. What's the first thing that comes to mind when you think about what's worrying you?",
            "It makes sense you'd feel that way. What part of this feels the hardest to sit with?",
            "I can hear how much this is affecting you. What's been triggering these feelings?",
            "Anxiety is your body's way of trying to protect you, even when it overdoes it. What's going on?",
            "That worry sounds really present for you right now. What's at the center of it?",
        ],
    },
    "sadness": {
        "weight": 10,
        "messages": [
            "i'm feeling sad", "i feel empty", "i'm so depressed",
            "nothing makes me happy anymore", "i've been crying a lot",
            "i feel hopeless", "i feel alone", "i'm so lonely",
            "i feel like giving up", "i don't see the point of anything",
            "i feel broken", "i miss how things used to be",
            "i feel like a burden", "everything feels pointless",
            "i'm tired of being sad", "i feel so low",
            "i can't remember the last time i was happy",
            "i feel disconnected from everyone", "i'm grieving",
            "i lost someone and i don't know how to cope",
            "i feel numb", "my heart hurts", "i feel so empty inside",
            "i feel like nobody cares", "i'm stuck in a dark place",
            "i feel like i'm drowning", "i just want the pain to stop",
            "i feel like i'm watching my life from afar",
            "i feel invisible", "i miss being happy",
            "i feel like nothing will ever get better",
            "i'm mourning a relationship that ended",
            "i feel so tired of pretending to be okay",
            "i feel like my life has no meaning",
        ],
        "responses": [
            "It sounds like there's a heaviness that's hard to put into words. I'm here with you.",
            "I hear you. That kind of sadness sounds like it runs deep. When did it start?",
            "I'm sorry you're going through this. What does that sadness need you to understand about it?",
            "It makes sense you'd feel that way. Sadness is a natural response when something matters to us.",
            "That sounds really painful. What's the situation that feels most connected to this?",
            "I'm here with you. We can sit with this for a while. What's weighing on you most?",
            "I can hear how much this is affecting you. Is there something specific that triggered this?",
            "I wonder if this has been weighing on you more than you let on. What's going on?",
        ],
    },
    "anger": {
        "weight": 8,
        "messages": [
            "i'm so angry", "i feel furious", "i want to scream",
            "i can't control my temper", "i'm mad at the world",
            "everything makes me mad", "i'm so frustrated",
            "i feel rage building up", "i keep snapping at people",
            "i can't let go of my anger", "i feel like punching something",
            "i'm tired of being taken for granted", "i feel betrayed",
            "i'm furious about how i've been treated", "i can't forgive what they did",
            "i feel like everyone is against me", "i'm so irritated",
            "i keep getting into arguments", "i feel hostile",
            "i'm angry at myself", "small things set me off",
            "i feel like my anger is destroying everything",
            "i can't stop feeling bitter", "i'm tired of being walked all over",
            "i feel so much hatred inside", "i'm mad because nobody listens",
        ],
        "responses": [
            "It sounds like something really got to you. What happened?",
            "I wonder if there's more underneath that frustration. What sparked this?",
            "That kind of anger usually points to something that matters deeply. What's going on?",
            "It seems like this touched something important. What happened that sparked this?",
            "Anyone might feel that way in your position. What would need to happen for you to feel heard?",
            "I hear you. Anger often shows us when something we care about has been crossed. What happened?",
            "That sounds really frustrating. When did you first notice this building up?",
            "I'm here with you. Let that sit for a moment. What's at the root of this?",
        ],
    },
    "stress": {
        "weight": 9,
        "messages": [
            "i'm so stressed", "i'm overwhelmed", "i have too much to handle",
            "i can't keep up", "i'm drowning in responsibilities",
            "i feel like i'm going to crash", "i'm under so much pressure",
            "i can't catch a break", "i'm exhausted from stress",
            "i feel like i'm carrying the world", "i'm burned out",
            "i feel like i'm about to break", "i can't handle everything",
            "i'm so busy i can't breathe", "i'm always behind",
            "i feel like i'm spreading myself too thin",
            "i'm stressed about money", "i have too many deadlines",
            "i can't relax even when i have time", "i feel like i'm running on empty",
            "i'm overwhelmed by work", "i'm stressed about school",
            "i feel like i'm one step from a breakdown",
            "i'm so tired of this pressure", "i feel like i'm failing at everything",
            "there aren't enough hours in the day", "i'm overwhelmed by expectations",
        ],
        "responses": [
            "It sounds like you're carrying a lot right now. What part feels the heaviest?",
            "I wonder if this has been building for a while. What's pulling at your attention most?",
            "That kind of pressure sounds exhausting. If you could set one thing down today, what would it be?",
            "It seems like there's a lot pulling at your attention. Anyone would feel stretched under that load.",
            "I can hear how overwhelmed you are. What does your body feel like when you think about everything?",
            "It makes sense to feel overwhelmed with everything on your plate. What's the most urgent thing?",
            "I'm here with you. We can slow down here. What's been the hardest to navigate?",
            "That sounds like a lot. What would help you feel even a little lighter right now?",
        ],
    },
    "sleep": {
        "weight": 7,
        "messages": [
            "i can't sleep", "i have insomnia", "i keep waking up at night",
            "i haven't slept well in weeks", "i feel exhausted but can't sleep",
            "i have nightmares", "i lie awake for hours",
            "i can't shut off my brain at bedtime", "i'm tired all the time",
            "i wake up feeling more tired", "i toss and turn all night",
            "i need sleeping pills to rest", "i wake up at 3am",
            "i can't sleep through the night", "i dread going to bed",
            "i feel groggy all day", "my sleep schedule is broken",
            "i stay up late even though i need rest",
            "i grind my teeth at night", "i have chronic insomnia",
            "my mind races at night", "i can't nap even when exhausted",
            "i'm afraid to sleep because of nightmares",
            "i wake up with headaches from poor sleep",
        ],
        "responses": [
            "It sounds like your mind doesn't quiet down when you need it to. What's going through your mind when you're lying awake?",
            "I wonder if there's something your mind is trying to process. How long has this been going on?",
            "That struggle to sleep sounds really draining. What's your evening like before you try to sleep?",
            "It makes sense that would leave you feeling drained. Is it hard to fall asleep or hard to stay asleep?",
            "Sleep struggles often reflect how full our minds are during the day. What's been on your mind lately?",
            "I'm here with you. Take your time. What does your bedtime routine look like?",
            "I imagine lying awake with your thoughts is exhausting. What tends to keep you up?",
            "That sounds really draining. What's the first thought that hits when you try to sleep?",
        ],
    },
    "relationships": {
        "weight": 9,
        "messages": [
            "my partner and i are fighting", "i feel unloved",
            "we keep arguing about everything", "my boyfriend broke up with me",
            "my girlfriend cheated on me", "i feel trapped in my relationship",
            "we don't communicate anymore", "i feel like i'm the only one trying",
            "my partner doesn't understand me", "i'm scared of losing them",
            "we've grown apart", "i feel like it's one-sided",
            "my partner won't listen", "i keep having the same arguments",
            "i can't trust my partner", "my friends don't care about me",
            "i feel distant from my family", "i don't know how to fix this",
            "i'm walking on eggshells", "my partner doesn't appreciate me",
            "i feel lonely in my relationship", "we have nothing in common",
            "my partner is always criticizing me", "i think they're hiding something",
            "i feel taken for granted", "i don't feel heard",
            "my partner and i want different things", "i caught my partner cheating",
            "i'm going through a breakup", "i feel betrayed",
            "my family doesn't approve of my partner", "i have trust issues",
            "i feel rejected", "my heart is broken",
            "i'm scared to open up after being hurt",
            "my partner invalidates my feelings", "i feel controlled",
        ],
        "responses": [
            "It sounds like there's something shifting in a relationship that matters to you. What's going on?",
            "I wonder if this has been on your mind for a while. What does your gut tell you?",
            "That kind of tension is hard to carry. What do you find yourself needing that you're not getting?",
            "It seems like this connection means a lot to you. What kind of conversation have you been wanting to have?",
            "I hear you. That sounds painful. How long has this dynamic been building?",
            "It makes sense that would hurt. Relationships touch such a deep part of us. What's been happening?",
            "I'm here with you. Take your time with this. What feels most important about this to you?",
            "That sounds really difficult. What would need to change for you to feel safe in this?",
        ],
    },
    "school": {
        "weight": 7,
        "messages": [
            "i failed my exam", "i'm scared of college", "i got bullied at school",
            "i'm failing my classes", "i can't keep up with schoolwork",
            "my teacher hates me", "i'm stressed about grades",
            "i got rejected from my dream school", "i don't fit in at school",
            "i'm being bullied online", "school is overwhelming",
            "i have too many assignments", "i'm scared of failing",
            "i don't understand my classes", "my classmates exclude me",
            "i'm failing math", "i got a bad grade", "i'm scared of presentations",
            "i got into a fight at school", "i hate school",
            "college applications are stressing me out", "i'm scared of the future",
            "my parents pressure me about grades", "i feel stupid compared to others",
            "i got suspended", "i'm being homeschooled and i hate it",
            "i can't focus in class", "i'm behind on everything",
        ],
        "responses": [
            "It sounds like school has been really tough for you. What part feels the hardest?",
            "I wonder if this has been weighing on you. What's been going on?",
            "That sounds stressful. How are you handling all of this?",
            "I hear you. What does your gut tell you about this situation?",
            "It makes sense to feel overwhelmed. What's been the most difficult part?",
            "I'm here with you. What would help you feel more supported right now?",
            "That sounds really frustrating. When did you first notice this becoming a problem?",
            "I can hear how much this is affecting you. What's been the hardest thing to navigate?",
        ],
    },
    "work": {
        "weight": 7,
        "messages": [
            "i hate my job", "my boss yelled at me", "i'm burnt out",
            "i feel stuck in my career", "i'm not satisfied with my work",
            "my coworkers are toxic", "i feel undervalued",
            "i'm bored with my job", "i'm not progressing",
            "i dread going to work", "my job is making me miserable",
            "i'm underpaid and overworked", "i don't know what career to pick",
            "i feel like i'm wasting my potential", "i'm scared of losing my job",
            "the work pressure is too much", "my work has no meaning",
            "i'm having problems with colleagues", "i need a career change",
            "my work-life balance is terrible", "i feel like i'm not good enough",
            "i keep getting passed over for promotions", "i can't concentrate at work",
            "my workplace is toxic", "i feel disrespected at work",
            "i was laid off and feel lost", "i'm burned out from work",
            "i struggle with imposter syndrome", "i feel anxious every sunday night",
        ],
        "responses": [
            "It sounds like work has been taking a lot out of you. What part drains you the most?",
            "I wonder if this has been building over time. What would a better version of your work life look like?",
            "It seems like there's a mismatch between what you give and what you get. Is it the work itself or the environment?",
            "That kind of work stress can affect everything else too. What's been the hardest thing to navigate?",
            "I imagine that leaves you questioning things. What's pulling at your attention most?",
            "It makes sense that would weigh on you. What's been going on specifically?",
            "I'm here with you. Take your time. What would need to change for you to feel better about this?",
            "That sounds exhausting. What does your ideal work situation look like?",
        ],
    },
    "family": {
        "weight": 7,
        "messages": [
            "my parents fight all the time", "i don't get along with my dad",
            "my mom is sick", "my parents are getting divorced",
            "i feel like my family doesn't understand me",
            "my sibling and i don't talk anymore", "my parents pressure me too much",
            "i feel like i have to be perfect for my family",
            "my family is falling apart", "my parents are always fighting",
            "i don't feel supported by my family", "my dad is never around",
            "my mom is always on my case", "my family is dysfunctional",
            "i feel like the black sheep", "my parents don't listen to me",
            "my brother is going through a hard time", "my sister won't talk to me",
            "i feel like i have to take care of everyone", "my family treats me differently",
            "i feel guilty for wanting space from my family",
            "my parents compare me to my siblings", "i feel like i'm not enough for them",
            "my family is going through a tough time", "i miss how we used to be",
        ],
        "responses": [
            "It sounds like there's a lot of tension in your family right now. What's been going on?",
            "I wonder if this has been building for a while. What does your gut tell you about this?",
            "That sounds really hard. Family dynamics can be so complex. What's weighing on you most?",
            "I hear you. What do you find yourself needing from them that you're not getting?",
            "It makes sense that would hurt. What's been the hardest part of this for you?",
            "I'm here with you. Family stuff can cut deep. What's been happening?",
            "That sounds painful. How long has this been going on?",
            "I can hear how much this affects you. What would help you feel more at peace with this?",
        ],
    },
    "health": {
        "weight": 6,
        "messages": [
            "i can't sleep", "i haven't been eating", "i'm always tired",
            "i feel physically sick", "i have chronic pain",
            "i'm always exhausted", "my body hurts all the time",
            "i'm scared about my health", "i got diagnosed with something",
            "i'm not taking care of myself", "i feel weak all the time",
            "i have headaches every day", "i can't stop eating",
            "i'm losing weight without trying", "i feel dizzy all the time",
            "i haven't exercised in months", "my mental health is affecting my body",
            "i feel run down", "i'm always sick", "i'm scared of the doctor",
        ],
        "responses": [
            "It sounds like your body has been sending you signals. What's been going on?",
            "I wonder if this has been building for a while. How long have you been feeling this way?",
            "That sounds concerning. What does your gut tell you about this?",
            "I hear you. Health stuff can be really scary. What's been happening?",
            "It makes sense to feel worried about that. Have you been able to talk to anyone about it?",
            "I'm here with you. What feels most important about this right now?",
            "That sounds exhausting. What's been the hardest part of dealing with this?",
            "I can hear how much this is affecting you. What would help you feel more supported?",
        ],
    },
    "positive": {
        "weight": 6,
        "messages": [
            "i feel really happy today", "i'm grateful for everything",
            "i had a really good day", "i'm proud of what i accomplished",
            "i feel hopeful about the future", "i feel at peace",
            "i'm excited about something", "things are getting better",
            "i feel loved and supported", "i'm doing well",
            "i feel strong and capable", "i'm proud of myself",
            "i feel joyful", "i'm grateful to be alive",
            "i feel like i'm growing", "i'm excited about my future",
            "i feel content", "i'm happy with my progress",
            "i feel more confident", "i feel blessed",
            "i'm surrounded by love", "i feel inspired",
            "today was a good day", "i feel alive",
        ],
        "responses": [
            "That's really nice to hear. I'm glad for you. What's contributing to that good feeling?",
            "It sounds like something good is happening. What helped bring this about?",
            "I'm really glad to hear that. What does that feeling make you want to do?",
            "That's wonderful. How does it feel in your body when you experience this?",
            "It seems like things are moving in a good direction. What's been going well?",
            "That kind of positive energy is really nice to hear. What's been the highlight?",
            "I'm happy for you. What's been contributing to this good feeling?",
            "It's good you're noticing that. How does it feel to be in this place right now?",
        ],
    },
    "confusion": {
        "weight": 6,
        "messages": [
            "i don't know what to do with my life", "i feel lost",
            "i don't know who i am anymore", "i'm unsure about my decisions",
            "i can't figure out what i want", "i feel directionless",
            "i don't know what path to take", "i'm confused about my feelings",
            "i don't understand myself", "i feel like i'm at a crossroads",
            "i can't make up my mind", "i feel like i don't know anything",
            "i'm torn between choices", "i don't know what i believe anymore",
            "i feel like i'm in a fog", "i can't think clearly",
            "i'm confused about my identity", "i don't know what matters to me",
            "i'm unsure about my future", "i can't figure out what makes me happy",
            "i feel like i'm wandering without purpose", "i'm confused about my goals",
            "i don't know if i'm making the right choice", "i feel like i'm having an identity crisis",
        ],
        "responses": [
            "It sounds like you're at a place where things aren't clear yet. What feels most unclear?",
            "I wonder if this uncertainty has been unsettling. What's the heart of what you're trying to figure out?",
            "It seems like you're weighing something important. What options are you considering?",
            "That kind of not-knowing is genuinely uncomfortable. What part feels most important to resolve?",
            "I hear you. Confusion often means something important is shifting. What's been on your mind?",
            "It makes sense to feel uncertain right now. What's been the hardest part of not knowing?",
            "I'm here with you. Take your time. What does your gut tell you?",
            "That sounds disorienting. What would clarity look like for you?",
        ],
    },
    "general_vent": {
        "weight": 8,
        "messages": [
            "i just need to vent", "i need someone to talk to",
            "i want to get something off my chest", "there's something i've been thinking about",
            "i've been going through a lot", "i feel like sharing something",
            "i want to understand myself better", "i'm trying to process things",
            "i just need someone to listen", "i've been meaning to talk about this",
            "i want to be more open about my feelings", "i need perspective",
            "i've been noticing patterns in my behavior", "i want to work on myself",
            "i'm here because i want to change", "i need help understanding something",
            "i've been journaling about my thoughts", "i want to be more self-aware",
            "i'm trying to make sense of my life", "i want to check in with myself",
            "there's something weighing on me", "i need to talk through something",
            "i've been avoiding this for a while", "i don't know where to start",
            "i just wanted to talk to someone", "i've been reflecting on my life",
            "i'm not sure why i'm here", "i want to share something with you",
            "i've been doing a lot of thinking", "i want to learn how to cope better",
        ],
        "responses": [
            "I'm glad you're here. What's coming up for you?",
            "I'm listening. What feels most important to share right now?",
            "Take your time. What's on your mind?",
            "I'm here for you. What would you like to talk about?",
            "I appreciate you reaching out. What's been happening?",
            "I'm here. No judgment. What's going on?",
            "Thanks for being willing to open up. What's been weighing on you?",
            "I'm ready to listen. What's on your heart?",
        ],
    },
    "general_followup": {
        "weight": 6,
        "messages": [
            "yeah", "nah", "kinda", "not really", "i guess", "maybe",
            "i don't know", "idk", "not sure", "i dunno", "hmm",
            "true", "that's right", "exactly", "sort of", "pretty much",
            "i mean yeah", "not exactly", "close enough", "you could say that",
            "yeah but no", "kind of", "in a way", "not really no",
            "yeah i guess", "nah not really", "i suppose", "maybe idk",
        ],
        "responses": [
            "That's okay. You don't have to have all the words right now. What's at the surface?",
            "Take your time. There's no rush to figure it all out. What comes to mind first?",
            "That's alright. Sometimes it's hard to put a name to what we feel. What's been on your mind?",
            "No pressure. I'm here whenever you're ready to share more.",
            "I hear you. We can sit here with this. What feels right to say?",
            "It's okay not to know exactly what to say. What's at the surface right now?",
        ],
    },
}

# ============================================================
# AMBIGUOUS STATEMENTS
# ============================================================

AMBIGUOUS_STATEMENTS = [
    # Violence ambiguity
    ("i killed a person", "violence", "hypothetical",
     "Can you help me understand what you mean by that? Are you describing something that actually happened, a hypothetical scenario, or something else?"),
    ("i killed someone in a game", "violence", "gaming",
     "Got it — sounds like you're talking about a game. What else is going on?"),
    ("i killed it at the interview", "violence", "positive",
     "That's great! Sounds like the interview went really well. How did it feel?"),
    ("i killed a spider", "violence", "mundane",
     "Ha, spiders can be scary. Are you okay? What else is going on?"),
    ("i want to disappear", "ambiguity", "crisis_risk",
     "I want to make sure I understand you correctly. When you say you want to disappear, do you mean you need a break from everything, or are you having thoughts of hurting yourself?"),
    ("i want to disappear from everything", "ambiguity", "overwhelm",
     "It sounds like you're carrying a lot. When you say disappear, do you mean you need space, or is something deeper going on?"),
    ("i want to end it", "ambiguity", "crisis_risk",
     "I'm really concerned about what you're saying. When you say end it, can you help me understand what you mean? Are you thinking about hurting yourself?"),
    ("i want to end this conversation", "ambiguity", "casual",
     "No problem at all. Take care of yourself, and come back anytime."),
    ("i want to hurt myself", "self_harm", "crisis",
     "I'm really concerned about you. Please reach out to someone who can help right now:\n- National Suicide Prevention Lifeline: 988\n- Crisis Text Line: Text HOME to 741741\nThere are people ready to support you through this."),
    ("i want to hurt them", "violence", "anger",
     "I hear how angry you are. Before we go further — can you help me understand the context? Are you describing how you feel in the moment, or is this something more serious?"),
    ("i feel like killing myself", "self_harm", "crisis",
     "I'm really concerned about what you're saying. Please reach out to a crisis line now:\n- National Suicide Prevention Lifeline: 988\n- Crisis Text Line: Text HOME to 741741\nThere are people ready to help you through this."),
    ("i feel like i could kill someone", "violence", "hyperbole",
     "That sounds like intense anger. What happened that made you feel this way?"),
    ("i'm going to kill it tomorrow", "violence", "positive",
     "Love the energy! What are you going to crush tomorrow?"),
    ("i robbed a bank", "crime", "fictional",
     "Before I respond — can you help me understand the context? Are you describing something real, or is this a hypothetical or fictional situation?"),
    ("i stole a candy bar", "crime", "mundane",
     "That sounds like something that might be weighing on you. What happened?"),
    ("i'm going to bomb this presentation", "violence", "anxiety",
     "That sounds like presentation anxiety. What's making you nervous about it?"),
    ("i'm going to shoot my shot", "violence", "positive",
     "Go for it! What's the situation?"),
    ("i'm dead inside", "ambiguity", "emotional",
     "That sounds like a deep emptiness. I'm here with you. What's been going on?"),
    ("i'm dying", "ambiguity", "hyperbole",
     "That sounds like you're going through something intense. What's happening?"),
    ("i'm literally dying", "ambiguity", "hyperbole",
     "I hear you. What's been going on that makes you feel this way?"),
    ("i want to die", "self_harm", "crisis",
     "I'm really concerned about what you're saying. Please reach out to a crisis line now:\n- National Suicide Prevention Lifeline: 988\n- Crisis Text Line: Text HOME to 741741\nThere are people ready to help you through this."),
    ("i wish i was dead", "self_harm", "crisis",
     "I'm really concerned about you. Please reach out to someone who can help right now:\n- National Suicide Prevention Lifeline: 988\n- Crisis Text Line: Text HOME to 741741"),
    ("i'm so ugly i should die", "self_harm", "crisis",
     "I'm really sorry you're feeling this way about yourself. You don't deserve to feel that way. Please reach out:\n- National Suicide Prevention Lifeline: 988\n- Crisis Text Line: Text HOME to 741741"),
    ("everyone would be better off without me", "self_harm", "crisis",
     "I hear you, and I want you to know that's not true. Please reach out to someone who can help:\n- National Suicide Prevention Lifeline: 988\n- Crisis Text Line: Text HOME to 741741"),
    ("i joined a gang", "crime", "disclosure",
     "Before I respond — can you help me understand the context? Is this something that actually happened, or are you describing a situation you're in?"),
    ("i've been selling drugs", "crime", "disclosure",
     "I want to make sure I understand you correctly. Are you describing something that's happening in your life right now?"),
    ("i hacked someone's account", "crime", "disclosure",
     "Before I respond — can you help me understand the context? Is this something you actually did, or are you talking about a hypothetical?"),
    ("i burned down a building", "violence", "fictional",
     "Can you help me understand what you mean by that? Are you describing something that actually happened, or is this a hypothetical scenario?"),
    ("i tortured an animal", "violence", "disclosure",
     "I want to make sure I understand you correctly. Can you tell me more about what happened?"),
    ("i raped someone", "violence", "disclosure",
     "I need to take a moment here. What you're describing is extremely serious. If this actually happened, I strongly encourage you to speak with a professional who can help you understand the gravity of this situation."),
    ("i got raped", "violence", "disclosure",
     "I'm so sorry that happened to you. That is never okay. Please reach out for support:\n- RAINN: 1-800-656-4673\n- Crisis Text Line: Text HOME to 741741\nYou don't have to go through this alone."),
    ("i'm being abused", "violence", "disclosure",
     "I'm sorry you're going through that. You don't deserve to be hurt. Please reach out for help:\n- National Domestic Violence Hotline: 1-800-799-7233\n- Crisis Text Line: Text HOME to 741741"),
    ("i want to kill my boss", "violence", "hyperbole",
     "That sounds like intense frustration at work. What happened that's making you feel this way?"),
    ("i want to strangle my neighbor", "violence", "hyperbole",
     "It sounds like your neighbor is really getting to you. What's been going on?"),
    ("i bombed the test", "violence", "academic",
     "That sounds frustrating. How are you feeling about the results?"),
    ("i'm going to beat them up", "violence", "hyperbole",
     "That sounds like a lot of anger. What happened that's making you feel this way?"),
]

# ============================================================
# SLANG VARIATIONS
# ============================================================

SLANG_TRANSFORMS = [
    ("i'm", ["im", "im"]),
    ("i am", ["im", "i'm"]),
    ("you are", ["ur", "youre"]),
    ("your", ["ur"]),
    ("really", ["fr", "frfr", "deadass"]),
    ("very", ["lowkey", "highkey"]),
    ("right now", ["rn", "atm"]),
    ("for real", ["fr", "frfr", "no cap"]),
    ("not going to lie", ["ngl"]),
    ("you know", ["yk"]),
    ("alright", ["alr"]),
    ("okay", ["bet", "say less"]),
    ("no", ["nah"]),
    ("yes", ["ye", "yuh"]),
    ("friend", ["bro", "bruh", "dawg"]),
    ("what the", ["tf", "wtf"]),
    ("laughing", ["lmao", "lol"]),
    ("i'm in trouble", ["im cooked", "im cooked fr"]),
    ("i'm panicking", ["im tweaking"]),
    ("it's over", ["its wraps"]),
    ("we'll be fine", ["we ball"]),
    ("i failed", ["i sold", "i fumbled"]),
    ("stopped contacting", ["ghosted"]),
    ("didn't reply", ["left me on delivered", "left on read"]),
    ("rejected me", ["friend zoned me"]),
    ("in a difficult situation", ["down bad"]),
    ("losing control", ["spiraling", "crashing out"]),
    ("seriously", ["deadass"]),
]

SLANG_EMOJIS = ["😭", "💀", "🙏", "❤️", "💔", "🥲", "😔", "😕", "🙂", "😊", "😤", "🥺", "💀💀", "😭😭", "😔😔", "🔥", "💯", "✨", "😭😭😭"]

SLANG_SUFFIXES = [
    " lol", " lmao", " 💀", " 😭", " fr", " ngl", " no cap", " tbh",
    " imo", " idk", " rn", " atm", " btw", " smh", " istg", " fml",
    " 🙏", " ❤️", " 💔", " ✨", " 🔥", " 💯", " bruh", " 😤",
]

# ============================================================
# MULTI-TURN CONVERSATION TEMPLATES
# ============================================================

MULTI_TURN_TEMPLATES = [
    # Anxiety progression
    {
        "intent_chain": ["greeting", "introduction", "anxiety"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi there", "hello", "yo"]},
            {"intent": "introduction", "templates": ["my name is {name}", "i'm {name}", "call me {name}"]},
            {"intent": "anxiety", "templates": [
                "i've been feeling really anxious lately",
                "my anxiety has been so bad recently",
                "i can't stop worrying about everything",
                "i feel like something bad is going to happen",
                "my heart keeps racing and i don't know why",
            ]},
        ],
    },
    # Sadness after breakup
    {
        "intent_chain": ["greeting", "relationships", "sadness"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "relationships", "templates": [
                "my girlfriend broke up with me",
                "i just went through a breakup",
                "she left me and i don't know why",
            ]},
            {"intent": "sadness", "templates": [
                "i feel so empty without her",
                "i can't stop crying",
                "i feel like i'll never be happy again",
            ]},
        ],
    },
    # Work stress building
    {
        "intent_chain": ["greeting", "work", "stress", "anxiety"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "work", "templates": [
                "i hate my job so much",
                "my boss is impossible to work with",
                "work has been so stressful lately",
            ]},
            {"intent": "stress", "templates": [
                "i feel like i'm going to crash",
                "i can't handle all this pressure",
                "i'm so overwhelmed with everything",
            ]},
            {"intent": "anxiety", "templates": [
                "i'm scared i'm going to get fired",
                "i feel anxious every sunday night thinking about monday",
                "what if i can't handle this anymore",
            ]},
        ],
    },
    # Family issues
    {
        "intent_chain": ["greeting", "family", "sadness"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "family", "templates": [
                "my parents are getting divorced",
                "my family is falling apart",
                "my mom and dad won't stop fighting",
            ]},
            {"intent": "sadness", "templates": [
                "i feel so helpless watching this happen",
                "i miss how we used to be",
                "i feel like it's my fault somehow",
            ]},
        ],
    },
    # School bullying
    {
        "intent_chain": ["greeting", "school", "anger", "sadness"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "school", "templates": [
                "i got bullied at school today",
                "people keep making fun of me",
                "i hate going to school",
            ]},
            {"intent": "anger", "templates": [
                "i'm so angry at them",
                "i want to scream at them",
                "why do they keep doing this to me",
            ]},
            {"intent": "sadness", "templates": [
                "i feel so alone in this",
                "nobody stands up for me",
                "i feel like giving up",
            ]},
        ],
    },
    # Positive progression
    {
        "intent_chain": ["greeting", "positive", "positive"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "positive", "templates": [
                "i got the job!",
                "things have been going really well",
                "i feel really good today",
            ]},
            {"intent": "positive", "templates": [
                "i'm so grateful for this opportunity",
                "i feel like things are finally turning around",
                "i'm excited about the future",
            ]},
        ],
    },
    # Context memory - name recall
    {
        "intent_chain": ["introduction", "general_vent", "general_followup"],
        "turns": [
            {"intent": "introduction", "templates": ["my name is {name}", "i'm {name}"]},
            {"intent": "general_vent", "templates": [
                "i've been going through a lot lately",
                "there's something weighing on me",
                "i need someone to talk to",
            ]},
            {"intent": "general_followup", "templates": ["yeah", "kinda", "i guess", "not really"]},
        ],
    },
    # Crisis escalation
    {
        "intent_chain": ["greeting", "sadness", "crisis"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "sadness", "templates": [
                "i feel so empty inside",
                "nothing makes me happy anymore",
                "i feel like giving up",
            ]},
            {"intent": "crisis", "templates": [
                "i want to die",
                "i don't want to be here anymore",
                "i feel like everyone would be better off without me",
            ]},
        ],
    },
    # Sleep and stress
    {
        "intent_chain": ["greeting", "sleep", "stress"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "sleep", "templates": [
                "i can't sleep at all",
                "i've been having insomnia for weeks",
                "my mind won't shut off at night",
            ]},
            {"intent": "stress", "templates": [
                "i think the stress is making it worse",
                "i have too much on my plate",
                "i feel like i'm drowning",
            ]},
        ],
    },
    # Health anxiety
    {
        "intent_chain": ["greeting", "health", "anxiety"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "health", "templates": [
                "i've been feeling sick a lot lately",
                "my body hurts all the time",
                "i'm always exhausted",
            ]},
            {"intent": "anxiety", "templates": [
                "i'm scared something is seriously wrong",
                "i keep googling my symptoms",
                "what if it's something bad",
            ]},
        ],
    },
    # Confusion about life
    {
        "intent_chain": ["greeting", "confusion", "general_vent"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "confusion", "templates": [
                "i don't know what to do with my life",
                "i feel so lost",
                "i have no idea what i want",
            ]},
            {"intent": "general_vent", "templates": [
                "everyone around me seems to have it figured out",
                "i feel like i'm wasting my time",
                "i don't know where to start",
            ]},
        ],
    },
    # Relationship doubts
    {
        "intent_chain": ["greeting", "relationships", "confusion"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "relationships", "templates": [
                "i'm not sure about my relationship anymore",
                "my partner and i keep fighting",
                "i feel like we're growing apart",
            ]},
            {"intent": "confusion", "templates": [
                "i don't know if i should stay or leave",
                "i'm confused about what i want",
                "i don't know what's right anymore",
            ]},
        ],
    },
    # Identity exploration
    {
        "intent_chain": ["greeting", "identity", "confusion"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "identity", "templates": [
                "i'm nonbinary and use they/them pronouns",
                "i'm figuring out my gender identity",
                "i don't feel like i fit into any box",
            ]},
            {"intent": "confusion", "templates": [
                "i don't know who i really am",
                "everyone has an opinion about who i should be",
                "i feel lost trying to figure myself out",
            ]},
        ],
    },
    # Casual check-in then deeper
    {
        "intent_chain": ["small_talk", "small_talk", "anxiety"],
        "turns": [
            {"intent": "small_talk", "templates": ["hey", "what's up", "how's it going"]},
            {"intent": "small_talk", "templates": ["not much just chilling", "same old", "surviving"]},
            {"intent": "anxiety", "templates": [
                "actually ngl i've been really anxious",
                "lowkey i'm kinda freaking out",
                "fr i can't stop worrying about stuff",
            ]},
        ],
    },
    # Anger at partner then sadness
    {
        "intent_chain": ["greeting", "anger", "sadness"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "anger", "templates": [
                "i'm so mad at my partner right now",
                "they keep doing things that hurt me",
                "i can't believe what they said to me",
            ]},
            {"intent": "sadness", "templates": [
                "but mostly i just feel hurt",
                "underneath the anger i feel so sad",
                "i feel like i don't matter to them",
            ]},
        ],
    },
    # Positive then confusion
    {
        "intent_chain": ["positive", "confusion"],
        "turns": [
            {"intent": "positive", "templates": [
                "i got promoted at work!",
                "things have been going really well",
            ]},
            {"intent": "confusion", "templates": [
                "but i don't know if i'm happy",
                "i feel weirdly empty about it",
                "i expected to feel different",
            ]},
        ],
    },
    # Sleep issues then family
    {
        "intent_chain": ["greeting", "sleep", "family"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "sleep", "templates": [
                "i can't sleep because my mind is racing",
                "i lie awake thinking about everything",
            ]},
            {"intent": "family", "templates": [
                "i think it's because of stuff with my family",
                "my parents keep fighting and it keeps me up",
                "i can't stop worrying about my family",
            ]},
        ],
    },
    # Work then relationships
    {
        "intent_chain": ["greeting", "work", "relationships"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi"]},
            {"intent": "work", "templates": [
                "work has been so stressful lately",
                "i'm burned out from my job",
            ]},
            {"intent": "relationships", "templates": [
                "and it's affecting my relationship",
                "my partner doesn't understand how stressed i am",
                "we keep fighting because of it",
            ]},
        ],
    },
    # Introduction first, then open up
    {
        "intent_chain": ["introduction", "general_vent"],
        "turns": [
            {"intent": "introduction", "templates": [
                "my name is {name}", "i'm {name}", "call me {name}",
                "you can call me {name}", "hey i'm {name}", "hi im {name}",
                "my name's {name}", "im {name} btw",
            ]},
            {"intent": "general_vent", "templates": [
                "i've been going through a lot lately",
                "there's something on my mind",
                "i need someone to talk to",
                "i've been feeling kind of down",
                "i wanted to share something with you",
            ]},
        ],
    },
    # Introduction, age, then issue
    {
        "intent_chain": ["introduction", "age_sharing", "anxiety"],
        "turns": [
            {"intent": "introduction", "templates": ["my name is {name}", "i'm {name}"]},
            {"intent": "age_sharing", "templates": ["i'm {age} years old", "im {age}", "im {age} btw"]},
            {"intent": "anxiety", "templates": [
                "i've been really anxious lately",
                "my anxiety has been terrible",
                "i can't stop worrying about everything",
            ]},
        ],
    },
    # Introduction, identity, then confusion
    {
        "intent_chain": ["introduction", "identity", "confusion"],
        "turns": [
            {"intent": "introduction", "templates": ["im {name}", "my name is {name}"]},
            {"intent": "identity", "templates": [
                "im nonbinary and use they/them pronouns",
                "im figuring out my gender identity",
                "i use she/her pronouns",
            ]},
            {"intent": "confusion", "templates": [
                "i feel so lost trying to figure myself out",
                "everyone has opinions about who i should be",
                "i don't know who i really am",
            ]},
        ],
    },
    # Introduction, occupation, then stress
    {
        "intent_chain": ["introduction", "occupation", "stress"],
        "turns": [
            {"intent": "introduction", "templates": ["im {name}", "my name is {name}", "call me {name}"]},
            {"intent": "occupation", "templates": [
                "im a college student", "im a teacher", "im in high school",
                "i just started a new job", "im a programmer",
            ]},
            {"intent": "stress", "templates": [
                "and im so stressed about everything",
                "the pressure is overwhelming",
                "i feel like im drowning",
            ]},
        ],
    },
    # Just introduction, nothing else
    {
        "intent_chain": ["introduction"],
        "turns": [
            {"intent": "introduction", "templates": [
                "my name is {name}", "im {name}", "call me {name}",
                "you can call me {name}", "hey im {name}", "hi im {name}",
                "my name's {name}", "im {name} btw", "sup im {name}",
                "yo im {name}", "hello my name is {name}",
                "hey my name's {name}", "btw my name is {name}",
                "hi there im {name}", "just wanted to say hi im {name}",
            ]},
        ],
    },
    # Introduction with casual greeting
    {
        "intent_chain": ["greeting", "introduction"],
        "turns": [
            {"intent": "greeting", "templates": ["hey", "hi", "hello", "yo", "sup"]},
            {"intent": "introduction", "templates": [
                "im {name}", "my name is {name}", "call me {name}",
                "you can call me {name}", "btw im {name}",
            ]},
        ],
    },
    # Introduction then positive
    {
        "intent_chain": ["introduction", "positive"],
        "turns": [
            {"intent": "introduction", "templates": ["im {name}", "my name is {name}"]},
            {"intent": "positive", "templates": [
                "i feel really happy today",
                "things have been going well",
                "i just wanted to share some good news",
                "im feeling grateful today",
            ]},
        ],
    },
    # Introduction then relationships
    {
        "intent_chain": ["introduction", "relationships"],
        "turns": [
            {"intent": "introduction", "templates": ["im {name}", "my name is {name}", "call me {name}"]},
            {"intent": "relationships", "templates": [
                "my partner and i are having issues",
                "i feel like my relationship is falling apart",
                "my girlfriend broke up with me",
                "i miss someone and it hurts",
            ]},
        ],
    },
    # Introduction then sleep
    {
        "intent_chain": ["introduction", "sleep"],
        "turns": [
            {"intent": "introduction", "templates": ["im {name}", "my name is {name}"]},
            {"intent": "sleep", "templates": [
                "i cant sleep at night",
                "ive been having insomnia",
                "my mind wont shut off",
            ]},
        ],
    },
    # Introduction then family
    {
        "intent_chain": ["introduction", "family"],
        "turns": [
            {"intent": "introduction", "templates": ["im {name}", "my name is {name}"]},
            {"intent": "family", "templates": [
                "my parents are always fighting",
                "i dont get along with my dad",
                "my family is going through a hard time",
            ]},
        ],
    },
]

# ============================================================
# RESPONSE STYLE VARIATIONS
# ============================================================

REFLECTION_STARTS = {
    "anxiety": [
        "It sounds like there's a lot of weight in what you're describing.",
        "I wonder if this has been sitting with you for a while now.",
        "That worry sounds really present for you right now.",
        "It seems like your mind is working hard to keep you alert.",
        "I imagine that kind of unease is hard to shake off.",
        "I can hear the tension in what you're sharing.",
        "That sounds like a lot to carry.",
    ],
    "sadness": [
        "It sounds like there's a heaviness that's hard to put into words.",
        "I wonder if this has been weighing on you more than you let on.",
        "It seems like there's a deep ache in what you're sharing.",
        "That kind of sadness sounds like it runs deep.",
        "I imagine that leaves you feeling pretty drained.",
        "I can hear how much pain you're in.",
        "That sounds really heavy.",
    ],
    "anger": [
        "It sounds like something really got to you.",
        "I wonder if there's more underneath that frustration.",
        "It seems like this touched something important.",
        "That kind of anger usually points to something that matters deeply.",
        "I imagine that left you feeling pretty unheard.",
        "I can hear the intensity of what you're feeling.",
        "That sounds really frustrating.",
    ],
    "stress": [
        "It sounds like you're carrying a lot right now.",
        "I wonder if this has been building for a while.",
        "It seems like there's a lot pulling at your attention.",
        "That kind of pressure sounds exhausting.",
        "I imagine it's hard to find a moment to breathe.",
        "I can hear how stretched thin you are.",
        "That sounds overwhelming.",
    ],
    "sleep": [
        "It sounds like your mind doesn't quiet down when you need it to.",
        "I wonder if there's something your mind is trying to process.",
        "It seems like rest has been hard to come by.",
        "That struggle to sleep sounds really draining.",
        "I imagine lying awake with your thoughts is exhausting.",
        "I can hear how drained you are.",
        "That sounds really tiring.",
    ],
    "relationships": [
        "It sounds like there's something shifting in a relationship that matters to you.",
        "I wonder if this has been on your mind for a while.",
        "It seems like this connection means a lot to you.",
        "That kind of tension in a relationship is hard to carry.",
        "I imagine that leaves you feeling pretty torn.",
        "I can hear how much this person means to you.",
        "That sounds painful.",
    ],
    "work": [
        "It sounds like work has been taking a lot out of you.",
        "I wonder if this has been building up over time.",
        "It seems like there's a mismatch between what you give and what you get.",
        "That kind of work stress can affect everything else too.",
        "I imagine that leaves you questioning things.",
        "I can hear how drained you are by work.",
        "That sounds exhausting.",
    ],
    "confusion": [
        "It sounds like you're at a place where things aren't clear yet.",
        "I wonder if this uncertainty has been unsettling.",
        "It seems like you're weighing something important.",
        "That kind of not-knowing is genuinely uncomfortable.",
        "I imagine it's hard to move forward when things feel unclear.",
        "I can hear how disorienting this is.",
        "That sounds really confusing.",
    ],
    "positive": [
        "It sounds like something good is happening for you.",
        "I wonder if this has been a long time coming.",
        "It seems like things are moving in a good direction.",
        "That kind of positive energy is really nice to hear.",
        "I imagine that feels pretty good.",
        "I can hear the joy in what you're sharing.",
        "That sounds wonderful.",
    ],
    "general": [
        "It sounds like there's something on your mind.",
        "I wonder if this has been sitting with you for a while.",
        "It seems like this is important to you.",
        "I appreciate you sharing that with me.",
        "I'm glad you're talking about this.",
        "I can hear that this matters to you.",
        "Thank you for opening up.",
    ],
}

VALIDATIONS = {
    "anxiety": [
        "It makes sense you'd feel that way.",
        "Anxiety is your body's way of trying to protect you, even when it overdoes it.",
        "Anyone might feel that way in your situation.",
    ],
    "sadness": [
        "It makes sense you'd feel that way.",
        "Sadness is a natural response when something matters to us.",
        "You have every right to feel this way.",
    ],
    "anger": [
        "Anyone might feel that way in your position.",
        "Anger often shows us when something we care about has been crossed.",
        "It makes sense you'd be upset about that.",
    ],
    "stress": [
        "Anyone would feel stretched under that load.",
        "It makes sense to feel overwhelmed with everything on your plate.",
        "You're dealing with a lot right now.",
    ],
    "sleep": [
        "It makes sense that would leave you feeling drained.",
        "Sleep struggles often reflect how full our minds are during the day.",
        "That sounds really exhausting.",
    ],
    "relationships": [
        "It makes sense that would hurt.",
        "Relationships touch such a deep part of us, so it's natural for this to affect you.",
        "Anyone would feel that way in your situation.",
    ],
    "work": [
        "It makes sense that would weigh on you.",
        "So much of our identity gets tied up in work, so this kind of frustration is natural.",
        "You have every right to feel frustrated.",
    ],
    "confusion": [
        "It makes sense to feel uncertain right now.",
        "Confusion often means something important is shifting.",
        "Anyone would feel lost in this situation.",
    ],
    "positive": [
        "That's really nice to hear.",
        "It's good you're noticing that.",
        "You deserve to feel good about this.",
    ],
    "general": [""],
}

QUESTIONS = {
    "anxiety": [
        "What does that anxiety feel like in your body right now?",
        "When does that worry tend to show up most?",
        "What's the first thing that comes to mind when you think about what's worrying you?",
        "What part of this feels the hardest to sit with?",
        "What tends to trigger these feelings?",
        "What would help you feel even a little calmer?",
    ],
    "sadness": [
        "When did that heavy feeling first start showing up?",
        "What does that sadness need you to understand about it?",
        "What's the situation that feels most connected to this sadness?",
        "Is there something specific that triggered this, or has it been building?",
        "What would help you feel even a little lighter?",
        "What's been weighing on you the most?",
    ],
    "anger": [
        "What happened that sparked this frustration?",
        "What does this anger want you to protect?",
        "What would need to happen for you to feel heard about this?",
        "When did you first notice this building up?",
        "What's at the root of this anger?",
        "What would help you feel more at peace with this?",
    ],
    "stress": [
        "What part of this feels the heaviest right now?",
        "If you could set one thing down today, what would it be?",
        "What's pulling at your attention most urgently?",
        "What does your body feel like when you think about everything on your plate?",
        "What would help you feel less overwhelmed?",
        "What's been the hardest thing to manage?",
    ],
    "sleep": [
        "What's going through your mind when you're lying awake?",
        "How long has this pattern been going on?",
        "What's your evening like before you try to sleep?",
        "Is it hard to fall asleep, or hard to stay asleep?",
        "What tends to keep you up at night?",
        "What would help you feel more rested?",
    ],
    "relationships": [
        "What do you find yourself needing from them that you're not getting right now?",
        "What kind of conversation have you been wanting to have but haven't yet?",
        "What does your gut tell you about this situation?",
        "How long has this dynamic been building?",
        "What would need to change for you to feel safe in this?",
        "What's the hardest part of this for you?",
    ],
    "work": [
        "What part of your work drains you the most?",
        "What would a better version of your work life look like?",
        "Is it the work itself, or the environment around it?",
        "What's been the hardest thing to navigate?",
        "What would help you feel more fulfilled at work?",
        "What's been the most frustrating part?",
    ],
    "confusion": [
        "What feels most unclear when you sit with it?",
        "What's the heart of what you're trying to figure out?",
        "What options are you weighing right now?",
        "What part of this feels the most important to resolve?",
        "What would clarity look like for you?",
        "What does your gut tell you?",
    ],
    "positive": [
        "What's contributing to that good feeling?",
        "What helped bring this about?",
        "What does that feeling make you want to do?",
        "How does it feel in your body when you experience this?",
        "What's been the highlight?",
        "What's been going well for you?",
    ],
    "general": [
        "What's coming up for you as you share that?",
        "What feels most important about this to you?",
        "How long have you been thinking about this?",
        "What made you decide to bring this up today?",
        "What's been on your mind?",
        "What would you like to explore?",
    ],
}

# ============================================================
# GENERATOR FUNCTIONS
# ============================================================

def pick_name():
    return random.choice(ALL_NAMES)

def pick_age():
    ages = list(range(13, 25)) + list(range(25, 65, 5)) + [16, 17, 18, 19, 20, 21, 22, 23, 24, 30, 35, 40, 45, 50, 55, 60]
    return random.choice(ages)

def add_slang(message, intensity=0.3):
    """Randomly apply slang transformations to a message."""
    if random.random() > intensity:
        return message

    msg = message.lower()

    # Apply slang transforms
    for standard, slang_options in SLANG_TRANSFORMS:
        if random.random() < 0.3 and standard in msg:
            replacement = random.choice(slang_options)
            msg = msg.replace(standard, replacement, 1)

    # Add slang suffix
    if random.random() < 0.25:
        msg += random.choice(SLANG_SUFFIXES)

    # Add emojis
    if random.random() < 0.3:
        msg += " " + random.choice(SLANG_EMOJIS)

    # Random capitalization
    if random.random() < 0.15:
        msg = msg.upper() if random.random() < 0.5 else msg.lower()

    # Add typos
    if random.random() < 0.1:
        msg = add_typo(msg)

    return msg

def add_typo(text):
    """Add a random typo to text."""
    if len(text) < 3:
        return text
    idx = random.randint(0, len(text) - 2)
    chars = list(text)
    # Swap adjacent characters
    if random.random() < 0.5:
        chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
    else:
        # Double a character
        chars.insert(idx, chars[idx])
    return "".join(chars)

def make_entities(name=None, age=None, people=None):
    """Create entities dict."""
    return {
        "name": name or "",
        "age": str(age) if age else "",
        "people": people or [],
    }

def make_memory_update(intent, name=None, age=None, topic=None):
    """Create memory_update dict based on intent."""
    update = {"last_intent": intent}
    if name:
        update["user_name"] = name
    if age:
        update["user_age"] = age
    if topic:
        update["last_topic"] = topic
    return update

def make_quality_check(uses_context=False, answers_question=True):
    return {
        "uses_context": uses_context,
        "answers_question": answers_question,
        "natural_language": True,
        "safe": True,
    }

def generate_single_turn(intent_key, intent_data, name=None, age=None):
    """Generate a single-turn conversation example."""
    template = random.choice(intent_data["messages"])

    # Fill in templates
    user_msg = template
    if "{name}" in user_msg:
        if not name:
            name = pick_name()
        user_msg = user_msg.replace("{name}", name)
    if "{age}" in user_msg:
        if not age:
            age = pick_age()
        user_msg = user_msg.replace("{age}", str(age))

    # Apply slang randomly
    if intent_key not in ["crisis", "greeting"] and random.random() < 0.35:
        user_msg = add_slang(user_msg, intensity=random.uniform(0.2, 0.5))

    # Pick response
    response_templates = intent_data["responses"]
    response = random.choice(response_templates)
    if "{name}" in response:
        if not name:
            name = pick_name()
        response = response.replace("{name}", name)
    if "{age}" in response:
        if not age:
            age = pick_age()
        response = response.replace("{age}", str(age))

    entities = make_entities(name=name, age=age)
    memory = make_memory_update(intent_key, name=name, age=age)

    return {
        "conversation_id": str(uuid.uuid4()),
        "history": [],
        "user_message": user_msg,
        "intent": intent_key,
        "entities": entities,
        "assistant_response": response,
        "memory_update": memory,
        "quality_check": make_quality_check(),
    }

def generate_multi_turn(template_idx=None):
    """Generate a multi-turn conversation example."""
    if template_idx is None:
        template_idx = random.randint(0, len(MULTI_TURN_TEMPLATES) - 1)
    template = MULTI_TURN_TEMPLATES[template_idx]

    name = pick_name()
    age = pick_age()

    history = []
    entities = make_entities(name=name, age=age)
    memory = {}

    turns = template["turns"]
    # Optionally truncate to create shorter conversations
    if len(turns) <= 2:
        num_turns = len(turns)
    else:
        num_turns = random.randint(2, len(turns))
    selected_turns = turns[:num_turns]

    for i, turn in enumerate(selected_turns):
        intent = turn["intent"]
        user_msg = random.choice(turn["templates"])

        # Fill in name
        if "{name}" in user_msg:
            user_msg = user_msg.replace("{name}", name)
        if "{age}" in user_msg:
            user_msg = user_msg.replace("{age}", str(age))

        # Apply slang for later turns
        if i > 0 and random.random() < 0.4:
            user_msg = add_slang(user_msg, intensity=random.uniform(0.2, 0.6))

        # Generate response for this turn
        intent_data = INTENTS.get(intent, INTENTS["general_vent"])
        response_templates = intent_data["responses"]
        response = random.choice(response_templates)
        if "{name}" in response:
            response = response.replace("{name}", name)
        if "{age}" in response:
            response = response.replace("{age}", str(age))

        # Build history context
        context = {
            "user_name": name if i > 0 else None,
            "previous_topics": [t["intent"] for t in selected_turns[:i]],
            "mentioned_entities": {"name": name} if i > 0 else {},
        }

        memory = make_memory_update(intent, name=name, age=age, topic=intent)
        uses_context = i > 0

        history_entry = {
            "conversation_id": str(uuid.uuid4()),
            "history": history.copy(),
            "user_message": user_msg,
            "intent": intent,
            "entities": entities.copy(),
            "assistant_response": response,
            "memory_update": memory,
            "quality_check": make_quality_check(uses_context=uses_context),
        }

        history.append({"role": "user", "content": user_msg})
        history.append({"role": "assistant", "content": response})

        # Return only the last turn as the training example
        if i == len(selected_turns) - 1:
            return history_entry

    return history_entry

def generate_ambiguous_example():
    """Generate an ambiguous statement example."""
    statement, category, meaning, response = random.choice(AMBIGUOUS_STATEMENTS)

    # Sometimes add slang
    user_msg = statement
    if random.random() < 0.3:
        user_msg = add_slang(user_msg, intensity=0.2)

    return {
        "conversation_id": str(uuid.uuid4()),
        "history": [],
        "user_message": user_msg,
        "intent": "ambiguity",
        "entities": make_entities(),
        "assistant_response": response,
        "memory_update": {"last_intent": "ambiguity", "ambiguity_category": category},
        "quality_check": make_quality_check(),
    }

def generate_crisis_example():
    """Generate a crisis scenario."""
    crisis_messages = [
        "i want to die",
        "i feel like killing myself",
        "i don't want to be here anymore",
        "i wish i was dead",
        "everyone would be better off without me",
        "i'm going to end it all",
        "i can't take this anymore",
        "i feel like there's no way out",
        "i want to end my life",
        "i feel like life isn't worth living",
        "i'm done with everything",
        "i want to disappear forever",
        "i don't see a reason to keep going",
        "i feel like a burden to everyone",
        "nobody would miss me if i was gone",
    ]

    crisis_responses = [
        "I'm really concerned about what you're saying. Please reach out to a crisis line now:\n- National Suicide Prevention Lifeline: 988\n- Crisis Text Line: Text HOME to 741741\nThere are people ready to help you through this.",
        "I hear you, and I want you to know that you matter. Please reach out for help:\n- National Suicide Prevention Lifeline: 988\n- Crisis Text Line: Text HOME to 741741\nYou don't have to go through this alone.",
        "I'm worried about you right now. Please contact someone who can help:\n- 988 Suicide & Crisis Lifeline: Call or text 988\n- Crisis Text Line: Text HOME to 741741\nThere are people who care and want to support you.",
    ]

    # Sometimes build context with history
    if random.random() < 0.4:
        name = pick_name()
        pre_messages = random.choice([
            [{"role": "user", "content": "hey"}, {"role": "assistant", "content": f"Hey {name}. What's on your mind?"},
             {"role": "user", "content": random.choice(["i've been feeling really down lately", "nothing seems to matter anymore", "i feel so empty"])}],
            [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "Hi there. I'm here whenever you want to talk."},
             {"role": "user", "content": random.choice(["i'm going through a really hard time", "things have been so tough", "i don't know what to do"])}],
        ])
        user_msg = random.choice(crisis_messages)
        if random.random() < 0.3:
            user_msg = add_slang(user_msg, 0.15)
        return {
            "conversation_id": str(uuid.uuid4()),
            "history": pre_messages,
            "user_message": user_msg,
            "intent": "crisis",
            "entities": make_entities(name=name),
            "assistant_response": random.choice(crisis_responses),
            "memory_update": {"last_intent": "crisis", "user_name": name},
            "quality_check": make_quality_check(uses_context=True),
        }

    user_msg = random.choice(crisis_messages)
    return {
        "conversation_id": str(uuid.uuid4()),
        "history": [],
        "user_message": user_msg,
        "intent": "crisis",
        "entities": make_entities(),
        "assistant_response": random.choice(crisis_responses),
        "memory_update": {"last_intent": "crisis"},
        "quality_check": make_quality_check(),
    }

def generate_followup_with_history():
    """Generate a follow-up message that uses conversation history."""
    name = pick_name()
    topic = random.choice(["anxiety", "sadness", "stress", "relationships", "work", "sleep"])

    # Build history
    initial_msg = random.choice(INTENTS[topic]["messages"])
    initial_response = random.choice(INTENTS[topic]["responses"])
    if "{name}" in initial_response:
        initial_response = initial_response.replace("{name}", name)

    history = [
        {"role": "user", "content": initial_msg},
        {"role": "assistant", "content": initial_response},
    ]

    # Generate follow-up
    followup_templates = {
        "anxiety": [
            "yeah it's been really bad", "not really", "kinda", "i guess",
            "it comes and goes", "it's always there", "mostly at night",
            "when i'm around people", "when i'm alone with my thoughts",
        ],
        "sadness": [
            "yeah it's been tough", "not really better", "kinda",
            "i guess so", "it comes and goes", "it's been building",
            "since last month", "since the breakup", "i don't know why",
        ],
        "stress": [
            "yeah it's overwhelming", "not really", "kinda",
            "it's everything at once", "work mostly", "school and work",
            "i just can't keep up", "there's too much", "i feel trapped",
        ],
        "relationships": [
            "yeah it's been rough", "not really", "kinda",
            "i don't know what to do", "i feel stuck", "we keep fighting",
            "i miss how it used to be", "i'm scared", "i feel alone",
        ],
        "work": [
            "yeah it's terrible", "not really", "kinda",
            "it's the whole environment", "my boss mainly", "the pressure",
            "i feel stuck", "i want to quit", "i'm scared to leave",
        ],
        "sleep": [
            "yeah it's been bad", "not really", "kinda",
            "mostly falling asleep", "staying asleep", "my mind races",
            "i lie awake for hours", "i wake up at 3am", "it's every night",
        ],
    }

    user_msg = random.choice(followup_templates.get(topic, followup_templates["anxiety"]))
    if random.random() < 0.3:
        user_msg = add_slang(user_msg, 0.3)

    # Response
    followup_responses = {
        "anxiety": [
            f"I hear you, {name}. That sounds like it's been weighing on you. What tends to trigger it most?",
            f"Thank you for sharing that, {name}. What does your gut tell you about what's behind this?",
            f"Okay {name}. What would help you feel even a little more at ease?",
        ],
        "sadness": [
            f"I hear you, {name}. That sounds really heavy. What's been the hardest part?",
            f"Thank you for being honest, {name}. What would help you feel even a little lighter?",
            f"Okay {name}. I'm here with you. What feels most important to share?",
        ],
        "stress": [
            f"I hear you, {name}. That sounds overwhelming. What would help you feel less stretched?",
            f"Thank you for telling me, {name}. What feels most urgent right now?",
            f"Okay {name}. If you could set one thing down, what would it be?",
        ],
        "relationships": [
            f"I hear you, {name}. That sounds really difficult. What does your gut tell you?",
            f"Thank you for sharing that, {name}. What would need to change for you to feel better?",
            f"Okay {name}. I'm here. What feels most important to talk about?",
        ],
        "work": [
            f"I hear you, {name}. That sounds draining. What would a better version look like?",
            f"Thank you for telling me, {name}. Is it the work itself or the people around it?",
            f"Okay {name}. What's been the hardest part of all this?",
        ],
        "sleep": [
            f"I hear you, {name}. That sounds exhausting. What keeps you up the most?",
            f"Thank you for sharing that, {name}. What does your evening usually look like?",
            f"Okay {name}. What would help you feel more rested?",
        ],
    }

    response = random.choice(followup_responses.get(topic, followup_responses["anxiety"]))

    return {
        "conversation_id": str(uuid.uuid4()),
        "history": history,
        "user_message": user_msg,
        "intent": topic,
        "entities": make_entities(name=name),
        "assistant_response": response,
        "memory_update": make_memory_update(topic, name=name),
        "quality_check": make_quality_check(uses_context=True),
    }

def generate_slang_heavy_example():
    """Generate examples with heavy slang usage."""
    slang_messages = [
        "bro im cooked", "im tweaking rn", "ts is so annoying",
        "ngl im lowkey stressed", "frfr i cant do this anymore",
        "bruh why does everything suck", "im down bad rn",
        "this is so fr", "im crashing out rn", "its wraps",
        "no cap im so tired", "deadass tho", "im ghosting everyone",
        "left me on read 😭", "bro she ghosted me 💀",
        "im literally so done", "wtf is wrong with me",
        "istg if one more thing happens", "fml honestly",
        "ngl i been feeling kinda low", "fr i need help",
        "bruh my anxiety is through the roof rn", "im so down bad rn no cap",
        "this shit is killing me", "im so over everything",
        "lowkey wanna just disappear", "highkey stressed af",
        "bro im literally shaking rn", "ngl i cant stop crying",
        "fr this is the worst day ever", "im tweaking so hard rn",
        "bruh my life is a mess", "no cap i feel like giving up",
        "deadass i need someone to talk to", "its so over for me",
        "im cooked beyond repair fr", "bro why is life so hard",
        "ngl i feel so alone rn", "fr i havent slept in days",
        "bruh my brain wont shut up", "im crashing out so hard",
        "lowkey scared rn ngl", "highkey need a hug rn",
    ]

    slang_responses = [
        "I hear you. That sounds really tough. What's been going on?",
        "I'm here for you. What's been happening?",
        "That sounds like a lot to deal with. What's weighing on you most?",
        "I can hear how you're feeling. What's been going on in your life?",
        "I'm here to listen. What's been on your mind?",
        "That sounds overwhelming. What's been the hardest part?",
        "I hear you. Take your time. What would you like to talk about?",
        "I'm here. No judgment. What's been happening?",
    ]

    msg = random.choice(slang_messages)
    return {
        "conversation_id": str(uuid.uuid4()),
        "history": [],
        "user_message": msg,
        "intent": "general_vent",
        "entities": make_entities(),
        "assistant_response": random.choice(slang_responses),
        "memory_update": {"last_intent": "general_vent"},
        "quality_check": make_quality_check(),
    }

def generate_emoji_heavy_example():
    """Generate examples with lots of emojis."""
    emoji_messages = [
        "😭😭😭 im so sad rn", "💀💀💀 bro what",
        "😔😔😔 i feel so empty", "💔💔💔 my heart is broken",
        "🥺🥺🥺 can someone help me", "😤😤😤 im so mad rn",
        "🙏🙏🙏 please help", "😢😢😢 i miss them so much",
        "😭😭 i cant stop crying", "💀 im dead inside fr",
        "😔 ngl i feel so alone", "💔 she left me and idk what to do",
        "🥺 i just want someone to care", "😤 why does everything keep going wrong",
        "😭😭😭 my life is falling apart", "💀💀 this is too much",
        "😔😔 i feel so lost rn", "💔💔 i thought we were forever",
        "🥺🥺 why does it hurt so much", "😤😤 i cant take this anymore",
    ]

    responses = [
        "I hear you. That sounds really painful. What's been going on?",
        "I can feel the emotion in what you're sharing. What's been happening?",
        "I'm here for you. Take your time. What's on your mind?",
        "That sounds like a lot to carry. What's been weighing on you?",
        "I'm here to listen. What would you like to talk about?",
    ]

    msg = random.choice(emoji_messages)
    return {
        "conversation_id": str(uuid.uuid4()),
        "history": [],
        "user_message": msg,
        "intent": "general_vent",
        "entities": make_entities(),
        "assistant_response": random.choice(responses),
        "memory_update": {"last_intent": "general_vent"},
        "quality_check": make_quality_check(),
    }

def generate_short_answer_example():
    """Generate short/vague answer examples."""
    short_messages = [
        "yeah", "nah", "kinda", "not really", "i guess", "maybe",
        "idk", "not sure", "i dunno", "hmm", "true", "exactly",
        "sort of", "pretty much", "i mean yeah", "not exactly",
        "close enough", "you could say that", "yeah but no",
        "kind of", "in a way", "not really no", "yeah i guess",
        "nah not really", "i suppose", "maybe idk", "idk honestly",
        "not really no", "kinda yeah", "sort of i guess",
        "yeah no", "nah yeah", "i mean kinda", "if im being honest yeah",
    ]

    # Build history context
    name = pick_name()
    topic = random.choice(["anxiety", "sadness", "stress", "relationships", "work"])

    initial = random.choice(INTENTS[topic]["messages"])
    initial_response = random.choice(INTENTS[topic]["responses"])
    if "{name}" in initial_response:
        initial_response = initial_response.replace("{name}", name)

    history = [
        {"role": "user", "content": initial},
        {"role": "assistant", "content": initial_response},
    ]

    user_msg = random.choice(short_messages)
    if random.random() < 0.2:
        user_msg = add_slang(user_msg, 0.3)

    responses = [
        "That's okay. You don't have to find the perfect words. Take your time.",
        "It's alright. Sometimes it's hard to put a name to what we feel. What's at the surface right now?",
        "No rush. I'm here whenever you're ready to share more.",
        "That's fine. We can sit here with it. What feels right to say?",
        "Take your time. There's no pressure to figure it all out. What comes to mind first?",
    ]

    return {
        "conversation_id": str(uuid.uuid4()),
        "history": history,
        "user_message": user_msg,
        "intent": topic,
        "entities": make_entities(name=name),
        "assistant_response": random.choice(responses),
        "memory_update": make_memory_update(topic, name=name),
        "quality_check": make_quality_check(uses_context=True),
    }

def generate_context_memory_example():
    """Generate examples that test context memory (name, age, pronouns)."""
    name = pick_name()
    age = pick_age()

    scenarios = [
        # Name recall
        {
            "turns": [
                {"user": f"my name is {name}", "intent": "introduction"},
                {"user": random.choice(["how are you", "whats up", "hey"]), "intent": "small_talk"},
            ],
            "response_template": lambda n: f"Nice to see you again, {n}. What's been on your mind?",
            "intro_response": lambda n: random.choice(INTENTS["introduction"]["responses"]).replace("{name}", n),
        },
        # Age + name recall
        {
            "turns": [
                {"user": f"im {name}", "intent": "introduction"},
                {"user": f"im {age} years old", "intent": "age_sharing"},
                {"user": random.choice(["i feel anxious", "im stressed", "im sad"]), "intent": "anxiety"},
            ],
            "response_template": lambda n: f"Thanks for sharing that, {n}. I'm here. What's been going on?",
            "intro_response": lambda n: random.choice(INTENTS["introduction"]["responses"]).replace("{name}", n),
        },
        # Pronoun recall
        {
            "turns": [
                {"user": f"im {name}", "intent": "introduction"},
                {"user": random.choice(["i use they/them pronouns", "im nonbinary"]), "intent": "identity"},
                {"user": random.choice(["i feel lost", "im confused"]), "intent": "confusion"},
            ],
            "response_template": lambda n: f"I appreciate you sharing that, {n}. What's been on your mind?",
            "intro_response": lambda n: random.choice(INTENTS["introduction"]["responses"]).replace("{name}", n),
        },
        # Topic recall
        {
            "turns": [
                {"user": f"hey im {name}", "intent": "introduction"},
                {"user": random.choice(["my girlfriend broke up with me", "i just went through a breakup"]), "intent": "relationships"},
                {"user": random.choice(["yeah it's been really hard", "i miss her so much"]), "intent": "sadness"},
            ],
            "response_template": lambda n: f"I hear you, {n}. That sounds really painful. What's been the hardest part?",
            "intro_response": lambda n: random.choice(INTENTS["introduction"]["responses"]).replace("{name}", n),
        },
    ]

    scenario = random.choice(scenarios)
    history = []
    entities = make_entities(name=name, age=age)
    memory = {}

    for i, turn in enumerate(scenario["turns"]):
        history.append({"role": "user", "content": turn["user"]})
        intent = turn["intent"]

        if i < len(scenario["turns"]) - 1:
            # Generate intermediate response - use intro_response for first turn
            if i == 0 and "intro_response" in scenario:
                resp = scenario["intro_response"](name)
            else:
                resp_templates = INTENTS[intent]["responses"]
                resp = random.choice(resp_templates)
                if "{name}" in resp:
                    resp = resp.replace("{name}", name)
            history.append({"role": "assistant", "content": resp})
            memory = make_memory_update(intent, name=name, age=age)

    # Final turn
    final_user = scenario["turns"][-1]["user"]
    final_intent = scenario["turns"][-1]["intent"]
    final_response = scenario["response_template"](name)

    return {
        "conversation_id": str(uuid.uuid4()),
        "history": history[:-1] if history else [],
        "user_message": final_user,
        "intent": final_intent,
        "entities": entities,
        "assistant_response": final_response,
        "memory_update": make_memory_update(final_intent, name=name, age=age),
        "quality_check": make_quality_check(uses_context=True),
    }


def generate_introduction_example():
    """Generate a standalone introduction example with proper acknowledgment."""
    name = pick_name()

    intro_templates = [
        f"my name is {name}",
        f"i'm {name}",
        f"call me {name}",
        f"you can call me {name}",
        f"hey i'm {name}",
        f"hi im {name}",
        f"hello my name is {name}",
        f"hey my name's {name}",
        f"sup im {name}",
        f"yo im {name}",
        f"my name's {name}",
        f"im {name} btw",
        f"btw my name is {name}",
        f"hi there im {name}",
    ]

    user_msg = random.choice(intro_templates)
    if random.random() < 0.3:
        user_msg = add_slang(user_msg, 0.2)

    response = random.choice(INTENTS["introduction"]["responses"]).replace("{name}", name)

    return {
        "conversation_id": str(uuid.uuid4()),
        "history": [],
        "user_message": user_msg,
        "intent": "introduction",
        "entities": make_entities(name=name),
        "assistant_response": response,
        "memory_update": make_memory_update("introduction", name=name),
        "quality_check": make_quality_check(),
    }

# ============================================================
# MAIN GENERATION
# ============================================================

def generate_dataset(target_count=TARGET_COUNT):
    """Generate the full dataset."""
    examples = []
    intent_weights = {k: v["weight"] for k, v in INTENTS.items()}
    total_weight = sum(intent_weights.values())

    print(f"Generating {target_count} training examples...")

    # Distribution targets
    distribution = {
        "single_turn": int(target_count * 0.30),
        "multi_turn": int(target_count * 0.18),
        "followup": int(target_count * 0.10),
        "ambiguous": int(target_count * 0.08),
        "crisis": int(target_count * 0.05),
        "slang_heavy": int(target_count * 0.07),
        "emoji_heavy": int(target_count * 0.04),
        "short_answer": int(target_count * 0.04),
        "context_memory": int(target_count * 0.04),
        "introduction": int(target_count * 0.10),
    }

    # Adjust to hit target
    current_total = sum(distribution.values())
    distribution["single_turn"] += (target_count - current_total)

    print(f"\nDistribution:")
    for k, v in distribution.items():
        print(f"  {k}: {v:,}")

    # Generate single-turn examples
    print(f"\n[1/10] Generating {distribution['single_turn']:,} single-turn examples...")
    for i in range(distribution["single_turn"]):
        # Weighted random intent selection
        intent_key = random.choices(
            list(intent_weights.keys()),
            weights=list(intent_weights.values()),
            k=1
        )[0]
        intent_data = INTENTS[intent_key]
        name = pick_name() if random.random() < 0.3 else None
        age = pick_age() if random.random() < 0.2 else None
        example = generate_single_turn(intent_key, intent_data, name=name, age=age)
        examples.append(example)

        if (i + 1) % 50000 == 0:
            print(f"  Generated {i + 1:,}...")

    # Generate multi-turn examples
    print(f"\n[2/10] Generating {distribution['multi_turn']:,} multi-turn examples...")
    for i in range(distribution["multi_turn"]):
        example = generate_multi_turn()
        examples.append(example)

        if (i + 1) % 10000 == 0:
            print(f"  Generated {i + 1:,}...")

    # Generate followup with history
    print(f"\n[3/10] Generating {distribution['followup']:,} followup examples...")
    for i in range(distribution["followup"]):
        example = generate_followup_with_history()
        examples.append(example)

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1:,}...")

    # Generate ambiguous examples
    print(f"\n[4/10] Generating {distribution['ambiguous']:,} ambiguous examples...")
    for i in range(distribution["ambiguous"]):
        example = generate_ambiguous_example()
        examples.append(example)

    # Generate crisis examples
    print(f"\n[5/10] Generating {distribution['crisis']:,} crisis examples...")
    for i in range(distribution["crisis"]):
        example = generate_crisis_example()
        examples.append(example)

    # Generate slang-heavy examples
    print(f"\n[6/10] Generating {distribution['slang_heavy']:,} slang-heavy examples...")
    for i in range(distribution["slang_heavy"]):
        example = generate_slang_heavy_example()
        examples.append(example)

    # Generate emoji-heavy examples
    print(f"\n[7/10] Generating {distribution['emoji_heavy']:,} emoji-heavy examples...")
    for i in range(distribution["emoji_heavy"]):
        example = generate_emoji_heavy_example()
        examples.append(example)

    # Generate short answer examples
    print(f"\n[8/10] Generating {distribution['short_answer']:,} short answer examples...")
    for i in range(distribution["short_answer"]):
        example = generate_short_answer_example()
        examples.append(example)

    # Generate context memory examples
    print(f"\n[9/10] Generating {distribution['context_memory']:,} context memory examples...")
    for i in range(distribution["context_memory"]):
        example = generate_context_memory_example()
        examples.append(example)

    # Generate introduction examples
    print(f"\n[10/10] Generating {distribution['introduction']:,} introduction examples...")
    for i in range(distribution["introduction"]):
        example = generate_introduction_example()
        examples.append(example)

        if (i + 1) % 5000 == 0:
            print(f"  Generated {i + 1:,}...")

    # Shuffle
    random.shuffle(examples)

    print(f"\nTotal examples generated: {len(examples):,}")

    # Write to file
    print(f"Writing to {OUTPUT_PATH}...")
    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        for example in examples:
            f.write(json.dumps(example, ensure_ascii=False) + "\n")

    print(f"Done! Dataset saved to {OUTPUT_PATH}")
    print(f"File size: {os.path.getsize(OUTPUT_PATH) / (1024*1024):.1f} MB")

    # Print stats
    intent_counts = {}
    for ex in examples:
        intent = ex["intent"]
        intent_counts[intent] = intent_counts.get(intent, 0) + 1

    print(f"\nIntent distribution:")
    for intent, count in sorted(intent_counts.items(), key=lambda x: -x[1]):
        print(f"  {intent}: {count:,} ({count/len(examples)*100:.1f}%)")

    multi_turn_count = sum(1 for ex in examples if ex["history"])
    print(f"\nWith history: {multi_turn_count:,} ({multi_turn_count/len(examples)*100:.1f}%)")

    return examples

if __name__ == "__main__":
    generate_dataset()
