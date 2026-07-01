from counselor import Counselor
bot = Counselor()

print("=== GREET ===")
print(bot.greet())

print("\n=== NAME ===")
print(bot.respond("Alex"))

print("\n=== ANXIETY ===")
print(bot.respond("I have an interview tomorrow and I feel really anxious about it"))

print("\n=== SADNESS ===")
print(bot.respond("I feel so sad about my breakup"))

print("\n=== FOLLOW-UP ===")
bot.respond("It happened two weeks ago and I cannot stop thinking about her")
print(bot.respond("It happened two weeks ago and I cannot stop thinking about her"))

print("\n=== STRESS ===")
print(bot.respond("I have too much work and I feel stressed and overwhelmed"))

print("\n=== CRISIS ===")
print(bot.respond("I want to die"))

print("\n=== POSITIVE ===")
print(bot.respond("I just got a promotion and I feel great"))
