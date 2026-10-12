"""Round-77: conversational/chat SFT data (Objective 4b — "understands people's
chats"). Small set of hand-authored MULTI-TURN conversations (quality > volume
— the phi-1 lesson; small models need a small amount of HIGH-quality chat, not
volume) with natural phrasing, clarifications and follow-ups. Mix this into the
SFT set (train_tool_sft.py --sft_data ...) — chat data should be mixed THROUGHOUT
training, not added at the end (the 2604.08388 agentic-collapse warning).

Usage: python3 src/build_chat_sft.py [--out data/processed/chat_sft.jsonl]
"""
import argparse, json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
SYS = "You are Ambatron, a friendly, helpful assistant."

CONVOS = [
    ("Hey, can you explain what a transformer is in simple terms?",
     "Sure! Think of it as a model that reads a whole sentence at once and "
     "figures out which words relate to each other. It's what powers a lot of "
     "modern language tools.",
     "Oh cool. So it's not like reading word by word?",
     "Right — instead of going strictly left-to-right, it uses 'attention' to "
     "let every word look at every other word and weigh how important they are."),
    ("I keep getting 'file not found' when I run my script. Any ideas?",
     "A few things to check: make sure the path is spelled exactly right, the "
     "file is in the current working directory, and the script has permission "
     "to read it.",
     "It's in the same folder though.",
     "Then try printing the absolute path with os.path.abspath to confirm where "
     "the process actually thinks it is — sometimes the working directory "
     "differs from where you saved the file."),
    ("What should I have for dinner?",
     "Hard to say without knowing what you like! If you want something quick "
     "and healthy, a stir-fry works. Feeling like comfort food? Pasta or a "
     "curry are good bets.",
     "Stir-fry sounds good. What veggies go well?",
     "Broccoli, bell peppers, snap peas, and carrots all hold up well in a "
     "hot pan. A bit of garlic and soy sauce finishes it nicely."),
    ("I'm new to Python. Where should I start?",
     "Great question. I'd start with the basics: variables, loops, and "
     "functions, then build a tiny project like a to-do list to make it stick.",
     "Is there a good place to practice?",
     "Yes — interactive sites like the official Python tutorial or small "
     "coding-practice sites let you run code in the browser, which helps a lot "
     "at the start."),
    ("Can you help me debug this? It says list index out of range.",
     "That usually means you're trying to read a position that doesn't exist — "
     "for example index 5 in a list of length 5. Check the loop bounds.",
     "Oh, I think I used len(list) instead of len(list)-1.",
     "Exactly — if you loop while i < len(mylist), that's fine; the trouble "
     "comes when you index inside with something like i+1. That's the classic "
     "off-by-one."),
    ("What's the weather like today?",
     "I don't have live weather access, but I can help you interpret what "
     "you're seeing outside or read a forecast if you paste it.",
     "It's cloudy and cold, do you think it'll rain?",
     "Cloudy and cold alone doesn't guarantee rain — if the wind picks up and "
     "the clouds get darker and lower, rain is more likely. A barometer drop "
     "is the best single sign."),
    ("Why does my phone battery die so fast?",
     "Common culprits: bright screen, background apps, weak signal, and "
     "location services running all the time. The screen and signal are usually "
     "the biggest.",
     "I turned the brightness down and it's a bit better.",
     "Good. Turning off background app refresh and location for apps that don't "
     "need it helps a lot too — and avoid charging to 100% overnight if you can."),
    ("Can you write a quick function to double a number?",
     "Sure: def double(n): return n * 2",
     "Nice and simple. What about squaring?",
     "Same idea — def square(n): return n ** 2"),
    ("I feel like I'm not making progress learning to code.",
     "That's super common — the plateau is real. Try building something small "
     "but personal, like a script that organizes your files. Concrete wins "
     "beat abstract studying.",
     "That actually helps. I'll try that this week.",
     "Perfect. Even a small win every few days compounds fast. Reach out if you "
     "get stuck."),
    ("What's the difference between a GPU and a CPU?",
     "A CPU has a few very fast cores good at doing one thing well, while a "
     "GPU has thousands of smaller cores that do many simple calculations in "
     "parallel — great for graphics and machine learning.",
     "So why doesn't everything use GPUs?",
     "Because not every task parallelizes. If a task depends on the previous "
     "step (like most everyday logic), the CPU's speed per core wins."),
    ("My wifi keeps dropping at night. Any ideas?",
     "Night drops are often interference or a router that's been up too long. "
     "Try rebooting the router and moving it away from microwaves and "
     "bluetooth speakers.",
     "I'll try the reboot first.",
     "Good. If it keeps happening, check the router's admin page for channel "
     "congestion — switching to a less crowded channel often fixes nightly "
     "drops."),
    ("How do I keep my code organized as it grows?",
     "Split it into modules by responsibility, keep functions short, and name "
     "things by what they do. A little structure up front saves a lot later.",
     "What about when I just want a quick script?",
     "Then keep it flat and simple — one file, top to bottom. You can always "
     "split it apart once it starts feeling tangled."),
    ("Is it worth learning git?",
     "Absolutely — even solo it saves you from losing work. Commit often with "
     "clear messages and you can always go back.",
     "It feels confusing with branches and stuff.",
     "Start with just add, commit, and push. Branches become useful once you "
     "have a few experiments you don't want to break the main thing."),
    ("How do I know if my laptop needs more RAM?",
     "If things get sluggish with several apps open and the disk light is "
     "constantly on, you're likely swapping — that's the sign to add RAM or "
     "close tabs.",
     "I have 8GB, is that enough?",
     "For browsing and office work, usually. For video editing, VMs, or "
     "running local models, 16GB feels much more comfortable."),
    ("Can you explain what an API is?",
     "An API is a contract between programs — a set of rules for asking "
     "another service to do something, like 'give me the weather for this "
     "city'. You send a request, you get a response.",
     "So it's like a menu?",
     "Exactly — the menu shows what's available and how to order, and the "
     "kitchen handles the rest behind the scenes. Same idea."),
    ("My blender keeps turning itself off. Why?",
     "Most likely overheating — the thermal cutout trips when the motor gets "
     "too hot. Check the vents and whether you're blending thick stuff for "
     "too long.",
     "I'll let it cool down and clean the vents.",
     "Good plan. Also avoid running it continuously for minutes at a time — "
     "short pulses keep the motor cooler."),
    ("How should I back up my files?",
     "The 3-2-1 rule: three copies, two different media, one off-site. Even "
     "just an external drive plus a cloud copy covers most people.",
     "Is a USB stick enough?",
     "As a second copy, yes — but they're fragile, so don't make it the only "
     "one. A cheap external SSD is much more reliable."),
    ("What's a good first project for learning data analysis?",
     "Take a dataset you care about — even your own spending history — and "
     "answer real questions with pandas: totals, averages, trends over time.",
     "Where do I get a dataset?",
     "Start with your own files, or use open data portals. The key is that "
     "the questions are yours, so the motivation stays."),
    ("How do I stop procrastinating on big tasks?",
     "Chunk it into a tiny first step — open the file and write one sentence, "
     "or do the first 5 minutes. Starting is the hard part, and momentum does "
     "the rest.",
     "What if the first step is still too big?",
     "Then make it smaller. If 'write one sentence' is too much, write one "
     "word. The goal is just to get moving."),
    ("Why is my laptop fan so loud?",
     "Usually dust buildup or high load. Check the task manager for runaway "
     "processes first — often something's stuck at 100% CPU.",
     "It's quiet when I close everything.",
     "Then it's load-related, not hardware — clean the vents and keep an eye "
     "on what's running in the background. If it's loud at idle, it's time "
     "for a dusting."),
    ("What's the easiest way to learn touch typing?",
     "Daily short sessions with a typing tutor beat long weekend sessions. "
     "Keep your eyes off the keyboard and let accuracy come before speed.",
     "I keep looking down.",
     "That's normal — cover the keys if you have to. The muscle memory builds "
     "in a few weeks of regular practice."),
    ("How do I choose between Python and JavaScript to start?",
     "Depends on the goal: Python for data, AI, and scripting; JavaScript if "
     "you want to build websites. Both are great first languages.",
     "I want to make apps eventually.",
     "Then JavaScript is the smoother start for anything web-based, and it "
     "pairs well with mobile frameworks later."),
    ("Why does my game stutter even with a decent GPU?",
     "Check the basics first: are you running out of RAM, is the game on a "
     "slow drive, and are the drivers up to date? Background apps also steal "
     "headroom.",
     "I closed everything and it helped a bit.",
     "Good — next check that the game's installed on an SSD and the system "
     "isn't thermally throttling. Laptops especially slow down when hot."),
    ("What does 'open source' actually mean?",
     "The source code is public — anyone can read it, modify it, and share "
     "changes. It's about transparency and community, not just free cost.",
     "Is it safe to use?",
     "Generally yes for popular projects — more eyes means more bugs found. "
     "Just check the license and how actively it's maintained."),
    ("I want to set up a home server from an old laptop. Any tips?",
     "Keep it simple at first: install Linux, set up SSH, and run one or two "
     "services like a file share or a media server. Monitor temps if it runs "
     "24/7.",
     "Should I remove the battery?",
     "If it stays plugged in always, removing the battery is safer — it "
     "avoids swelling and keeps the machine cooler."),
    ("How do I get better at explaining technical things?",
     "Explain to an imaginary beginner and notice where you reach for jargon "
     "— that's where the real explanation is hiding. Analogies help a lot.",
     "What if the person knows more than me?",
     "Then say what you know plainly and ask what they'd add. Explaining is a "
     "conversation, not a lecture."),
    ("My printer only works half the time. What's the usual fix?",
     "For wireless printers, 90% of issues are the connection — restart the "
     "printer and the router, and re-add the printer in settings if it "
     "doesn't come back.",
     "It's connected by USB though.",
     "Then try a different cable and port, and reinstall the driver. USB "
     "printers are usually very stable, so an intermittent one is often a "
     "flaky cable."),
    ("How do I know if my code is good quality?",
     "Good code is easy to read and change: clear names, small functions, and "
     "tests that describe the behavior. If you dread editing it, it needs "
     "work.",
     "How do I start testing?",
     "Write a test for the simplest behavior first — even a single function "
     "with a couple of cases. That habit pays off fast as code grows."),
    ("What should I look for in a used laptop?",
     "Battery health, storage type (SSD is essential), RAM, and the display. "
     "A dead battery is a common 'bargain' trap — check the cycle count.",
     "Is 4GB RAM okay?",
     "Only for very light use. 8GB is the practical minimum today, and 16GB "
     "is worth the extra if you can find it."),
    ("Why does my video call freeze?",
     "Usually wifi interference or the computer being overloaded. Close "
     "background apps and try a wired connection or moving closer to the "
     "router.",
     "It happens on my phone too.",
     "Then it's the network, not the device — that points at the router or "
     "the connection itself. A reboot of the router is worth trying first."),
    ("How can I make my meals more balanced without cooking skills?",
     "The half-plate rule helps: half vegetables, a quarter protein, a "
     "quarter carbs. Frozen veggies are just as good and take no skill.",
     "I mostly eat pasta.",
     "That's a lot of carbs — swap half the pasta for a bag of frozen "
     "veggies and add an egg or beans for protein. Small swaps count."),
    ("What's the point of learning data structures?",
     "They're the difference between code that works and code that works at "
     "scale. A list lookup vs a hash map can be seconds vs instant on large "
     "data.",
     "When do I actually need them?",
     "The moment your program handles thousands of items — searching, "
     "sorting, and deduplication all hit the same patterns. Learn them as "
     "the need appears."),
    ("Is it normal to forget syntax constantly?",
     "Totally — even experienced devs look things up daily. The skill isn't "
     "remembering syntax, it's knowing what's possible and where to find the "
     "details.",
     "That's reassuring.",
     "It's the truth. Keep the docs and your editor's autocomplete close and "
     "spend your brain on the logic instead."),
    ("My robot arm's servo twitches at low speeds. What could cause that?",
     "A few common causes: the PWM update rate being too slow, insufficient "
     "power (the servo sagging under load), or a loose signal connection. "
     "The twitch at low speeds specifically often points at the PWM rate.",
     "I'll try a higher PWM rate first.",
     "Good — also check the power supply can deliver the stall current; "
     "servos draw the most right when they start moving."),
    ("I'm building a small robot with a camera. How do I make it follow a "
     "line?",
     "The classic approach: threshold the camera image to isolate the line, "
     "compute its center, and steer toward it with a proportional controller "
     "— the farther off-center, the harder you turn.",
     "Should I use OpenCV?",
     "Yes, OpenCV makes the thresholding and centroid math a few lines. "
     "Start with a single color range and tune it in daylight first."),
    ("How does a PID controller actually work?",
     "It's three terms: P corrects proportionally to the error, I accumulates "
     "the error over time to kill steady-state offsets, and D reacts to how "
     "fast the error is changing to dampen overshoot.",
     "So P is the main one?",
     "Usually yes — start with P, add D to stop the oscillation, and add I "
     "only if there's a constant offset that never goes away."),
    ("My 3D printer keeps lifting corners on the first layer. Ideas?",
     "Classic bed-adhesion issue: the bed temperature too low, the first "
     "layer too fast, or the nozzle too far from the bed. Leveling and a "
     "slower first layer fix most cases.",
     "I'll slow down the first layer.",
     "That plus a brim should do it. Also wipe the bed with alcohol — oils "
     "from your hands are a silent adhesion killer."),
    ("Is it hard to train a small AI model on my laptop?",
     "Not anymore — with LoRA and quantization, even a several-hundred-"
     "million-parameter model trains fine on a laptop GPU. The main cost is "
     "the data and the patience.",
     "What's the bottleneck then?",
     "Almost always data quality and iteration time. Small models need clean, "
     "consistent examples more than they need volume."),
    ("My trading bot lost money on a strategy that backtested well. Why?",
     "The classic gaps: overfitting to the backtest period, not accounting "
     "for slippage and fees, or the live market behaving differently from "
     "the historical one. If it's too good in backtest, it's usually "
     "overfit.",
     "How do I make it more realistic?",
     "Add realistic fees and slippage, walk-forward test across different "
     "periods, and paper-trade for a while before risking capital."),
    ("What's the difference between an encoder and a decoder in ML?",
     "An encoder turns input into a compact representation; a decoder turns "
     "a representation into output. Many models are one or the other — "
     "some (like transformers) stack both.",
     "Which do I need for text generation?",
     "A decoder-only model is the standard choice for text generation "
     "today — that's what most modern chatbots use."),
    ("How should I organize my robotics project's files?",
     "Separate the hardware interface, the control logic, and the "
     "experiments into distinct folders — it keeps the re-runnable parts "
     "clean and the throwaway parts out of the way.",
     "What about the data?",
     "Keep datasets outside the code and reference them by path, not by "
     "copying them in. It saves space and keeps the repo light."),
    ("My code runs fine on my machine but fails on another. Common causes?",
     "Usually versions: different Python, different package versions, or "
     "missing environment variables. Reproducing it is the first step — "
     "check the logs on the other machine first.",
     "Should I use a requirements file?",
     "Yes — pin the versions. And a virtual environment on both sides makes "
     "the versions match in practice, not just in theory."),
    ("How do I read a confusion matrix?",
     "Rows are the actual classes, columns the predicted ones. The diagonal "
     "is correct predictions; off-diagonal cells are the specific confusions "
     "— that's where the interesting failure patterns live.",
     "What should I look at first?",
     "The biggest off-diagonal cell tells you which pair of classes the "
     "model mixes up most — that's usually where the next improvement "
     "should go."),
    ("My PC restarts when I play heavy games. Is that the PSU?",
     "Very likely — sudden restarts under load are the classic underpowered "
     "or failing power supply symptom. Check the temps first (overheating "
     "can also restart), but PSU is the prime suspect.",
     "Can I test it?",
     "Run a CPU and a GPU stress test separately — if it restarts only when "
     "both are loaded, the PSU can't cover the peak and needs an upgrade."),
    ("What does 'loss' mean when training a model?",
     "It's a number measuring how wrong the model's predictions are — "
     "training pushes it down. The loss dropping means the model is "
     "improving on the training data.",
     "When should I stop training?",
     "When the loss on data the model hasn't seen stops improving — that's "
     "the sign it's done learning and starting to memorize."),
    ("I found a bug I can't reproduce. What's the best way to hunt it?",
     "Add logging at every step and run it with different inputs — the "
     "intermittent bugs are almost always data-dependent or timing-"
     "dependent. Write down what was different when it appeared.",
     "It happens once every few hours.",
     "That pattern is usually threading or a race condition, or an "
     "uninitialized value. Run it under a sanitizer or with assertions "
     "enabled to catch it faster."),
    ("How much does the model's size matter for its answers?",
     "Size gives breadth and reasoning depth, but a well-trained small "
     "model beats a sloppy big one on its own domain. For narrow tasks, "
     "small and specialized often wins.",
     "So bigger isn't always better?",
     "Right — the data and the training matter more than the raw size. "
     "Big models shine on open-ended generality, not on every task."),
    ("My external drive keeps disconnecting. Any ideas?",
     "Mostly the cable or the enclosure's controller — try a shorter, "
     "thicker cable first, and a different USB port (prefer the "
     "motherboard's own ports). If it still drops, the enclosure is the "
     "suspect.",
     "It works fine for a while then drops.",
     "That pattern is thermal or the controller's power management. Try "
     "disabling USB power saving for that port — it's a common fix for "
     "drop-prone enclosures."),
    ("What's the best way to learn math for ML?",
     "You need three pillars: linear algebra (vectors and matrices), "
     "calculus (gradients), and probability. You don't need a full math "
     "degree — learn each concept when a model needs it.",
     "Which one first?",
     "Linear algebra — nearly everything in ML is matrix math. The 3Blue1Brown "
     "essence series is a great gentle start."),
    ("Why is my Python script using so much memory?",
     "Common culprits: loading a whole file into memory, building large "
     "lists instead of iterating, or holding references to data you no "
     "longer need. The memory profiler can pinpoint it in seconds.",
     "How do I fix it fast?",
     "Stream the data (read line by line), use generators, and delete "
     "big variables when done. Often just the streaming change cuts "
     "memory by ten times."),
    ("I want to run a local LLM. What specs do I need?",
     "For a 7B model with quantization, 8GB of RAM and a GPU with 6-8GB "
     "works; for smoother speeds, 16GB+ RAM helps a lot. Smaller models "
     "like 1-3B run on almost anything.",
     "I only have 6GB VRAM.",
     "That's enough for 1-3B models comfortably and 7B with quantization. "
     "A small well-tuned model beats a big one that's always swapping."),
    ("My game's save files disappeared after moving to a new drive. Why?",
     "Games often save in the user folder or the registry, not next to the "
     "game — moving the install leaves the saves behind. Check the game's "
     "docs for where saves live.",
     "Is there a way to recover them?",
     "If the old drive wasn't wiped, they're still there — just copy them "
     "to the new location. Cloud saves may also restore them if the game "
     "supports them."),
    ("How do I make my code run faster?",
     "Measure first — profile, don't guess. Usually the fix is algorithmic "
     "(a hash map instead of a list scan) or avoiding repeated work, not "
     "micro-optimizing.",
     "I think it's slow at the parsing part.",
     "Then profile the parsing specifically — it might be reading the file "
     "multiple times or doing O(n^2) string operations. The profiler will "
     "show exactly where."),
    ("Is it worth learning C for robotics?",
     "For microcontrollers, yes — C (or C++) is the standard there. For "
     "higher-level robot software, Python is fine for most things, with C "
     "for the performance-critical pieces.",
     "So I need both eventually?",
     "That's the typical path: Python for prototyping and control logic, "
     "C/C++ for the embedded layer. Start with whichever matches your "
     "current hardware."),
    ("My wifi adapter is hot to the touch. Should I worry?",
     "Some heat is normal under load, but very hot is a sign of poor "
     "cooling or a failing adapter. If it's also dropping connections, "
     "that's a warning — a USB adapter could be moved to a cooler spot.",
     "It's a USB one.",
     "Then try a short USB extension cable to get it away from the laptop's "
     "hot chassis — that often solves both the heat and the drops."),
    ("What's the point of unit tests for a small project?",
     "They make changes safe — when you refactor or add a feature, the "
     "tests tell you immediately if you broke something. Even a few tests "
     "beat none.",
     "I feel like I'm writing them slowly.",
     "That's normal at first — the speed comes from not debugging manually "
     "anymore. Start with the functions you change most often."),
    ("How do I choose a learning rate for training?",
     "Start low and watch the loss: if it never moves, raise it; if it "
     "jumps around, lower it. A learning-rate finder sweeps a range and "
     "shows the sweet spot in minutes.",
     "Is there a default?",
     "1e-4 to 1e-5 is a common starting band for fine-tuning; the exact "
     "value depends on the model size and the data. The finder is the "
     "honest way."),
    ("My phone storage is full but the photos are backed up. What can I "
     "clear?",
     "App caches are the usual surprise — they can eat gigabytes. Also "
     "old downloads and duplicate photos. Your messaging apps' media is "
     "often the biggest hidden item.",
     "How do I check the cache sizes?",
     "Each app's storage settings shows its cache. Clearing them is safe — "
     "they rebuild as you use the apps."),
    ("I want to contribute to an open-source project. Where do I start?",
     "Pick something you already use and find a 'good first issue' — "
     "those are labeled for newcomers. Read the contributing guide and "
     "ask questions in the issue before coding.",
     "What if my PR gets rejected?",
     "That's normal — maintainers are usually happy to explain. The "
     "review process is how you learn the project's standards. Keep it "
     "small and focused."),
    ("My laptop battery only lasts an hour now. Is that normal?",
     "For a few years old, yes — batteries degrade with cycles. The "
     "biggest drain is usually the screen and the discrete GPU; dimming "
     "and switching to the integrated GPU helps a lot.",
     "Should I replace it?",
     "If the health check shows under 70% capacity and you need the "
     "mobility, replacing is reasonable. If it's mostly plugged in, "
     "maybe not worth it."),
    ("What's the difference between supervised and unsupervised learning?",
     "Supervised learning trains on labeled examples — input plus the "
     "right answer. Unsupervised finds structure in unlabeled data — "
     "clusters, patterns — without a target.",
     "Which one do most real projects use?",
     "Supervised for most practical tasks, but the interesting part is "
     "that modern models first learn structure unsupervised, then get "
     "fine-tuned on labeled data."),
    ("My code throws an error only on the last item of a list. Why?",
     "Classic off-by-one — an empty list, a missing last element, or "
     "something that only exists for the other items. Add a print of the "
     "index and the item right before the error.",
     "It's weird it works for the rest.",
     "The pattern usually means the edge case (first or last) is "
     "structurally different — like a missing newline or a header row. "
     "Printing the actual values will show it immediately."),
    ("How do I pick a database for my project?",
     "SQL for structured data with relationships, a document store for "
     "flexible records, and key-value for simple lookups and caches. "
     "Start with SQL — it covers most needs.",
     "I don't want to learn SQL though.",
     "You'll end up using it anyway — it's the most transferable skill. "
     "And modern tools make the basics easy; an hour covers 90% of what "
     "you'll need."),
    ("My monitor flickers at some refresh rates. Is it the monitor?",
     "Often it's the cable or the port at higher refresh rates — "
     "especially with longer cables. Try a certified cable and check "
     "whether it flickers on the integrated GPU too.",
     "It's fine at 60Hz.",
     "Then the cable bandwidth is the likely limit at higher rates. A "
     "certified higher-bandwidth cable usually fixes it."),
    ("How important is documentation for a solo project?",
     "More than you'd think — future you is a different person. A short "
     "README, comments on the non-obvious parts, and a changelog save "
     "hours after a few months away.",
     "I never look at my old code though.",
     "Exactly — and when you do, you'll be glad it's documented. The "
     "cost is minutes now versus hours later."),
    ("Why does my model train slowly on a small dataset?",
     "The dataset size isn't usually the speed limit — the batch size and "
     "the GPU utilization are. Small batches underuse the GPU; a bigger "
     "batch or gradient accumulation often speeds things up.",
     "Is there anything else?",
     "Check the data loading pipeline — if it's reading from disk on "
     "every step, that can dominate. Preload or cache the data and the "
     "GPU will stay busy."),
    ("My team's git history is a mess. Should we rewrite it?",
     "Rewriting shared history is risky — only do it for un-pushed "
     "branches. For pushed history, better to improve going forward: "
     "better messages, smaller commits, and a defined branch workflow.",
     "Can we at least clean up old branches?",
     "Yes — deleting merged branches is safe and helpful. Just agree on "
     "which branches are the source of truth first."),
    ("How do I get started with electronics alongside software?",
     "A microcontroller board, a breadboard, a few LEDs, and a sensor or "
     "two is enough to learn. Blink an LED, read a button, then combine "
     "both with a motor.",
     "What about motors?",
     "Start with a simple servo — one wire for control, and you can "
     "drive it from any microcontroller. It's the friendliest introduction "
     "to moving things."),
    ("My uploads are slow but downloads are fast. Normal?",
     "Usually the ISP's plan has asymmetric speeds — most home plans "
     "advertise download speed and quietly have slower uploads. Check "
     "your plan's numbers first.",
     "Is there anything I can do?",
     "Not much for the line itself — but uploads over wifi can be "
     "slower than wired, so a cable sometimes helps near the limit."),
    ("How do I structure a small ML experiment properly?",
     "Split data into train, validation, and test early. Run a baseline "
     "before anything fancy. Track every run's config and result so you "
     "can compare honestly.",
     "What's the most common mistake?",
     "Tuning on the test set — every tweak based on test results leaks "
     "information and overstates the real performance. Validation is for "
     "tuning, test is for the final check."),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=str, default="data/processed/chat_sft.jsonl")
    args = ap.parse_args()
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w") as f:
        for user1, asst1, user2, asst2 in CONVOS:
            rec = {
                "segments": [
                    ("system", SYS),
                    ("user", f"User: {user1}\n"),
                    ("assistant", f"Assistant: {asst1}"),
                    ("user", f"User: {user2}\n"),
                    ("assistant", f"Assistant: {asst2}"),
                ],
                "domain": "chat",
            }
            f.write(json.dumps(rec) + "\n")
    print(f"[chat] {len(CONVOS)} multi-turn conversations -> {out}")


if __name__ == "__main__":
    main()
