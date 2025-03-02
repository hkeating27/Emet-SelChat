# Emet-SelChat
This is a chatbot simulating the character of Emet-Selch made using machine learning and AI (PyTorch and GPT-2 Large). These files create the bot first, since it's ~40GB which Github won't allow.  
Note: You want some beefy hardware to run the training as it implements GPT Large. With 32 GB of RAM and a RTX 4070 Ti Super it still took a half-hour to train. 16GB and a 4060 are the recommended minimum.

Here are the instructions to run it:
Step 1.) Open the command line/terminal and go to the directory with the two files
Step 1.) To train, run "python script.py". You may need to use "pip install torch" or "pip install transformers" first.  
This will automatically make a new folder in the same directory script.py is that contains the model. Everything needs to be within the same folder to run.  
Step 2.) To chat with Emet-Selch, run "python run_emet_selch.py".  
Step 3.) Use "quit" or "exit" to return to the command line when you can't stand his ego anymore
