def setup():

    configuration = [False, False, False, False]
    
    
    user_utterance = ""
    while user_utterance != "0":
        message = f"System: Hello, please choose your configuration:\n" \
          f"Start: 0\n" \
          f"Text to speech: 1 \t status: {configuration[0]}\n" \
          f"Response delay: 2 \t status {configuration[1]}\n" \
          f"User feedback: 3 \t status: {configuration[2]}\n" \
          f"Allow user not: 4 \t status: {configuration[3]}"
        print(message)
        user_utterance = input("Config choice: ")
        switch = int(user_utterance)
        if switch <= 4 and switch >= 1:
            configuration[switch-1] = not configuration[switch-1]

    return configuration