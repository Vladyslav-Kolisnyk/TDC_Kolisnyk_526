import speech_recognition as srec

def recognize_speech(rec, mic):
    with mic as source:
        rec.adjust_for_ambient_noise(source)
        audio = rec.listen(source)
    result = {"Текст": None}
    result["Текст"] = rec.recognize_google(audio, show_all=False, language="uk-UA")
    return result

if __name__ == "__main__":
    recognizer = srec.Recognizer()
    mic = srec.Microphone()
    while True:
        result = recognize_speech(recognizer, mic)
        print('Ви сказали: \n{}'.format(result['Текст']))
