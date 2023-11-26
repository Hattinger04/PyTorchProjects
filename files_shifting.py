import shutil

destination_dir = r"C:\Users\s8gre\Documents\Schule\KerasProjects\VoiceRecongnition\AI\data\speech_commands"
source_dir = r"C:\Users\s8gre\Documents\Schule\5BHWII\HCIN\VoiceRecognition\AI\data\SpeechCommands\speech_commands_v0.02"
shutil.copytree(source_dir, destination_dir)