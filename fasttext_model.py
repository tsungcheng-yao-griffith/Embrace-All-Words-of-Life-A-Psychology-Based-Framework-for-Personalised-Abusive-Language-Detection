import fasttext #original facebook version

model = fasttext.train_supervised(input="", autotuneValidationFile="", autotuneDuration=36000) #building a new model
model.save_model("") #save model for the future use