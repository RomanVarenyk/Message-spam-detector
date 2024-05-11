import main
new_message = input("What message would you like to check? ")
label, confidence = main.predict_message(main.model, main.vectorizer, new_message)
print(f"The message is predicted as {label} with {confidence * 100:.2f}% confidence.")